"""
Database connection and management utilities
"""

import logging
import os
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlmodel import SQLModel

from backend.etl.models.publications import Base as PublicationsBase
from backend.etl.models.tracking import SQLModel as TrackingBase

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get database connection string from environment variable or use default SQLite
DATABASE_URL = os.getenv(
    "DATABASE_URL", f"sqlite:///{Path(__file__).parent}/data/etl_tracking.db"
)

# Create engine
engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False}
    if DATABASE_URL.startswith("sqlite")
    else {},
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Type variable for generic model types
T = TypeVar("T")


# Enable foreign key constraints for SQLite
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable foreign key support for SQLite"""
    if DATABASE_URL.startswith("sqlite"):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def get_db_session():
    """Dependency to get a database session"""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db(schema_path: Path | None = None):
    """Initialize the database with the defined schema"""
    # Create SQLModel tables
    SQLModel.metadata.create_all(engine)

    # Execute the SQL schema if provided
    if schema_path:
        try:
            with open(schema_path) as f:
                sql_schema = f.read()

            # Execute the schema statements individually
            with engine.connect() as conn:
                # Split statements by semicolon and execute individually
                statements = [
                    stmt.strip() for stmt in sql_schema.split(";") if stmt.strip()
                ]
                for statement in statements:
                    if statement:
                        conn.execute(text(statement))
                conn.commit()

            logger.info(f"Successfully initialized database from schema: {schema_path}")
        except Exception as e:
            logger.error(f"Error initializing database from schema: {e}")
            raise

    logger.info("Database initialization complete")


def ensure_db_initialized():
    """Ensure the database is initialized with required tables"""
    # Create data directory if using SQLite
    if DATABASE_URL.startswith("sqlite"):
        db_path = Path(DATABASE_URL.replace("sqlite:///", ""))
        os.makedirs(db_path.parent, exist_ok=True)

    # Initialize database with schema
    schema_path = Path(__file__).parent / "etl_metadata_schema.sql"
    if schema_path.exists():
        init_db(schema_path)
    else:
        # Just create the SQLModel tables if no schema file
        init_db()
        logger.warning(
            f"Schema file not found at {schema_path}, created tables from SQLModel only"
        )


# Initialize the database when the module is imported
if os.getenv("AUTO_INIT_DB", "true").lower() in ("true", "1", "yes"):
    ensure_db_initialized()


class Database:
    """Database connection and session management"""

    def __init__(self, db_url: str | None = None):
        """Initialize database connection"""
        self.db_url = db_url or os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError(
                "Database URL not provided and not found in environment variables"
            )

        self.engine = create_engine(self.db_url)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database tables"""
        try:
            # Create all tables
            TrackingBase.metadata.create_all(bind=self.engine)
            PublicationsBase.metadata.create_all(bind=self.engine)
            logger.info("Database tables initialized successfully")
        except SQLAlchemyError as e:
            logger.error(f"Error initializing database tables: {e}")
            raise

    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()

    def get_or_create(
        self,
        session: Session,
        model: type[T],
        defaults: dict[str, Any] | None = None,
        **kwargs,
    ) -> tuple[T, bool]:
        """Get an existing record or create a new one"""
        instance = session.query(model).filter_by(**kwargs).first()
        if instance:
            return instance, False

        if defaults:
            kwargs.update(defaults)
        instance = model(**kwargs)
        session.add(instance)
        session.commit()
        session.refresh(instance)
        return instance, True

    def bulk_create(
        self, session: Session, model: type[T], objects: list[dict[str, Any]]
    ) -> list[T]:
        """Bulk create records"""
        instances = [model(**obj) for obj in objects]
        session.bulk_save_objects(instances)
        session.commit()
        return instances

    def bulk_update(
        self,
        session: Session,
        model: type[T],
        objects: list[dict[str, Any]],
        id_field: str,
    ) -> None:
        """Bulk update records"""
        for obj in objects:
            instance = (
                session.query(model).filter_by(**{id_field: obj[id_field]}).first()
            )
            if instance:
                for key, value in obj.items():
                    setattr(instance, key, value)
        session.commit()

    def get_by_id(self, session: Session, model: type[T], id_value: Any) -> T | None:
        """Get a record by ID"""
        return session.query(model).filter_by(id=id_value).first()

    def get_all(self, session: Session, model: type[T]) -> list[T]:
        """Get all records of a model"""
        return session.query(model).all()

    def delete(self, session: Session, model: type[T], id_value: Any) -> bool:
        """Delete a record by ID"""
        instance = self.get_by_id(session, model, id_value)
        if instance:
            session.delete(instance)
            session.commit()
            return True
        return False

    def exists(self, session: Session, model: type[T], **kwargs) -> bool:
        """Check if a record exists"""
        return session.query(model).filter_by(**kwargs).first() is not None

    def count(self, session: Session, model: type[T], **kwargs) -> int:
        """Count records matching criteria"""
        query = session.query(model)
        if kwargs:
            query = query.filter_by(**kwargs)
        return query.count()

    def get_table_names(self) -> list[str]:
        """Get list of all table names"""
        inspector = inspect(self.engine)
        return inspector.get_table_names()

    def get_column_info(self, table_name: str) -> list[dict[str, Any]]:
        """Get column information for a table"""
        inspector = inspect(self.engine)
        return [
            {
                "name": column["name"],
                "type": str(column["type"]),
                "nullable": column["nullable"],
                "default": column["default"],
                "primary_key": column.get("primary_key", False),
            }
            for column in inspector.get_columns(table_name)
        ]

    def execute_raw_sql(
        self, session: Session, sql: str, params: dict[str, Any] | None = None
    ) -> Any:
        """Execute raw SQL query"""
        try:
            result = session.execute(sql, params or {})
            session.commit()
            return result
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error executing raw SQL: {e}")
            raise

    def close(self) -> None:
        """Close database connection"""
        self.engine.dispose()
        logger.info("Database connection closed")
