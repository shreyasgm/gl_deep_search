import logging
import os
from typing import Any, TypeVar

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from ..models.publications import Base as PublicationsBase
from ..models.tracking import Base as TrackingBase

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic model types
T = TypeVar("T")


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
                "primary_key": column["primary_key"],
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
