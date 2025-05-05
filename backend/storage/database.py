"""
Database connection and management utilities
"""

import logging
import os
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get database connection string from environment variable or use default SQLite
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    f"sqlite:///{Path(__file__).parent}/data/etl_tracking.db"
)

# Create engine
engine = create_engine(
    DATABASE_URL, 
    echo=False,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


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


def init_db(schema_path: Optional[Path] = None):
    """Initialize the database with the defined schema"""
    # Create SQLModel tables
    SQLModel.metadata.create_all(engine)
    
    # Execute the SQL schema if provided
    if schema_path:
        try:
            with open(schema_path, "r") as f:
                sql_schema = f.read()
                
            # Execute the schema statements individually
            with engine.connect() as conn:
                # Split statements by semicolon and execute individually
                statements = [stmt.strip() for stmt in sql_schema.split(';') if stmt.strip()]
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
        logger.warning(f"Schema file not found at {schema_path}, created tables from SQLModel only")


# Initialize the database when the module is imported
if os.getenv("AUTO_INIT_DB", "true").lower() in ("true", "1", "yes"):
    ensure_db_initialized()