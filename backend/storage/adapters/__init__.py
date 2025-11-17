"""
Database adapters for publication tracking.

This package provides database adapters that abstract away the specific database
implementation from the business logic. The adapter pattern allows the ETL pipeline
and API services to work with different database backends (SQLite, Supabase, etc.)
without changing the core logic.
"""

from backend.storage.adapters.base import DatabaseAdapter
from backend.storage.adapters.factory import (
    create_and_initialize_adapter,
    get_database_adapter,
)
from backend.storage.adapters.supabase_adapter import SupabaseAdapter

__all__ = [
    "DatabaseAdapter",
    "SupabaseAdapter",
    "get_database_adapter",
    "create_and_initialize_adapter",
]
