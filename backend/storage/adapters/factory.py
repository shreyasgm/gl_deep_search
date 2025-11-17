"""
Factory pattern for creating database adapters.

This module provides a factory function that creates the appropriate database
adapter based on configuration (environment variables). This allows the ETL
pipeline and API services to work with different database backends without
changing business logic.

Supported adapters:
- "supabase": SupabaseAdapter (PostgreSQL via Supabase)
- "sqlite": SQLiteAdapter (local SQLite database) - TODO: Not yet implemented

Usage:
    # In ETL pipeline or API service
    adapter = get_database_adapter()
    await adapter.initialize()

    # Use adapter methods
    publications = await adapter.get_publications_for_download()
"""

import logging
import os

from backend.storage.adapters.base import DatabaseAdapter
from backend.storage.adapters.supabase_adapter import SupabaseAdapter

logger = logging.getLogger(__name__)


def get_database_adapter(adapter_type: str | None = None) -> DatabaseAdapter:
    """
    Factory function to create the appropriate database adapter.

    The adapter type is determined by:
    1. The adapter_type parameter (if provided)
    2. The DATABASE_ADAPTER environment variable
    3. Falls back to "supabase" as default

    Args:
        adapter_type: Optional adapter type override ("supabase" or "sqlite")

    Returns:
        DatabaseAdapter instance (not yet initialized)

    Raises:
        ValueError: If adapter_type is unsupported
        ValueError: If required environment variables are missing

    Example:
        >>> adapter = get_database_adapter()
        >>> await adapter.initialize()
        >>> publications = await adapter.get_publications_for_download()
    """
    # Determine adapter type from parameter or environment
    adapter = adapter_type or os.environ.get("DATABASE_ADAPTER", "supabase")

    logger.info(f"Creating database adapter: {adapter}")

    if adapter == "supabase":
        # Supabase adapter requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY
        return SupabaseAdapter()

    elif adapter == "sqlite":
        # SQLite adapter not yet implemented
        # When implemented, it should work with local SQLite database
        # for development and testing
        raise NotImplementedError(
            "SQLite adapter not yet implemented. "
            "Use DATABASE_ADAPTER=supabase or implement SQLiteAdapter."
        )

    else:
        raise ValueError(
            f"Unsupported database adapter: {adapter}. "
            f"Supported adapters: supabase, sqlite"
        )


async def create_and_initialize_adapter(
    adapter_type: str | None = None,
) -> DatabaseAdapter:
    """
    Convenience function to create and initialize adapter in one step.

    This is a helper function that combines adapter creation and initialization.
    Useful for simple scripts and testing.

    Args:
        adapter_type: Optional adapter type override

    Returns:
        Initialized DatabaseAdapter instance

    Raises:
        ValueError: If adapter_type is unsupported
        ConnectionError: If initialization fails

    Example:
        >>> adapter = await create_and_initialize_adapter()
        >>> publications = await adapter.get_publications_for_download()
        >>> await adapter.close()
    """
    adapter = get_database_adapter(adapter_type)
    await adapter.initialize()
    return adapter
