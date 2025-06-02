"""
Database connection and query utilities.
This module provides asynchronous database access for the publication tracking database.
"""

import json
import logging
from pathlib import Path
from typing import Any

import aiosqlite
from fastapi import HTTPException

from backend.service.models import PublicationStatus, PublicationStatusFilter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define database path
PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "publication_tracking.db"


async def get_db_connection():
    """Get an async connection to the SQLite database"""
    try:
        conn = await aiosqlite.connect(DB_PATH)
        conn.row_factory = aiosqlite.Row  # Return rows as dictionaries
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection error") from e


async def close_db_connection(conn):
    """Close the database connection"""
    await conn.close()


def parse_json_field(value: str) -> list[str]:
    """Parse JSON array string to a Python list"""
    if not value:
        return []
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return []


def build_query_params(filters: PublicationStatusFilter) -> tuple[str, dict[str, Any]]:
    """
    Build SQL WHERE clause and parameters based on provided filters

    Args:
        filters: Filter criteria for publication status

    Returns:
        Tuple containing (WHERE clause string, parameters dictionary)
    """
    wheres = []
    params = {}

    # Filter by status
    if filters.download_status:
        wheres.append("download_status = :download_status")
        params["download_status"] = filters.download_status.value

    if filters.processing_status:
        wheres.append("processing_status = :processing_status")
        params["processing_status"] = filters.processing_status.value

    if filters.embedding_status:
        wheres.append("embedding_status = :embedding_status")
        params["embedding_status"] = filters.embedding_status.value

    if filters.ingestion_status:
        wheres.append("ingestion_status = :ingestion_status")
        params["ingestion_status"] = filters.ingestion_status.value

    # Filter by date range
    if filters.start_date:
        wheres.append("last_updated >= :start_date")
        params["start_date"] = filters.start_date.isoformat()

    if filters.end_date:
        wheres.append("last_updated <= :end_date")
        params["end_date"] = filters.end_date.isoformat()

    # Filter by title
    if filters.title_contains:
        wheres.append("title LIKE :title_pattern")
        params["title_pattern"] = f"%{filters.title_contains}%"  # Filter by year
    if filters.year:
        wheres.append("year = :year")
        params["year"] = str(filters.year)  # Convert int to str for SQLite parameter

    # Combine all WHERE clauses
    where_clause = " AND ".join(wheres) if wheres else "1=1"

    return where_clause, params


async def get_publication_statuses(
    filters: PublicationStatusFilter,
) -> tuple[list[PublicationStatus], int]:
    """
    Query publication status data with filtering, sorting and pagination

    Args:
        filters: Filter, sort and pagination criteria

    Returns:
        Tuple: (list of objects, total count of records matching filter)
    """
    conn = await get_db_connection()
    try:
        # Build WHERE clause and parameters based on filters
        where_clause, params = build_query_params(filters)

        # Validation for sort column (prevent SQL injection)
        valid_sort_columns = [
            "publication_id",
            "title",
            "authors",
            "year",
            "download_status",
            "processing_status",
            "embedding_status",
            "ingestion_status",
            "discovery_timestamp",
            "last_updated",
        ]

        sort_column = (
            filters.sort_by if filters.sort_by in valid_sort_columns else "last_updated"
        )
        sort_order = "ASC" if filters.sort_order.value == "asc" else "DESC"

        # Calculate pagination offsets
        offset = (filters.page - 1) * filters.page_size

        # Get total count matching filter
        count_query = f"SELECT COUNT(*) as count FROM \
                publication_tracking WHERE {where_clause}"
        async with conn.execute(count_query, params) as cursor:
            total = await cursor.fetchone()
            total_count = total["count"] if total else 0

        # Query data with pagination
        query = f"""
            SELECT *
            FROM publication_tracking
            WHERE {where_clause}
            ORDER BY {sort_column} {sort_order}
            LIMIT :limit OFFSET :offset
        """

        query_params = {**params, "limit": filters.page_size, "offset": offset}

        publications = []
        async with conn.execute(query, query_params) as cursor:
            rows = await cursor.fetchall()

            for row in rows:
                # Convert sqlite Row to dict
                row_dict = dict(row)

                # Parse JSON fields
                if row_dict.get("file_urls"):
                    row_dict["file_urls"] = parse_json_field(row_dict["file_urls"])

                # Convert to PublicationStatus model
                publication = PublicationStatus.model_validate(row_dict)
                publications.append(publication)

        return publications, total_count

    except Exception as e:
        logger.error(f"Database query error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Database query error: {str(e)}"
        ) from e
    finally:
        await close_db_connection(conn)


async def get_publication_by_id(publication_id: str) -> PublicationStatus | None:
    """
    Get a single publication by its ID

    Args:
        publication_id: The unique identifier of the publication

    Returns:
        PublicationStatus object if found, None otherwise
    """
    conn = await get_db_connection()
    try:
        query = "SELECT * FROM publication_tracking WHERE publication_id = ?"
        async with conn.execute(query, (publication_id,)) as cursor:
            row = await cursor.fetchone()

            if not row:
                return None

            # Convert sqlite Row to dict
            row_dict = dict(row)

            # Parse JSON fields
            if row_dict.get("file_urls"):
                row_dict["file_urls"] = parse_json_field(row_dict["file_urls"])

            # Convert to PublicationStatus model
            return PublicationStatus.model_validate(row_dict)

    except Exception as e:
        logger.error(f"Database query error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Database query error: {str(e)}"
        ) from e
    finally:
        await close_db_connection(conn)
