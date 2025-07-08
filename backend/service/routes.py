"""
API routes for publication tracking status.
This module defines the FastAPI endpoints for querying publication status.
"""

from fastapi import APIRouter, HTTPException

from backend.service.database import get_publication_by_id, get_publication_statuses
from backend.service.models import (
    DownloadStatus,
    EmbeddingStatus,
    IngestionStatus,
    ProcessingStatus,
    PublicationStatusFilter,
    PublicationStatusResponse,
    SortOrder,
)

# Create router
router = APIRouter(tags=["publications"])


@router.get("/publications/status", response_model=PublicationStatusResponse)
async def get_publications_status(
    page: int = 1,
    page_size: int = 10,
    download_status: DownloadStatus | None = None,
    processing_status: ProcessingStatus | None = None,
    embedding_status: EmbeddingStatus | None = None,
    ingestion_status: IngestionStatus | None = None,
    sort_by: str = "last_updated",
    sort_order: SortOrder = SortOrder.DESC,
    title_contains: str | None = None,
    year: int | None = None,
):
    """
    Get paginated list of publication statuses with filtering options

    Query parameters can be used to filter results by status, date, title, etc.
    Results are paginated and can be sorted by specified field.
    """
    # Validate parameters
    if page < 1:
        page = 1
    if page_size < 1 or page_size > 100:
        page_size = min(max(1, page_size), 100)

    # Create filter object from query parameters
    filters = PublicationStatusFilter(
        page=page,
        page_size=page_size,
        download_status=download_status,
        processing_status=processing_status,
        embedding_status=embedding_status,
        ingestion_status=ingestion_status,
        sort_by=sort_by,
        sort_order=sort_order,
        title_contains=title_contains,
        year=year,
    )

    # Query publications with filters
    publications, total_count = await get_publication_statuses(filters)

    # Return paginated response
    return PublicationStatusResponse(
        total=total_count, page=page, page_size=page_size, items=publications
    )


@router.get("/publications/status/{publication_id}")
async def get_publication_status(publication_id: str):
    """
    Get status details for a specific publication

    Args:
        publication_id: Unique identifier of the publication

    Returns:
        Complete publication status record
    """
    publication = await get_publication_by_id(publication_id)

    if not publication:
        raise HTTPException(
            status_code=404, detail=f"Publication with ID {publication_id} not found"
        )

    return publication
