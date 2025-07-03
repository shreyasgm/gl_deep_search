"""
Pydantic models for API request and response schema definition.
This module contains models for publication status endpoints.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class DownloadStatus(str, Enum):
    """
    Represents the current download status of a publication document.
    - PENDING: Document is queued for download but not started yet
    - IN_PROGRESS: Download has started but not completed
    - DOWNLOADED: Document has been successfully downloaded
    - FAILED: Download attempt failed after all retries
    """

    PENDING = "Pending"
    IN_PROGRESS = "In Progress"
    DOWNLOADED = "Downloaded"
    FAILED = "Failed"


class ProcessingStatus(str, Enum):
    """
    Represents the current processing status of a publication document.
    - PENDING: Document is queued for processing but not started yet
    - IN_PROGRESS: Processing has started but not completed
    - PROCESSED: Document has been successfully processed
    - OCR_FAILED: OCR processing failed
    - CHUNKING_FAILED: Chunking of document failed
    - FAILED: Processing attempt failed after all retries
    """

    PENDING = "Pending"
    IN_PROGRESS = "In Progress"
    PROCESSED = "Processed"
    OCR_FAILED = "OCR_Failed"
    CHUNKING_FAILED = "Chunking_Failed"
    FAILED = "Failed"


class EmbeddingStatus(str, Enum):
    """Represents the current embedding status of a publication document.
    - PENDING: Document is queued for embedding but not started yet
    - IN_PROGRESS: Embedding has started but not completed
    - EMBEDDED: Document has been successfully embedded
    - FAILED: Embedding attempt failed after all retries
    """

    PENDING = "Pending"
    IN_PROGRESS = "In Progress"
    EMBEDDED = "Embedded"
    FAILED = "Failed"


class IngestionStatus(str, Enum):
    """
    Represents the current ingestion status of a publication document.
    - PENDING: Document is queued for ingestion but not started yet
    - IN_PROGRESS: Ingestion has started but not completed
    - INGESTED: Document has been successfully ingested into the database
    - FAILED: Ingestion attempt failed after all retries
    """

    PENDING = "Pending"
    IN_PROGRESS = "In Progress"
    INGESTED = "Ingested"
    FAILED = "Failed"


class SortOrder(str, Enum):
    """Represents the sort order for publication queries.
    - ASC: Ascending order
    - DESC: Descending order
    """

    ASC = "asc"
    DESC = "desc"


class PublicationStatusFilter(BaseModel):
    """
    Filter parameters for querying the publication tracking system.

    Use parameters to search, filter, and paginate through the publication database.
    All filters are optional and can be combined for more specific queries.
    Date filters use ISO 8601 format (YYYY-MM-DDThh:mm:ssZ).
    """

    page: int = Field(
        default=1, ge=1, description="Page number for pagination, starting from 1"
    )
    page_size: int = Field(
        default=10, ge=1, le=100, description="Number of results per page (max 100)"
    )
    download_status: DownloadStatus | None = Field(
        None, description="Filter by download (e.g., Downloaded, Failed, etc.)"
    )
    processing_status: ProcessingStatus | None = None
    embedding_status: EmbeddingStatus | None = None
    ingestion_status: IngestionStatus | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    sort_by: str = Field(default="last_updated", description="Field to sort by")
    sort_order: SortOrder = Field(default=SortOrder.DESC, description="Sort order")
    title_contains: str | None = None
    year: int | None = None


class PublicationStatus(BaseModel):
    """
    Detailed status information for a tracked publication.

    This model contains all metadata and processing status information for a document
    in the publication tracking system. Each document progresses through multiple
    processing stages (download, processing, embedding, ingestion), and this model
    tracks the status, timestamp, and attempt count for each stage.
    """

    publication_id: str
    title: str | None = None
    authors: str | None = None
    year: int | None = None
    abstract: str | None = None
    source_url: str
    file_urls: list[str] | None = None

    # Status tracking
    download_status: DownloadStatus
    download_timestamp: datetime | None = None
    download_attempt_count: int

    processing_status: ProcessingStatus
    processing_timestamp: datetime | None = None
    processing_attempt_count: int

    embedding_status: EmbeddingStatus
    embedding_timestamp: datetime | None = None
    embedding_attempt_count: int

    ingestion_status: IngestionStatus
    ingestion_timestamp: datetime | None = None
    ingestion_attempt_count: int

    discovery_timestamp: datetime
    last_updated: datetime
    error_message: str | None = None

    class Config:
        """Pydantic model configuration"""

        from_attributes = True  # For SQLAlchemy model compatibility


class PublicationStatusResponse(BaseModel):
    """
    Paginated response containing publication status records.

    This response includes pagination metadata (total count, current page, page size)
    along with the matching publication status records. If no records match the filter
    criteria, the 'items' list will be empty but the response will still be returned
    with a 200 status code.
    """

    total: int
    page: int
    page_size: int
    items: list[PublicationStatus]
