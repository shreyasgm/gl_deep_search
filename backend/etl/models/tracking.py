"""
Models for ETL pipeline tracking

This module defines the database schema and data models for tracking publications
through the ETL pipeline stages: discovery, download, processing, embedding,
and ingestion.
"""

import datetime
import json
from enum import Enum

import sqlalchemy as sa
from pydantic import field_validator
from sqlmodel import Field, MetaData, SQLModel

# Create metadata instance for SQLAlchemy table definitions
metadata = MetaData()


class DownloadStatus(str, Enum):
    """
    Enum defining possible states for the publication download stage.

    PENDING: Publication identified but download not yet started
    IN_PROGRESS: Download currently in progress
    DOWNLOADED: Publication files successfully downloaded
    FAILED: Download failed due to error (network, permissions, etc.)
    """

    PENDING = "Pending"
    IN_PROGRESS = "In Progress"
    DOWNLOADED = "Downloaded"
    FAILED = "Failed"


class ProcessingStatus(str, Enum):
    """
    Enum defining possible states for the document processing stage.

    PENDING: Downloaded files waiting to be processed
    IN_PROGRESS: Processing currently in progress
    PROCESSED: Documents successfully processed (OCR, text extraction, chunking)
    OCR_FAILED: OCR (Optical Character Recognition) step failed
    CHUNKING_FAILED: Text chunking step failed
    FAILED: General processing failure
    """

    PENDING = "Pending"
    IN_PROGRESS = "In Progress"
    PROCESSED = "Processed"
    OCR_FAILED = "OCR_Failed"
    CHUNKING_FAILED = "Chunking_Failed"
    FAILED = "Failed"


class EmbeddingStatus(str, Enum):
    """
    Enum defining possible states for the text embedding stage.

    PENDING: Processed text waiting to be converted to embeddings
    IN_PROGRESS: Embedding generation currently in progress
    EMBEDDED: Text successfully converted to vector embeddings
    FAILED: Embedding generation failed
    """

    PENDING = "Pending"
    IN_PROGRESS = "In Progress"
    EMBEDDED = "Embedded"
    FAILED = "Failed"


class IngestionStatus(str, Enum):
    """
    Enum defining possible states for the final ingestion stage.

    PENDING: Embedded content waiting to be ingested into search index
    IN_PROGRESS: Ingestion currently in progress
    INGESTED: Content successfully ingested into search system
    FAILED: Ingestion failed
    """

    PENDING = "Pending"
    IN_PROGRESS = "In Progress"
    INGESTED = "Ingested"
    FAILED = "Failed"


class PublicationTracking(SQLModel, table=True):  # type: ignore[call-arg]
    """
    SQLModel representing the database table for tracking publications
    through ETL pipeline.

    This table stores metadata about each publication and tracks its status through
    the four main ETL stages: download, processing, embedding, and ingestion.
    """

    __tablename__ = "publication_tracking"
    # Database table configuration for cross-platform compatibility
    __table_args__ = {
        "extend_existing": True,  # Allow table schema updates
        "schema": None,  # Use default schema
        "sqlite_autoincrement": True,  # Auto-increment primary keys in SQLite
        "mysql_engine": "InnoDB",  # Use InnoDB engine for MySQL
        "mysql_charset": "utf8mb4",  # Support full Unicode including emojis
        "mysql_collate": "utf8mb4_unicode_ci",  # Case-insensitive Unicode collation
        "comment": "",  # Table-level comment (empty for now)
    }

    # --- Core identification and metadata fields ---
    publication_id: str = Field(primary_key=True)  # Unique identifier (DOI, hash, etc.)
    source_url: str  # URL where the publication was discovered
    title: str | None = None  # Publication title (if available)
    authors: str | None = None  # Author names (comma-separated)
    year: int | None = None  # Publication year
    abstract: str | None = None  # Publication abstract/summary
    # File URLs stored as JSON array in text field for database compatibility
    file_urls_json: str | None = Field(
        sa_column=sa.Column("file_urls", sa.Text),
        default=None,
    )
    content_hash: str | None = None  # Hash of content for change detection

    # --- Discovery stage tracking ---
    # When the publication was first discovered by scrapers
    discovery_timestamp: datetime.datetime = Field(
        default_factory=datetime.datetime.now
    )

    # --- Download stage tracking ---
    download_status: DownloadStatus = Field(
        default=DownloadStatus.PENDING
    )  # Current download status
    # When last download attempt was made
    download_timestamp: datetime.datetime | None = None
    # Number of download attempts (for retry logic)
    download_attempt_count: int = Field(default=0)

    # --- Processing stage tracking ---
    # Current processing status
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    # When last processing attempt was made
    processing_timestamp: datetime.datetime | None = None
    # Number of processing attempts (for retry logic)
    processing_attempt_count: int = Field(default=0)

    # --- Embedding stage tracking ---
    # Current embedding status
    embedding_status: EmbeddingStatus = Field(default=EmbeddingStatus.PENDING)
    # When last embedding attempt was made
    embedding_timestamp: datetime.datetime | None = None
    # Number of embedding attempts (for retry logic)
    embedding_attempt_count: int = Field(default=0)

    # --- Ingestion stage tracking ---
    # Current ingestion status
    ingestion_status: IngestionStatus = Field(default=IngestionStatus.PENDING)
    # When last ingestion attempt was made
    ingestion_timestamp: datetime.datetime | None = None
    # Number of ingestion attempts (for retry logic)
    ingestion_attempt_count: int = Field(default=0)

    # --- General tracking fields ---
    # Automatically updated timestamp for any record modification
    last_updated: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        sa_column=sa.Column(sa.DateTime, onupdate=datetime.datetime.now),
    )
    error_message: str | None = None  # Last error message encountered during processing

    # --- Data validation methods ---
    @field_validator("year")
    @classmethod
    def validate_year(cls, v: int | None) -> int | None:
        """
        Validate that publication year is within a reasonable range.

        Ensures year values are between 1900-2100 to catch data entry errors
        or unrealistic publication dates.
        """
        if v is not None and (v < 1900 or v > 2100):
            raise ValueError("Year must be between 1900 and 2100")
        return v

    # --- File URLs property accessors ---
    # These methods provide a clean interface to work with file URLs as a list
    # while storing them as JSON in the database for compatibility
    @property
    def file_urls(self) -> list[str]:
        """
        Get the file URLs as a list of strings.

        Deserializes the JSON-stored file URLs back into a Python list.
        Returns empty list if no URLs are stored.
        """
        if not self.file_urls_json:
            return []
        return json.loads(self.file_urls_json)

    @file_urls.setter
    def file_urls(self, urls: list[str]):
        """
        Set the file URLs by converting a list to JSON storage format.

        Serializes the list of URLs to JSON for database storage.
        Sets to None if URLs list is None or empty.
        """
        if urls is None:
            self.file_urls_json = None
        else:
            self.file_urls_json = json.dumps(urls)

    # --- Helper methods for status updates ---
    # These methods provide a consistent way to update status across all pipeline stages
    # Each method updates: status, timestamp, attempt count, error message,
    # and last_updated

    def update_download_status(self, status: DownloadStatus, error: str | None = None):
        """
        Update download status with automatic timestamp tracking.

        Args:
            status: New download status to set
            error: Optional error message if status indicates failure

        Updates attempt count, timestamp, and general last_updated field.
        """
        self.download_status = status
        self.download_timestamp = datetime.datetime.now()
        self.download_attempt_count += 1
        if error:
            self.error_message = error
        self.last_updated = datetime.datetime.now()

    def update_processing_status(
        self,
        status: ProcessingStatus,
        error: str | None = None,
    ):
        """
        Update processing status with automatic timestamp tracking.

        Args:
            status: New processing status to set
            error: Optional error message if status indicates failure

        Updates attempt count, timestamp, and general last_updated field.
        """
        self.processing_status = status
        self.processing_timestamp = datetime.datetime.now()
        self.processing_attempt_count += 1
        if error:
            self.error_message = error
        self.last_updated = datetime.datetime.now()

    def update_embedding_status(
        self,
        status: EmbeddingStatus,
        error: str | None = None,
    ):
        """
        Update embedding status with automatic timestamp tracking.

        Args:
            status: New embedding status to set
            error: Optional error message if status indicates failure

        Updates attempt count, timestamp, and general last_updated field.
        """
        self.embedding_status = status
        self.embedding_timestamp = datetime.datetime.now()
        self.embedding_attempt_count += 1
        if error:
            self.error_message = error
        self.last_updated = datetime.datetime.now()

    def update_ingestion_status(
        self,
        status: IngestionStatus,
        error: str | None = None,
    ):
        """
        Update ingestion status with automatic timestamp tracking.

        Args:
            status: New ingestion status to set
            error: Optional error message if status indicates failure

        Updates attempt count, timestamp, and general last_updated field.
        """
        self.ingestion_status = status
        self.ingestion_timestamp = datetime.datetime.now()
        self.ingestion_attempt_count += 1
        if error:
            self.error_message = error
        self.last_updated = datetime.datetime.now()
