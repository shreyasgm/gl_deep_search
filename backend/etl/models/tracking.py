"""
Models for ETL pipeline tracking
"""

import datetime
import json
from enum import Enum

import sqlalchemy as sa
from pydantic import validator
from sqlmodel import Field, MetaData, SQLModel

# Create metadata instance
metadata = MetaData()


# Create Base class
class Base(SQLModel):
    """Base class for all models"""

    metadata = metadata


class DownloadStatus(str, Enum):
    """Enum for download status"""

    PENDING = "Pending"
    IN_PROGRESS = "In Progress"
    DOWNLOADED = "Downloaded"
    FAILED = "Failed"


class ProcessingStatus(str, Enum):
    """Enum for processing status"""

    PENDING = "Pending"
    IN_PROGRESS = "In Progress"
    PROCESSED = "Processed"
    OCR_FAILED = "OCR_Failed"
    CHUNKING_FAILED = "Chunking_Failed"
    FAILED = "Failed"


class EmbeddingStatus(str, Enum):
    """Enum for embedding status"""

    PENDING = "Pending"
    IN_PROGRESS = "In Progress"
    EMBEDDED = "Embedded"
    FAILED = "Failed"


class IngestionStatus(str, Enum):
    """Enum for ingestion status"""

    PENDING = "Pending"
    IN_PROGRESS = "In Progress"
    INGESTED = "Ingested"
    FAILED = "Failed"


class PublicationTracking(Base):
    """Model for tracking publications through the ETL pipeline"""

    __tablename__ = "publication_tracking"

    # Core identification fields
    publication_id: str = Field(primary_key=True)
    source_url: str
    title: str | None = None
    authors: str | None = None
    year: int | None = None
    abstract: str | None = None
    file_urls_json: str | None = Field(
        sa_column=sa.Column("file_urls", sa.Text), default=None
    )
    content_hash: str | None = None

    # Discovery stage
    discovery_timestamp: datetime.datetime = Field(
        default_factory=datetime.datetime.now
    )

    # Download stage
    download_status: DownloadStatus = Field(default=DownloadStatus.PENDING)
    download_timestamp: datetime.datetime | None = None
    download_attempt_count: int = Field(default=0)

    # Processing stage
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    processing_timestamp: datetime.datetime | None = None
    processing_attempt_count: int = Field(default=0)

    # Embedding stage
    embedding_status: EmbeddingStatus = Field(default=EmbeddingStatus.PENDING)
    embedding_timestamp: datetime.datetime | None = None
    embedding_attempt_count: int = Field(default=0)

    # Ingestion stage
    ingestion_status: IngestionStatus = Field(default=IngestionStatus.PENDING)
    ingestion_timestamp: datetime.datetime | None = None
    ingestion_attempt_count: int = Field(default=0)

    # General tracking
    last_updated: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        sa_column=sa.Column(sa.DateTime, onupdate=datetime.datetime.now),
    )
    error_message: str | None = None

    # Year validation
    @validator("year")
    def validate_year(self, v):
        """Validate that year is within a reasonable range"""
        if v is not None and (v < 1900 or v > 2100):
            raise ValueError("Year must be between 1900 and 2100")
        return v

    # File URLs property accessors
    @property
    def file_urls(self) -> list[str]:
        """Get the file URLs as a list"""
        if not self.file_urls_json:
            return []
        return json.loads(self.file_urls_json)

    @file_urls.setter
    def file_urls(self, urls: list[str]):
        """Set the file URLs as a JSON string"""
        if urls is None:
            self.file_urls_json = None
        else:
            self.file_urls_json = json.dumps(urls)

    # Helper methods for status updates
    def update_download_status(self, status: DownloadStatus, error: str | None = None):
        """Update download status with timestamp and error if applicable"""
        self.download_status = status
        self.download_timestamp = datetime.datetime.now()
        self.download_attempt_count += 1
        if error:
            self.error_message = error
        self.last_updated = datetime.datetime.now()

    def update_processing_status(
        self, status: ProcessingStatus, error: str | None = None
    ):
        """Update processing status with timestamp and error if applicable"""
        self.processing_status = status
        self.processing_timestamp = datetime.datetime.now()
        self.processing_attempt_count += 1
        if error:
            self.error_message = error
        self.last_updated = datetime.datetime.now()

    def update_embedding_status(
        self, status: EmbeddingStatus, error: str | None = None
    ):
        """Update embedding status with timestamp and error if applicable"""
        self.embedding_status = status
        self.embedding_timestamp = datetime.datetime.now()
        self.embedding_attempt_count += 1
        if error:
            self.error_message = error
        self.last_updated = datetime.datetime.now()

    def update_ingestion_status(
        self, status: IngestionStatus, error: str | None = None
    ):
        """Update ingestion status with timestamp and error if applicable"""
        self.ingestion_status = status
        self.ingestion_timestamp = datetime.datetime.now()
        self.ingestion_attempt_count += 1
        if error:
            self.error_message = error
        self.last_updated = datetime.datetime.now()
