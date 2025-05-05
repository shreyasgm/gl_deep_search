"""
Models for ETL pipeline tracking
"""

import datetime
import json
from enum import Enum
from typing import List, Optional

import sqlalchemy as sa
from pydantic import AnyHttpUrl, validator
from sqlmodel import Field, SQLModel


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


class PublicationTracking(SQLModel, table=True):
    """Model for tracking publications through the ETL pipeline"""
    
    __tablename__ = "publication_tracking"
    
    # Core identification fields
    publication_id: str = Field(primary_key=True)
    source_url: str
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[int] = None
    abstract: Optional[str] = None
    file_urls_json: Optional[str] = Field(sa_column=sa.Column("file_urls", sa.Text), default=None)
    content_hash: Optional[str] = None
    
    # Discovery stage
    discovery_timestamp: datetime.datetime = Field(
        default_factory=datetime.datetime.now
    )
    
    # Download stage
    download_status: DownloadStatus = Field(default=DownloadStatus.PENDING)
    download_timestamp: Optional[datetime.datetime] = None
    download_attempt_count: int = Field(default=0)
    
    # Processing stage
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    processing_timestamp: Optional[datetime.datetime] = None
    processing_attempt_count: int = Field(default=0)
    
    # Embedding stage
    embedding_status: EmbeddingStatus = Field(default=EmbeddingStatus.PENDING)
    embedding_timestamp: Optional[datetime.datetime] = None
    embedding_attempt_count: int = Field(default=0)
    
    # Ingestion stage
    ingestion_status: IngestionStatus = Field(default=IngestionStatus.PENDING)
    ingestion_timestamp: Optional[datetime.datetime] = None
    ingestion_attempt_count: int = Field(default=0)
    
    # General tracking
    last_updated: datetime.datetime = Field(
        default_factory=datetime.datetime.now, 
        sa_column=sa.Column(sa.DateTime, onupdate=datetime.datetime.now)
    )
    error_message: Optional[str] = None
    
    # Year validation
    @validator("year")
    def validate_year(cls, v):
        """Validate that year is within a reasonable range"""
        if v is not None and (v < 1900 or v > 2100):
            raise ValueError("Year must be between 1900 and 2100")
        return v
    
    # File URLs property accessors
    @property
    def file_urls(self) -> List[str]:
        """Get the file URLs as a list"""
        if not self.file_urls_json:
            return []
        return json.loads(self.file_urls_json)
    
    @file_urls.setter
    def file_urls(self, urls: List[str]):
        """Set the file URLs as a JSON string"""
        if urls is None:
            self.file_urls_json = None
        else:
            self.file_urls_json = json.dumps(urls)
    
    # Helper methods for status updates
    def update_download_status(self, status: DownloadStatus, error: Optional[str] = None):
        """Update download status with timestamp and error if applicable"""
        self.download_status = status
        self.download_timestamp = datetime.datetime.now()
        self.download_attempt_count += 1
        if error:
            self.error_message = error
        self.last_updated = datetime.datetime.now()
    
    def update_processing_status(self, status: ProcessingStatus, error: Optional[str] = None):
        """Update processing status with timestamp and error if applicable"""
        self.processing_status = status
        self.processing_timestamp = datetime.datetime.now()
        self.processing_attempt_count += 1
        if error:
            self.error_message = error
        self.last_updated = datetime.datetime.now()
    
    def update_embedding_status(self, status: EmbeddingStatus, error: Optional[str] = None):
        """Update embedding status with timestamp and error if applicable"""
        self.embedding_status = status
        self.embedding_timestamp = datetime.datetime.now()
        self.embedding_attempt_count += 1
        if error:
            self.error_message = error
        self.last_updated = datetime.datetime.now()
    
    def update_ingestion_status(self, status: IngestionStatus, error: Optional[str] = None):
        """Update ingestion status with timestamp and error if applicable"""
        self.ingestion_status = status
        self.ingestion_timestamp = datetime.datetime.now()
        self.ingestion_attempt_count += 1
        if error:
            self.error_message = error
        self.last_updated = datetime.datetime.now()