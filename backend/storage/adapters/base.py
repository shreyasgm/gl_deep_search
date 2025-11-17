"""
Abstract base class for database adapters.

This module defines the DatabaseAdapter interface that all database implementations
must follow. The adapter pattern ensures that:
1. Business logic in PublicationTracker is decoupled from database specifics
2. Different database backends (SQLite, Supabase, etc.) can be swapped easily
3. All database operations use async/await for better performance
4. Testing is easier with mock implementations
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from backend.etl.models.tracking import (
    DownloadStatus,
    EmbeddingStatus,
    IngestionStatus,
    ProcessingStatus,
    PublicationTracking,
)

if TYPE_CHECKING:
    from backend.etl.models.publications import (
        GrowthLabPublication,
        OpenAlexPublication,
    )


class DatabaseAdapter(ABC):
    """
    Abstract base class for publication tracking database adapters.

    All database implementations must provide async implementations of these methods.
    This ensures consistency across different backends while allowing
    backend-specific optimizations.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the database connection and schema.

        This method should:
        - Establish connection to the database
        - Create tables if they don't exist
        - Set up any necessary indexes or constraints
        - Perform connection health checks

        Raises:
            ConnectionError: If unable to connect to database
            Exception: For other initialization failures
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Close database connections and clean up resources.

        This method should:
        - Close all open connections
        - Release any connection pool resources
        - Flush any pending writes
        - Perform cleanup operations

        Should be called when the adapter is no longer needed.
        """
        pass

    @abstractmethod
    async def add_publication(
        self,
        publication: "GrowthLabPublication | OpenAlexPublication",
    ) -> PublicationTracking:
        """
        Add a new publication or update existing one in the tracking database.

        This method implements upsert logic:
        - If publication doesn't exist, creates new tracking record
        - If publication exists, updates metadata and resets pipeline stages if needed
        - Handles content hash comparison for change detection
        - Determines which pipeline stages need to be re-run

        Args:
            publication: Publication object from scraper
                        (GrowthLabPublication or OpenAlexPublication)

        Returns:
            PublicationTracking record (newly created or updated)

        Raises:
            ValueError: If publication data is invalid (missing required fields)
            Exception: For database errors
        """
        pass

    @abstractmethod
    async def add_publications(
        self,
        publications: list["GrowthLabPublication | OpenAlexPublication"],
    ) -> list[PublicationTracking]:
        """
        Add multiple publications in a batch operation.

        This method provides efficient bulk insertion/updates:
        - Processes multiple publications in a single transaction where possible
        - Returns results in the same order as input
        - Implements rollback behavior on errors (backend-specific)

        Args:
            publications: List of publication objects to add/update

        Returns:
            List of PublicationTracking records in same order as input

        Raises:
            ValueError: If any publication data is invalid
            Exception: For database errors
        """
        pass

    @abstractmethod
    async def get_publication(self, publication_id: str) -> PublicationTracking | None:
        """
        Retrieve a specific publication by its unique identifier.

        Args:
            publication_id: Unique identifier of the publication (DOI, hash, etc.)

        Returns:
            PublicationTracking record if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_publications_for_download(
        self, limit: int | None = None
    ) -> list[PublicationTracking]:
        """
        Get publications that need to be downloaded.

        Returns publications where download_status is 'Pending', ordered by
        discovery_timestamp (oldest first, FIFO queue behavior).

        Args:
            limit: Optional maximum number of publications to return

        Returns:
            List of publications ready for download stage
        """
        pass

    @abstractmethod
    async def get_publications_for_processing(
        self, limit: int | None = None
    ) -> list[PublicationTracking]:
        """
        Get publications that need to be processed (OCR, chunking, etc.).

        Returns publications where:
        - download_status is 'Downloaded' (prerequisite)
        - processing_status is 'Pending'

        Ordered by download_timestamp (oldest first, FIFO queue behavior).

        Args:
            limit: Optional maximum number of publications to return

        Returns:
            List of publications ready for processing stage
        """
        pass

    @abstractmethod
    async def get_publications_for_embedding(
        self, limit: int | None = None
    ) -> list[PublicationTracking]:
        """
        Get publications that need embedding generation.

        Returns publications where:
        - processing_status is 'Processed' (prerequisite)
        - embedding_status is 'Pending'

        Ordered by processing_timestamp (oldest first, FIFO queue behavior).

        Args:
            limit: Optional maximum number of publications to return

        Returns:
            List of publications ready for embedding stage
        """
        pass

    @abstractmethod
    async def get_publications_for_ingestion(
        self, limit: int | None = None
    ) -> list[PublicationTracking]:
        """
        Get publications that need ingestion into search index.

        Returns publications where:
        - embedding_status is 'Embedded' (prerequisite)
        - ingestion_status is 'Pending'

        Ordered by embedding_timestamp (oldest first, FIFO queue behavior).

        Args:
            limit: Optional maximum number of publications to return

        Returns:
            List of publications ready for ingestion stage
        """
        pass

    @abstractmethod
    async def update_download_status(
        self,
        publication_id: str,
        status: DownloadStatus,
        error: str | None = None,
    ) -> bool:
        """
        Update download status for a publication.

        This method should:
        - Update download_status field
        - Set download_timestamp to current time
        - Increment download_attempt_count
        - Set error_message if error is provided
        - Update last_updated timestamp

        Args:
            publication_id: Unique identifier of the publication
            status: New download status
            error: Optional error message if status is 'Failed'

        Returns:
            True if publication was found and updated, False if not found

        Raises:
            Exception: For database errors
        """
        pass

    @abstractmethod
    async def update_processing_status(
        self,
        publication_id: str,
        status: ProcessingStatus,
        error: str | None = None,
    ) -> bool:
        """
        Update processing status for a publication.

        This method should:
        - Update processing_status field
        - Set processing_timestamp to current time
        - Increment processing_attempt_count
        - Set error_message if error is provided
        - Update last_updated timestamp

        Args:
            publication_id: Unique identifier of the publication
            status: New processing status
            error: Optional error message if status indicates failure

        Returns:
            True if publication was found and updated, False if not found

        Raises:
            Exception: For database errors
        """
        pass

    @abstractmethod
    async def update_embedding_status(
        self,
        publication_id: str,
        status: EmbeddingStatus,
        error: str | None = None,
    ) -> bool:
        """
        Update embedding status for a publication.

        This method should:
        - Update embedding_status field
        - Set embedding_timestamp to current time
        - Increment embedding_attempt_count
        - Set error_message if error is provided
        - Update last_updated timestamp

        Args:
            publication_id: Unique identifier of the publication
            status: New embedding status
            error: Optional error message if status is 'Failed'

        Returns:
            True if publication was found and updated, False if not found

        Raises:
            Exception: For database errors
        """
        pass

    @abstractmethod
    async def update_ingestion_status(
        self,
        publication_id: str,
        status: IngestionStatus,
        error: str | None = None,
    ) -> bool:
        """
        Update ingestion status for a publication.

        This method should:
        - Update ingestion_status field
        - Set ingestion_timestamp to current time
        - Increment ingestion_attempt_count
        - Set error_message if error is provided
        - Update last_updated timestamp

        Args:
            publication_id: Unique identifier of the publication
            status: New ingestion status
            error: Optional error message if status is 'Failed'

        Returns:
            True if publication was found and updated, False if not found

        Raises:
            Exception: For database errors
        """
        pass

    @abstractmethod
    async def get_publication_status(self, publication_id: str) -> dict | None:
        """
        Get the current status of a publication across all pipeline stages.

        Returns a dictionary with status information for display/monitoring purposes.

        Args:
            publication_id: Unique identifier of the publication

        Returns:
            Dictionary with keys:
            - publication_id: str
            - title: str | None
            - download_status: str
            - processing_status: str
            - embedding_status: str
            - ingestion_status: str
            - error_message: str | None
            - last_updated: datetime

            Returns None if publication not found.
        """
        pass

    @abstractmethod
    async def get_pipeline_summary(self) -> dict:
        """
        Get summary statistics for the entire ETL pipeline.

        Returns aggregated counts for monitoring and dashboard display.

        Returns:
            Dictionary with keys:
            - total_publications: int
            - downloaded: int
            - processed: int
            - embedded: int
            - ingested: int
            - failed: int
            - pending_download: int
            - pending_processing: int
            - pending_embedding: int
            - pending_ingestion: int
        """
        pass
