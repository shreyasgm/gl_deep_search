"""
Supabase adapter implementation for publication tracking.

This module provides a concrete implementation of DatabaseAdapter for Supabase
(PostgreSQL). It handles:
- Async operations using supabase-py async client
- PostgreSQL-specific data types (JSONB, TIMESTAMPTZ)
- Upsert operations for add_publication
- Row Level Security (RLS) via service role key
"""

import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

from supabase import AsyncClient, acreate_client

from backend.etl.models.tracking import (
    DownloadStatus,
    EmbeddingStatus,
    IngestionStatus,
    ProcessingStatus,
    PublicationTracking,
)
from backend.storage.adapters.base import DatabaseAdapter

if TYPE_CHECKING:
    from backend.etl.models.publications import (
        GrowthLabPublication,
        OpenAlexPublication,
    )

logger = logging.getLogger(__name__)


class SupabaseAdapter(DatabaseAdapter):
    """
    Supabase (PostgreSQL) implementation of DatabaseAdapter.

    Uses supabase-py async client for all operations. Expects environment variables:
    - SUPABASE_URL: Your Supabase project URL
    - SUPABASE_SERVICE_ROLE_KEY: Service role key for full database access

    Note: Uses service_role key (not anon key) to bypass Row Level Security
    for ETL pipeline operations.
    """

    def __init__(
        self,
        url: str | None = None,
        service_role_key: str | None = None,
    ):
        """
        Initialize Supabase adapter.

        Args:
            url: Supabase project URL (defaults to SUPABASE_URL env var)
            service_role_key: Service role key (defaults to
                             SUPABASE_SERVICE_ROLE_KEY env var)

        Raises:
            ValueError: If required credentials are not provided
        """
        self.url = url or os.environ.get("SUPABASE_URL")
        self.service_role_key = service_role_key or os.environ.get(
            "SUPABASE_SERVICE_ROLE_KEY"
        )

        if not self.url or not self.service_role_key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be provided "
                "either as arguments or environment variables"
            )

        self.client: AsyncClient | None = None
        self.table_name = "publication_tracking"

    async def initialize(self) -> None:
        """
        Initialize the Supabase client connection.

        Creates async client instance. The schema is assumed to already exist
        (created via SQL migration in Supabase dashboard).

        Raises:
            ConnectionError: If unable to connect to Supabase
        """
        try:
            self.client = await acreate_client(self.url, self.service_role_key)
            logger.info("Supabase client initialized successfully")

            # Verify connection by running a simple query
            await self.client.table(self.table_name).select("publication_id").limit(
                1
            ).execute()
            logger.info("Supabase connection verified")

        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise ConnectionError(f"Failed to connect to Supabase: {e}") from e

    async def close(self) -> None:
        """
        Close Supabase client connection.

        Cleans up the async client session.
        """
        if self.client:
            await self.client.auth.sign_out()
            self.client = None
            logger.info("Supabase client closed")

    def _ensure_client(self) -> AsyncClient:
        """
        Ensure client is initialized before operations.

        Returns:
            The initialized AsyncClient

        Raises:
            RuntimeError: If client not initialized
        """
        if not self.client:
            raise RuntimeError(
                "Supabase client not initialized. Call initialize() first."
            )
        return self.client

    def _publication_to_dict(
        self, publication: "GrowthLabPublication | OpenAlexPublication"
    ) -> dict:
        """
        Convert publication object to dictionary for Supabase insertion.

        Args:
            publication: Publication object from scraper

        Returns:
            Dictionary with Supabase-compatible field names and types
        """
        return {
            "publication_id": publication.paper_id,
            "source_url": str(publication.pub_url) if publication.pub_url else "",
            "title": publication.title,
            "authors": publication.authors,
            "year": publication.year,
            "abstract": publication.abstract,
            "file_urls": [str(url) for url in publication.file_urls],
            "content_hash": publication.content_hash,
            "discovery_timestamp": datetime.now().isoformat(),
        }

    def _dict_to_tracking(self, data: dict) -> PublicationTracking:
        """
        Convert Supabase response dict to PublicationTracking model.

        Args:
            data: Dictionary from Supabase query

        Returns:
            PublicationTracking instance
        """
        # Parse timestamps
        discovery_ts = (
            datetime.fromisoformat(data["discovery_timestamp"])
            if data.get("discovery_timestamp")
            else datetime.now()
        )
        last_updated = (
            datetime.fromisoformat(data["last_updated"])
            if data.get("last_updated")
            else datetime.now()
        )
        download_ts = (
            datetime.fromisoformat(data["download_timestamp"])
            if data.get("download_timestamp")
            else None
        )
        processing_ts = (
            datetime.fromisoformat(data["processing_timestamp"])
            if data.get("processing_timestamp")
            else None
        )
        embedding_ts = (
            datetime.fromisoformat(data["embedding_timestamp"])
            if data.get("embedding_timestamp")
            else None
        )
        ingestion_ts = (
            datetime.fromisoformat(data["ingestion_timestamp"])
            if data.get("ingestion_timestamp")
            else None
        )

        # Create tracking instance
        tracking = PublicationTracking(
            publication_id=data["publication_id"],
            source_url=data.get("source_url", ""),
            title=data.get("title"),
            authors=data.get("authors"),
            year=data.get("year"),
            abstract=data.get("abstract"),
            content_hash=data.get("content_hash"),
            discovery_timestamp=discovery_ts,
            last_updated=last_updated,
            download_status=DownloadStatus(data.get("download_status", "Pending")),
            download_timestamp=download_ts,
            download_attempt_count=data.get("download_attempt_count", 0),
            processing_status=ProcessingStatus(
                data.get("processing_status", "Pending")
            ),
            processing_timestamp=processing_ts,
            processing_attempt_count=data.get("processing_attempt_count", 0),
            embedding_status=EmbeddingStatus(data.get("embedding_status", "Pending")),
            embedding_timestamp=embedding_ts,
            embedding_attempt_count=data.get("embedding_attempt_count", 0),
            ingestion_status=IngestionStatus(data.get("ingestion_status", "Pending")),
            ingestion_timestamp=ingestion_ts,
            ingestion_attempt_count=data.get("ingestion_attempt_count", 0),
            error_message=data.get("error_message"),
        )

        # Set file URLs using the property setter
        tracking.file_urls = data.get("file_urls", [])

        return tracking

    async def add_publication(
        self,
        publication: "GrowthLabPublication | OpenAlexPublication",
    ) -> PublicationTracking:
        """
        Add or update publication using Supabase upsert.

        Implements upsert logic:
        - If publication doesn't exist, creates new record
        - If exists and content unchanged, no-op
        - If exists and content changed, resets appropriate pipeline stages

        Args:
            publication: Publication object from scraper

        Returns:
            PublicationTracking record

        Raises:
            ValueError: If publication data is invalid
            Exception: For database errors
        """
        client = self._ensure_client()

        try:
            # Validate required fields
            if not publication.paper_id:
                raise ValueError("Publication must have a paper_id")
            if not publication.title:
                raise ValueError("Publication must have a title")

            # Check if publication exists
            existing_response = await client.table(self.table_name).select("*").eq(
                "publication_id", publication.paper_id
            ).execute()

            existing = existing_response.data[0] if existing_response.data else None

            # Prepare base data
            pub_dict = self._publication_to_dict(publication)

            if existing:
                # UPDATE LOGIC: Determine what needs to be reset
                needs_download = False
                needs_processing = False
                needs_embedding = False
                needs_ingestion = False

                # Check if content changed
                if existing.get("content_hash") != publication.content_hash:
                    needs_download = True
                    needs_processing = True
                    needs_embedding = True
                    needs_ingestion = True
                    logger.info(
                        f"Content hash changed for {publication.paper_id}, "
                        "resetting all stages"
                    )

                # Check if files changed
                existing_files = set(existing.get("file_urls", []))
                new_files = set([str(url) for url in publication.file_urls])
                if existing_files != new_files:
                    needs_download = True
                    needs_processing = True
                    needs_embedding = True
                    needs_ingestion = True
                    logger.info(
                        f"Files changed for {publication.paper_id}, "
                        "resetting all stages"
                    )

                # Build update dict with reset statuses if needed
                update_dict: dict[str, Any] = {
                    **pub_dict,
                    "last_updated": datetime.now().isoformat(),
                }

                if needs_download:
                    update_dict.update(
                        {
                            "download_status": DownloadStatus.PENDING.value,
                            "download_timestamp": None,
                        }
                    )
                if needs_processing:
                    update_dict.update(
                        {
                            "processing_status": ProcessingStatus.PENDING.value,
                            "processing_timestamp": None,
                        }
                    )
                if needs_embedding:
                    update_dict.update(
                        {
                            "embedding_status": EmbeddingStatus.PENDING.value,
                            "embedding_timestamp": None,
                        }
                    )
                if needs_ingestion:
                    update_dict.update(
                        {
                            "ingestion_status": IngestionStatus.PENDING.value,
                            "ingestion_timestamp": None,
                        }
                    )

                # Update existing record
                response = await client.table(self.table_name).update(update_dict).eq(
                    "publication_id", publication.paper_id
                ).execute()

                logger.info(f"Updated publication: {publication.paper_id}")

            else:
                # INSERT LOGIC: New publication
                pub_dict["download_status"] = DownloadStatus.PENDING.value
                pub_dict["processing_status"] = ProcessingStatus.PENDING.value
                pub_dict["embedding_status"] = EmbeddingStatus.PENDING.value
                pub_dict["ingestion_status"] = IngestionStatus.PENDING.value
                pub_dict["download_attempt_count"] = 0
                pub_dict["processing_attempt_count"] = 0
                pub_dict["embedding_attempt_count"] = 0
                pub_dict["ingestion_attempt_count"] = 0

                response = await client.table(self.table_name).insert(
                    pub_dict
                ).execute()

                logger.info(f"Inserted new publication: {publication.paper_id}")

            # Return the tracking record
            if response.data:
                return self._dict_to_tracking(response.data[0])
            else:
                raise Exception(f"Failed to add/update publication: {response}")

        except Exception as e:
            logger.error(
                f"Error adding/updating publication {publication.paper_id}: {e}"
            )
            raise

    async def add_publications(
        self,
        publications: list["GrowthLabPublication | OpenAlexPublication"],
    ) -> list[PublicationTracking]:
        """
        Add multiple publications sequentially.

        Note: Supabase doesn't support batch upsert with conflict resolution,
        so we process each publication individually. For large batches,
        consider implementing batching logic in the caller.

        Args:
            publications: List of publication objects

        Returns:
            List of PublicationTracking records
        """
        results = []
        for pub in publications:
            tracking = await self.add_publication(pub)
            results.append(tracking)
        return results

    async def get_publication(self, publication_id: str) -> PublicationTracking | None:
        """
        Retrieve a specific publication by ID.

        Args:
            publication_id: Unique identifier

        Returns:
            PublicationTracking if found, None otherwise
        """
        client = self._ensure_client()

        try:
            response = await client.table(self.table_name).select("*").eq(
                "publication_id", publication_id
            ).execute()

            if response.data:
                return self._dict_to_tracking(response.data[0])
            return None

        except Exception as e:
            logger.error(f"Error fetching publication {publication_id}: {e}")
            raise

    async def get_publications_for_download(
        self, limit: int | None = None
    ) -> list[PublicationTracking]:
        """
        Get publications ready for download stage.

        Returns publications where download_status = 'Pending',
        ordered by discovery_timestamp ASC (FIFO).
        """
        client = self._ensure_client()

        try:
            query = (
                client.table(self.table_name)
                .select("*")
                .eq("download_status", DownloadStatus.PENDING.value)
                .order("discovery_timestamp", desc=False)
            )

            if limit:
                query = query.limit(limit)

            response = await query.execute()
            return [self._dict_to_tracking(item) for item in response.data]

        except Exception as e:
            logger.error(f"Error fetching publications for download: {e}")
            raise

    async def get_publications_for_processing(
        self, limit: int | None = None
    ) -> list[PublicationTracking]:
        """
        Get publications ready for processing stage.

        Returns publications where:
        - download_status = 'Downloaded'
        - processing_status = 'Pending'
        """
        client = self._ensure_client()

        try:
            query = (
                client.table(self.table_name)
                .select("*")
                .eq("download_status", DownloadStatus.DOWNLOADED.value)
                .eq("processing_status", ProcessingStatus.PENDING.value)
                .order("download_timestamp", desc=False)
            )

            if limit:
                query = query.limit(limit)

            response = await query.execute()
            return [self._dict_to_tracking(item) for item in response.data]

        except Exception as e:
            logger.error(f"Error fetching publications for processing: {e}")
            raise

    async def get_publications_for_embedding(
        self, limit: int | None = None
    ) -> list[PublicationTracking]:
        """
        Get publications ready for embedding stage.

        Returns publications where:
        - processing_status = 'Processed'
        - embedding_status = 'Pending'
        """
        client = self._ensure_client()

        try:
            query = (
                client.table(self.table_name)
                .select("*")
                .eq("processing_status", ProcessingStatus.PROCESSED.value)
                .eq("embedding_status", EmbeddingStatus.PENDING.value)
                .order("processing_timestamp", desc=False)
            )

            if limit:
                query = query.limit(limit)

            response = await query.execute()
            return [self._dict_to_tracking(item) for item in response.data]

        except Exception as e:
            logger.error(f"Error fetching publications for embedding: {e}")
            raise

    async def get_publications_for_ingestion(
        self, limit: int | None = None
    ) -> list[PublicationTracking]:
        """
        Get publications ready for ingestion stage.

        Returns publications where:
        - embedding_status = 'Embedded'
        - ingestion_status = 'Pending'
        """
        client = self._ensure_client()

        try:
            query = (
                client.table(self.table_name)
                .select("*")
                .eq("embedding_status", EmbeddingStatus.EMBEDDED.value)
                .eq("ingestion_status", IngestionStatus.PENDING.value)
                .order("embedding_timestamp", desc=False)
            )

            if limit:
                query = query.limit(limit)

            response = await query.execute()
            return [self._dict_to_tracking(item) for item in response.data]

        except Exception as e:
            logger.error(f"Error fetching publications for ingestion: {e}")
            raise

    async def update_download_status(
        self,
        publication_id: str,
        status: DownloadStatus,
        error: str | None = None,
    ) -> bool:
        """
        Update download status for a publication.

        Updates download_status, timestamp, attempt count, and error message.
        """
        client = self._ensure_client()

        try:
            update_dict: dict[str, Any] = {
                "download_status": status.value,
                "download_timestamp": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
            }

            if error:
                update_dict["error_message"] = error

            # Increment attempt count using PostgreSQL
            # Note: Supabase doesn't support increment operations directly,
            # so we need to fetch current value first
            current = await self.get_publication(publication_id)
            if not current:
                logger.warning(f"Publication not found: {publication_id}")
                return False

            update_dict["download_attempt_count"] = current.download_attempt_count + 1

            response = await client.table(self.table_name).update(update_dict).eq(
                "publication_id", publication_id
            ).execute()

            return len(response.data) > 0

        except Exception as e:
            logger.error(
                f"Error updating download status for {publication_id}: {e}"
            )
            raise

    async def update_processing_status(
        self,
        publication_id: str,
        status: ProcessingStatus,
        error: str | None = None,
    ) -> bool:
        """
        Update processing status for a publication.
        """
        client = self._ensure_client()

        try:
            current = await self.get_publication(publication_id)
            if not current:
                logger.warning(f"Publication not found: {publication_id}")
                return False

            update_dict: dict[str, Any] = {
                "processing_status": status.value,
                "processing_timestamp": datetime.now().isoformat(),
                "processing_attempt_count": current.processing_attempt_count + 1,
                "last_updated": datetime.now().isoformat(),
            }

            if error:
                update_dict["error_message"] = error

            response = await client.table(self.table_name).update(update_dict).eq(
                "publication_id", publication_id
            ).execute()

            return len(response.data) > 0

        except Exception as e:
            logger.error(
                f"Error updating processing status for {publication_id}: {e}"
            )
            raise

    async def update_embedding_status(
        self,
        publication_id: str,
        status: EmbeddingStatus,
        error: str | None = None,
    ) -> bool:
        """
        Update embedding status for a publication.
        """
        client = self._ensure_client()

        try:
            current = await self.get_publication(publication_id)
            if not current:
                logger.warning(f"Publication not found: {publication_id}")
                return False

            update_dict: dict[str, Any] = {
                "embedding_status": status.value,
                "embedding_timestamp": datetime.now().isoformat(),
                "embedding_attempt_count": current.embedding_attempt_count + 1,
                "last_updated": datetime.now().isoformat(),
            }

            if error:
                update_dict["error_message"] = error

            response = await client.table(self.table_name).update(update_dict).eq(
                "publication_id", publication_id
            ).execute()

            return len(response.data) > 0

        except Exception as e:
            logger.error(
                f"Error updating embedding status for {publication_id}: {e}"
            )
            raise

    async def update_ingestion_status(
        self,
        publication_id: str,
        status: IngestionStatus,
        error: str | None = None,
    ) -> bool:
        """
        Update ingestion status for a publication.
        """
        client = self._ensure_client()

        try:
            current = await self.get_publication(publication_id)
            if not current:
                logger.warning(f"Publication not found: {publication_id}")
                return False

            update_dict: dict[str, Any] = {
                "ingestion_status": status.value,
                "ingestion_timestamp": datetime.now().isoformat(),
                "ingestion_attempt_count": current.ingestion_attempt_count + 1,
                "last_updated": datetime.now().isoformat(),
            }

            if error:
                update_dict["error_message"] = error

            response = await client.table(self.table_name).update(update_dict).eq(
                "publication_id", publication_id
            ).execute()

            return len(response.data) > 0

        except Exception as e:
            logger.error(
                f"Error updating ingestion status for {publication_id}: {e}"
            )
            raise

    async def get_publication_status(self, publication_id: str) -> dict | None:
        """
        Get current status of a publication across all stages.

        Returns:
            Dictionary with status information or None if not found
        """
        tracking = await self.get_publication(publication_id)

        if not tracking:
            return None

        return {
            "publication_id": tracking.publication_id,
            "title": tracking.title,
            "download_status": tracking.download_status.value,
            "processing_status": tracking.processing_status.value,
            "embedding_status": tracking.embedding_status.value,
            "ingestion_status": tracking.ingestion_status.value,
            "error_message": tracking.error_message,
            "last_updated": tracking.last_updated,
        }

    async def get_pipeline_summary(self) -> dict:
        """
        Get summary statistics for the ETL pipeline.

        Uses Supabase's aggregation capabilities to efficiently compute counts.
        """
        client = self._ensure_client()

        try:
            # Get all publications to compute stats
            # Note: For very large tables, consider using Supabase's RPC
            # to call the etl_pipeline_summary view created in migration
            response = await client.table(self.table_name).select(
                "download_status,processing_status,embedding_status,ingestion_status"
            ).execute()

            data = response.data
            total = len(data)

            # Count by status
            downloaded = sum(
                1 for r in data if r["download_status"] == "Downloaded"
            )
            processed = sum(
                1 for r in data if r["processing_status"] == "Processed"
            )
            embedded = sum(1 for r in data if r["embedding_status"] == "Embedded")
            ingested = sum(1 for r in data if r["ingestion_status"] == "Ingested")

            failed = sum(
                1
                for r in data
                if r["download_status"] == "Failed"
                or "Failed" in r["processing_status"]
                or r["embedding_status"] == "Failed"
                or r["ingestion_status"] == "Failed"
            )

            pending_download = sum(
                1 for r in data if r["download_status"] == "Pending"
            )
            pending_processing = sum(
                1
                for r in data
                if r["download_status"] == "Downloaded"
                and r["processing_status"] == "Pending"
            )
            pending_embedding = sum(
                1
                for r in data
                if r["processing_status"] == "Processed"
                and r["embedding_status"] == "Pending"
            )
            pending_ingestion = sum(
                1
                for r in data
                if r["embedding_status"] == "Embedded"
                and r["ingestion_status"] == "Pending"
            )

            return {
                "total_publications": total,
                "downloaded": downloaded,
                "processed": processed,
                "embedded": embedded,
                "ingested": ingested,
                "failed": failed,
                "pending_download": pending_download,
                "pending_processing": pending_processing,
                "pending_embedding": pending_embedding,
                "pending_ingestion": pending_ingestion,
            }

        except Exception as e:
            logger.error(f"Error fetching pipeline summary: {e}")
            raise
