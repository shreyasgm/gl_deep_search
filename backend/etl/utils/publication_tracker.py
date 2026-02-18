"""
Publication tracking utility for ETL pipeline.

This module provides the business logic layer for managing publication lifecycle
through the ETL pipeline stages (discovery, download, processing, embedding, ingestion).
It acts as the main interface between scrapers, processing components,
and the tracking database.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass

from sqlmodel import Session, SQLModel, select

# Import publication data models from scrapers
from backend.etl.models.publications import GrowthLabPublication, OpenAlexPublication

# Import status enums and tracking database model
from backend.etl.models.tracking import (
    DownloadStatus,
    EmbeddingStatus,
    IngestionStatus,
    ProcessingStatus,
    PublicationTracking,
)

# Import scraper classes for publication discovery
# Import inside methods to avoid circular imports
# from backend.etl.scrapers.growthlab import GrowthLabScraper
# from backend.etl.scrapers.openalex import OpenAlexClient
# Import database connection utilities
from backend.storage.database import engine, ensure_db_initialized

logger = logging.getLogger(__name__)


@dataclass
class ProcessingPlan:
    """
    Represents a decision plan for what ETL stages need to be executed
    for a publication.

    This dataclass encapsulates the logic for determining which pipeline stages
    need to be run based on the current state of a publication and any detected changes.
    """

    publication_id: str  # Unique identifier for the publication
    needs_download: bool  # Whether files need to be downloaded/re-downloaded
    needs_processing: bool  # Whether document processing (OCR, chunking) is needed
    needs_embedding: bool  # Whether text embedding generation is needed
    needs_ingestion: bool  # Whether ingestion into search index is needed
    files_to_reprocess: list[str]  # Specific files that need reprocessing
    reason: str  # Human-readable explanation for why these actions are needed


class PublicationTracker:
    """
    Main business logic class for managing publication lifecycle through ETL pipeline.

    This class provides the primary interface for:
    - Discovering new publications from multiple sources
    - Determining what processing is needed for each publication
    - Adding/updating publications in the tracking database
    - Updating status as publications move through pipeline stages
    - Querying publications ready for each processing stage
    """

    def __init__(self, ensure_db: bool = True):
        """
        Initialize the publication tracker with database and scraper setup.

        Args:
            ensure_db: If True, ensures database tables are created and initialized.
                      Set to False in tests or when database is already set up.
        """
        # Initialize database schema if requested
        if ensure_db:
            ensure_db_initialized()  # Ensure database connection is ready
            SQLModel.metadata.create_all(engine)  # Create all tables defined in models

        # Initialize scraper instances for publication discovery
        # Import here to avoid circular imports
        from backend.etl.scrapers.growthlab import GrowthLabScraper
        from backend.etl.scrapers.openalex import OpenAlexClient

        self.growthlab_scraper = GrowthLabScraper()  # Harvard Growth Lab scraper
        self.openalex_client = OpenAlexClient()  # OpenAlex API client

    @contextmanager
    def _get_session(self, session: Session | None = None):
        """
        Context manager to handle database session lifecycle.

        If no session is provided, creates a new one and ensures it's closed.
        If a session is provided, uses it without closing (caller's responsibility).

        Args:
            session: Optional existing session to use

        Yields:
            Database session to use for operations
        """
        if session is None:
            # Create and manage our own session
            new_session = Session(engine)
            try:
                yield new_session
            finally:
                new_session.close()
        else:
            # Use provided session without closing it
            yield session

    async def discover_publications(
        self,
    ) -> list[tuple[GrowthLabPublication | OpenAlexPublication, str]]:
        """
        Discover new publications from all configured sources.

        This method orchestrates the discovery process by calling both
        the Growth Lab scraper and OpenAlex client to find publications.
        It's the main entry point for Issue #8 (Check for new publications).

        Returns:
            List of tuples where each tuple contains:
            - publication: The discovered publication object
            - source: String identifier of the source ("growthlab" or "openalex")

        Raises:
            Exception: If any scraper fails during discovery
        """
        publications = []

        try:
            # Discover publications from Harvard Growth Lab website
            # This scraper extracts publications from web pages and enriches metadata
            growthlab_pubs = (
                await self.growthlab_scraper.extract_and_enrich_publications()
            )
            publications.extend([(pub, "growthlab") for pub in growthlab_pubs])

            # Discover publications from OpenAlex API
            # This client fetches publications via API based on configured criteria
            openalex_pubs = await self.openalex_client.fetch_publications()
            publications.extend([(pub, "openalex") for pub in openalex_pubs])

        except Exception as e:
            logger.error(f"Error discovering publications: {str(e)}")
            raise

        return publications

    def generate_processing_plan(
        self,
        publication: GrowthLabPublication | OpenAlexPublication,
        session: Session | None = None,
    ) -> ProcessingPlan:
        """
        Generate a processing plan by comparing publication against existing
        tracking data.

        This is the core method for Issue #8 - it determines what processing steps
        are needed by comparing the discovered publication with any existing tracking
        record. It handles both new publications and updates to existing ones.

        Args:
            publication: The discovered publication to analyze
            session: Optional database session (creates new one if not provided)

        Returns:
            ProcessingPlan indicating what stages need to be executed and why
        """
        with self._get_session(session) as sess:
            # Query database to see if we've seen this publication before
            stmt = select(PublicationTracking).where(
                PublicationTracking.publication_id == publication.paper_id
            )
            existing = sess.exec(stmt).first()

            # CASE 1: Completely new publication - needs full pipeline
            if not existing:
                return ProcessingPlan(
                    publication_id=publication.paper_id,
                    needs_download=True,
                    needs_processing=True,
                    needs_embedding=True,
                    needs_ingestion=True,
                    files_to_reprocess=[],
                    reason="New publication",
                )

            # CASE 2: Content hash changed - indicates substantive changes
            # Content hash covers title, abstract, authors, etc.
            if existing.content_hash != publication.content_hash:
                return ProcessingPlan(
                    publication_id=publication.paper_id,
                    needs_download=True,
                    needs_processing=True,
                    needs_embedding=True,
                    needs_ingestion=True,
                    files_to_reprocess=[str(url) for url in publication.file_urls],
                    reason="Content hash changed",
                )

            # CASE 3: File URLs changed - new or removed files
            # Compare sets of file URLs to detect additions/removals
            existing_files = set(existing.file_urls)
            new_files = set([str(url) for url in publication.file_urls])
            added_files = new_files - existing_files
            removed_files = existing_files - new_files

            if added_files or removed_files:
                return ProcessingPlan(
                    publication_id=publication.paper_id,
                    needs_download=True,
                    needs_processing=True,
                    needs_embedding=True,
                    needs_ingestion=True,
                    files_to_reprocess=list(added_files),
                    reason="Files changed",
                )

            # CASE 4: No content changes - check pipeline status
            # Determine what stages need to run based on current status
            # Each stage depends on successful completion of previous stages
            needs_download = existing.download_status != DownloadStatus.DOWNLOADED
            needs_processing = (
                needs_download  # Can't process without successful download
                or existing.processing_status != ProcessingStatus.PROCESSED
            )
            needs_embedding = (
                needs_processing  # Can't embed without successful processing
                or existing.embedding_status != EmbeddingStatus.EMBEDDED
            )
            needs_ingestion = (
                needs_embedding  # Can't ingest without successful embedding
                or existing.ingestion_status != IngestionStatus.INGESTED
            )

            return ProcessingPlan(
                publication_id=publication.paper_id,
                needs_download=needs_download,
                needs_processing=needs_processing,
                needs_embedding=needs_embedding,
                needs_ingestion=needs_ingestion,
                files_to_reprocess=[],
                reason="Status check",
            )

    def add_publication(
        self,
        publication: GrowthLabPublication | OpenAlexPublication,
        session: Session | None = None,
    ) -> PublicationTracking:
        """
        Add a publication to the manifest system or update existing entry.

        This method implements the core functionality for Issue #19
        (Publication Manifest System).
        It handles both new publications and updates to existing ones, including:
        - Version control logic for publication updates
        - Deduplication to avoid processing same content multiple times
        - Status reset when content changes are detected

        Args:
            publication: The publication to track (from scraper)
            session: Optional database session (creates new one if not provided)

        Returns:
            The PublicationTracking record (new or updated)

        Raises:
            ValueError: If publication data is invalid
            sqlalchemy.exc.IntegrityError: If there's a database constraint violation
        """
        with self._get_session(session) as sess:
            try:
                # Validate required fields
                if not publication.paper_id:
                    raise ValueError(
                        "Publication must have a paper_id (unique identifier)"
                    )
                if not publication.title:
                    raise ValueError("Publication must have a title")
                # Generate processing plan to determine what needs to be done
                # This handles the deduplication and change detection logic
                plan = self.generate_processing_plan(publication, session=sess)

                # Check if we already have a tracking record for this publication
                stmt = select(PublicationTracking).where(
                    PublicationTracking.publication_id == publication.paper_id
                )
                existing = sess.exec(stmt).first()

                if existing:
                    # UPDATE EXISTING PUBLICATION (version control logic)
                    # Update metadata fields with latest information
                    existing.title = publication.title
                    existing.authors = publication.authors
                    existing.year = publication.year
                    existing.abstract = publication.abstract
                    existing.source_url = (
                        str(publication.pub_url) if publication.pub_url else ""
                    )
                    existing.content_hash = publication.content_hash
                    existing.file_urls = [str(url) for url in publication.file_urls]

                    # Reset pipeline stages based on what the processing plan determined
                    # This implements version control - if content changed,
                    # reset appropriate stages
                    if plan.needs_download:
                        existing.download_status = DownloadStatus.PENDING
                        existing.download_timestamp = None
                    if plan.needs_processing:
                        existing.processing_status = ProcessingStatus.PENDING
                        existing.processing_timestamp = None
                    if plan.needs_embedding:
                        existing.embedding_status = EmbeddingStatus.PENDING
                        existing.embedding_timestamp = None
                    if plan.needs_ingestion:
                        existing.ingestion_status = IngestionStatus.PENDING
                        existing.ingestion_timestamp = None

                    # Save changes to database
                    sess.add(existing)
                    sess.commit()
                    logger.info(
                        f"Updated tracking for publication: "
                        f"{publication.paper_id} - {plan.reason}"
                    )

                    # Prepare object for return (detach from session)
                    sess.refresh(existing)
                    sess.expunge(existing)
                    return existing

                # CREATE NEW TRACKING RECORD
                # This is a completely new publication - create initial tracking entry
                tracking = PublicationTracking(
                    publication_id=publication.paper_id,
                    source_url=str(publication.pub_url) if publication.pub_url else "",
                    title=publication.title,
                    year=publication.year,
                    abstract=publication.abstract,
                    content_hash=publication.content_hash,
                )

                # Set list fields via property setters (JSON serialization)
                tracking.authors = publication.authors
                tracking.file_urls = [str(url) for url in publication.file_urls]

                # Save new record to database
                sess.add(tracking)
                sess.commit()
                logger.info(f"Added tracking for publication: {publication.paper_id}")

                # Prepare object for return (detach from session)
                sess.refresh(tracking)
                sess.expunge(tracking)
                return tracking

            except Exception as e:
                # Roll back transaction on any error
                sess.rollback()
                logger.error(
                    f"Error adding/updating publication "
                    f"{publication.paper_id}: {str(e)}"
                )
                raise

    def add_publications(
        self,
        publications: list[GrowthLabPublication | OpenAlexPublication],
        session: Session | None = None,
    ) -> list[PublicationTracking]:
        """
        Add multiple publications to tracking.

        Args:
            publications: List of publications to track
            session: Optional database session to use

        Returns:
            List of tracking records
        """
        with self._get_session(session) as sess:
            tracking_records = []
            for pub in publications:
                tracking = self.add_publication(pub, session=sess)
                tracking_records.append(tracking)
            return tracking_records

    def _update_publication_status(
        self,
        publication_id: str,
        stage: str,
        status: DownloadStatus | ProcessingStatus | EmbeddingStatus | IngestionStatus,
        error: str | None = None,
        session: Session | None = None,
    ) -> bool:
        """
        Generic method to update any publication status stage.

        Args:
            publication_id: Unique identifier of the publication to update
            stage: Pipeline stage name (download, processing, embedding, ingestion)
            status: New status to set for the stage
            error: Optional error message if status indicates failure
            session: Optional database session (creates new one if not provided)

        Returns:
            True if publication was found and updated successfully, False otherwise
        """
        with self._get_session(session) as sess:
            # Find the publication tracking record
            stmt = select(PublicationTracking).where(
                PublicationTracking.publication_id == publication_id
            )
            pub = sess.exec(stmt).first()

            if not pub:
                logger.warning(f"Publication not found for tracking: {publication_id}")
                return False

            # Use the model's built-in method to update status with timestamp tracking
            update_method = getattr(pub, f"update_{stage}_status")
            # Ensure error is a string if provided
            # (handle case where it might be a dict)
            error_str = str(error) if error is not None else None
            update_method(status, error_str)
            sess.add(pub)
            sess.commit()
            return True

    def update_download_status(
        self,
        publication_id: str,
        status: DownloadStatus,
        error: str | None = None,
        session: Session | None = None,
    ) -> bool:
        """
        Update download status for a publication in the manifest.

        This method is called by the file downloader component to record
        progress through the download stage of the ETL pipeline.
        """
        return self._update_publication_status(
            publication_id, "download", status, error, session
        )

    def update_processing_status(
        self,
        publication_id: str,
        status: ProcessingStatus,
        error: str | None = None,
        session: Session | None = None,
    ) -> bool:
        """
        Update processing status for a publication in the manifest.

        This method is called by processing components to record
        progress through the processing stage of the ETL pipeline.
        """
        return self._update_publication_status(
            publication_id, "processing", status, error, session
        )

    def update_embedding_status(
        self,
        publication_id: str,
        status: EmbeddingStatus,
        error: str | None = None,
        session: Session | None = None,
    ) -> bool:
        """
        Update embedding status for a publication in the manifest.

        This method is called by embedding components to record
        progress through the embedding stage of the ETL pipeline.
        """
        return self._update_publication_status(
            publication_id, "embedding", status, error, session
        )

    def update_ingestion_status(
        self,
        publication_id: str,
        status: IngestionStatus,
        error: str | None = None,
        session: Session | None = None,
    ) -> bool:
        """
        Update ingestion status for a publication in the manifest.

        This method is called by ingestion components to record
        progress through the ingestion stage of the ETL pipeline.
        """
        return self._update_publication_status(
            publication_id, "ingestion", status, error, session
        )

    def get_publications_for_download(
        self, limit: int | None = None, session: Session | None = None
    ) -> list[PublicationTracking]:
        """
        Get publications that need to be downloaded.

        Args:
            limit: Optional limit on number of publications to return
            session: Optional database session to use

        Returns:
            List of publications to download
        """
        with self._get_session(session) as sess:
            stmt = select(PublicationTracking).where(
                PublicationTracking.download_status == DownloadStatus.PENDING
            )
            if limit:
                stmt = stmt.limit(limit)
            return list(sess.exec(stmt))

    def get_publications_for_processing(
        self, limit: int | None = None, session: Session | None = None
    ) -> list[PublicationTracking]:
        """
        Get publications that need to be processed.

        Only returns publications that have been successfully downloaded.

        Args:
            limit: Optional limit on number of publications to return
            session: Optional database session to use

        Returns:
            List of publications to process
        """
        with self._get_session(session) as sess:
            stmt = select(PublicationTracking).where(
                PublicationTracking.download_status == DownloadStatus.DOWNLOADED,
                PublicationTracking.processing_status == ProcessingStatus.PENDING,
            )
            if limit:
                stmt = stmt.limit(limit)
            return list(sess.exec(stmt))

    def get_publications_for_embedding(
        self, limit: int | None = None, session: Session | None = None
    ) -> list[PublicationTracking]:
        """
        Get publications that need to be embedded.

        Only returns publications that have been successfully processed.

        Args:
            limit: Optional limit on number of publications to return
            session: Optional database session to use

        Returns:
            List of publications to embed
        """
        with self._get_session(session) as sess:
            stmt = select(PublicationTracking).where(
                PublicationTracking.processing_status == ProcessingStatus.PROCESSED,
                PublicationTracking.embedding_status == EmbeddingStatus.PENDING,
            )
            if limit:
                stmt = stmt.limit(limit)
            return list(sess.exec(stmt))

    def get_publications_for_ingestion(
        self, limit: int | None = None, session: Session | None = None
    ) -> list[PublicationTracking]:
        """
        Get publications that need to be ingested.

        Only returns publications that have been successfully embedded.

        Args:
            limit: Optional limit on number of publications to return
            session: Optional database session to use

        Returns:
            List of publications to ingest
        """
        with self._get_session(session) as sess:
            stmt = select(PublicationTracking).where(
                PublicationTracking.embedding_status == EmbeddingStatus.EMBEDDED,
                PublicationTracking.ingestion_status == IngestionStatus.PENDING,
            )
            if limit:
                stmt = stmt.limit(limit)
            return list(sess.exec(stmt))

    def get_publication_status(
        self, publication_id: str, session: Session | None = None
    ) -> dict | None:
        """
        Get the current status of a publication.

        Args:
            publication_id: ID of the publication to check
            session: Optional database session to use

        Returns:
            Dictionary with status information or None if not found
        """
        with self._get_session(session) as sess:
            stmt = select(PublicationTracking).where(
                PublicationTracking.publication_id == publication_id
            )
            pub = sess.exec(stmt).first()

            if not pub:
                return None

            sess.refresh(pub)
            sess.expunge(pub)

            return {
                "publication_id": pub.publication_id,
                "title": pub.title,
                "download_status": pub.download_status,
                "processing_status": pub.processing_status,
                "embedding_status": pub.embedding_status,
                "ingestion_status": pub.ingestion_status,
                "error_message": pub.error_message,
            }
