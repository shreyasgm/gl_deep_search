"""
Publication tracking utility for ETL pipeline.

This module provides functionality to track publications through the ETL pipeline
stages (discovery, download, processing, embedding, ingestion).
"""

import logging
from dataclasses import dataclass

from sqlmodel import Session, SQLModel, select

from backend.etl.models.publications import GrowthLabPublication, OpenAlexPublication
from backend.etl.models.tracking import (
    DownloadStatus,
    EmbeddingStatus,
    IngestionStatus,
    ProcessingStatus,
    PublicationTracking,
)
from backend.etl.scrapers.growthlab import GrowthLabScraper
from backend.etl.scrapers.openalex import OpenAlexClient
from backend.storage.database import engine, ensure_db_initialized

logger = logging.getLogger(__name__)


@dataclass
class ProcessingPlan:
    """Represents a plan for processing a publication through the ETL pipeline."""

    publication_id: str
    needs_download: bool
    needs_processing: bool
    needs_embedding: bool
    needs_ingestion: bool
    files_to_reprocess: list[str]
    reason: str


class PublicationTracker:
    """
    Manages the tracking of publications through the ETL pipeline.
    """

    def __init__(self, ensure_db: bool = True):
        """
        Initialize the publication tracker.

        Args:
            ensure_db: Whether to ensure the database is initialized
        """
        if ensure_db:
            ensure_db_initialized()
            SQLModel.metadata.create_all(engine)
        self.growthlab_scraper = GrowthLabScraper()
        self.openalex_client = OpenAlexClient()

    async def discover_publications(
        self
    ) -> list[tuple[GrowthLabPublication | OpenAlexPublication, str]]:
        """
        Discover new publications using configured scrapers.

        Returns:
            List of tuples containing (publication, source)
        """
        publications = []

        try:
            # Discover from GrowthLab
            growthlab_pubs = await self.growthlab_scraper.extract_and_enrich_publications()
            publications.extend([(pub, "growthlab") for pub in growthlab_pubs])

            # Discover from OpenAlex
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
        Generate a processing plan for a publication.

        Args:
            publication: The publication to plan for
            session: Optional database session to use

        Returns:
            ProcessingPlan object detailing what needs to be done
        """
        close_session = False
        if session is None:
            session = Session(engine)
            close_session = True
        try:
            # Check if publication exists
            stmt = select(PublicationTracking).where(
                PublicationTracking.publication_id == publication.paper_id
            )
            existing = session.exec(stmt).first()

            if not existing:
                # New publication needs everything
                return ProcessingPlan(
                    publication_id=publication.paper_id,
                    needs_download=True,
                    needs_processing=True,
                    needs_embedding=True,
                    needs_ingestion=True,
                    files_to_reprocess=[],
                    reason="New publication",
                )

            # Check content hash for changes
            if existing.content_hash != publication.content_hash:
                # Content changed, need to reprocess everything
                return ProcessingPlan(
                    publication_id=publication.paper_id,
                    needs_download=True,
                    needs_processing=True,
                    needs_embedding=True,
                    needs_ingestion=True,
                    files_to_reprocess=[str(url) for url in publication.file_urls],
                    reason="Content hash changed",
                )

            # Check file URLs for changes
            existing_files = set(existing.file_urls)
            new_files = set([str(url) for url in publication.file_urls])
            added_files = new_files - existing_files
            removed_files = existing_files - new_files

            if added_files or removed_files:
                # Files changed, need to reprocess
                return ProcessingPlan(
                    publication_id=publication.paper_id,
                    needs_download=True,
                    needs_processing=True,
                    needs_embedding=True,
                    needs_ingestion=True,
                    files_to_reprocess=list(added_files),
                    reason="Files changed",
                )

            # Check status of each stage
            needs_download = existing.download_status != DownloadStatus.DOWNLOADED
            needs_processing = (
                needs_download
                or existing.processing_status != ProcessingStatus.PROCESSED
            )
            needs_embedding = (
                needs_processing
                or existing.embedding_status != EmbeddingStatus.EMBEDDED
            )
            needs_ingestion = (
                needs_embedding or existing.ingestion_status != IngestionStatus.INGESTED
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
        finally:
            if close_session:
                session.close()

    def add_publication(
        self,
        publication: GrowthLabPublication | OpenAlexPublication,
        session: Session | None = None,
    ) -> PublicationTracking:
        """
        Add a publication to tracking or update if it already exists.

        Args:
            publication: The publication to track
            session: Optional database session to use

        Returns:
            The tracking record

        Raises:
            ValueError: If publication data is invalid
            sqlalchemy.exc.IntegrityError: If there's a database constraint violation
        """
        close_session = False
        if session is None:
            session = Session(engine)
            close_session = True

        try:
            # Generate processing plan
            plan = self.generate_processing_plan(publication, session=session)

            # Check if publication already exists
            stmt = select(PublicationTracking).where(
                PublicationTracking.publication_id == publication.paper_id
            )
            existing = session.exec(stmt).first()

            if existing:
                # Update existing record
                existing.title = publication.title
                existing.authors = publication.authors
                existing.year = publication.year
                existing.abstract = publication.abstract
                existing.source_url = (
                    str(publication.pub_url) if publication.pub_url else ""
                )
                existing.content_hash = publication.content_hash
                existing.file_urls = [str(url) for url in publication.file_urls]

                # Reset statuses based on processing plan
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

                session.add(existing)
                session.commit()
                logger.info(
                    f"Updated tracking for publication: "
                    f"{publication.paper_id} - {plan.reason}"
                )

                session.refresh(existing)
                session.expunge(existing)
                return existing

            # Create new tracking record
            tracking = PublicationTracking(
                publication_id=publication.paper_id,
                source_url=str(publication.pub_url) if publication.pub_url else "",
                title=publication.title,
                authors=publication.authors,
                year=publication.year,
                abstract=publication.abstract,
                content_hash=publication.content_hash,
            )

            # Set file URLs
            tracking.file_urls = [str(url) for url in publication.file_urls]

            # Add to database
            session.add(tracking)
            session.commit()
            logger.info(f"Added tracking for publication: {publication.paper_id}")

            session.refresh(tracking)
            session.expunge(tracking)
            return tracking

        except Exception as e:
            session.rollback()
            logger.error(
                f"Error adding/updating publication {publication.paper_id}: {str(e)}"
            )
            raise

        finally:
            if close_session:
                session.close()

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
        close_session = False
        if session is None:
            session = Session(engine)
            close_session = True
        try:
            tracking_records = []
            for pub in publications:
                tracking = self.add_publication(pub, session=session)
                tracking_records.append(tracking)
            return tracking_records
        finally:
            if close_session:
                session.close()

    def update_download_status(
        self,
        publication_id: str,
        status: DownloadStatus,
        error: str | None = None,
        session: Session | None = None,
    ) -> bool:
        """
        Update download status for a publication.

        Args:
            publication_id: ID of the publication to update
            status: New download status
            error: Optional error message
            session: Optional database session to use

        Returns:
            True if successfully updated
        """
        close_session = False
        if session is None:
            session = Session(engine)
            close_session = True
        try:
            stmt = select(PublicationTracking).where(
                PublicationTracking.publication_id == publication_id
            )
            pub = session.exec(stmt).first()

            if not pub:
                logger.warning(f"Publication not found for tracking: {publication_id}")
                return False

            pub.update_download_status(status, error)
            session.add(pub)
            session.commit()
            return True
        finally:
            if close_session:
                session.close()

    def update_processing_status(
        self,
        publication_id: str,
        status: ProcessingStatus,
        error: str | None = None,
        session: Session | None = None,
    ) -> bool:
        """
        Update processing status for a publication.

        Args:
            publication_id: ID of the publication to update
            status: New processing status
            error: Optional error message
            session: Optional database session to use

        Returns:
            True if successfully updated
        """
        close_session = False
        if session is None:
            session = Session(engine)
            close_session = True
        try:
            stmt = select(PublicationTracking).where(
                PublicationTracking.publication_id == publication_id
            )
            pub = session.exec(stmt).first()

            if not pub:
                logger.warning(f"Publication not found for tracking: {publication_id}")
                return False

            pub.update_processing_status(status, error)
            session.add(pub)
            session.commit()
            return True
        finally:
            if close_session:
                session.close()

    def update_embedding_status(
        self,
        publication_id: str,
        status: EmbeddingStatus,
        error: str | None = None,
        session: Session | None = None,
    ) -> bool:
        """
        Update embedding status for a publication.

        Args:
            publication_id: ID of the publication to update
            status: New embedding status
            error: Optional error message
            session: Optional database session to use

        Returns:
            True if successfully updated
        """
        close_session = False
        if session is None:
            session = Session(engine)
            close_session = True
        try:
            stmt = select(PublicationTracking).where(
                PublicationTracking.publication_id == publication_id
            )
            pub = session.exec(stmt).first()

            if not pub:
                logger.warning(f"Publication not found for tracking: {publication_id}")
                return False

            pub.update_embedding_status(status, error)
            session.add(pub)
            session.commit()
            return True
        finally:
            if close_session:
                session.close()

    def update_ingestion_status(
        self,
        publication_id: str,
        status: IngestionStatus,
        error: str | None = None,
        session: Session | None = None,
    ) -> bool:
        """
        Update ingestion status for a publication.

        Args:
            publication_id: ID of the publication to update
            status: New ingestion status
            error: Optional error message
            session: Optional database session to use

        Returns:
            True if successfully updated
        """
        close_session = False
        if session is None:
            session = Session(engine)
            close_session = True
        try:
            stmt = select(PublicationTracking).where(
                PublicationTracking.publication_id == publication_id
            )
            pub = session.exec(stmt).first()

            if not pub:
                logger.warning(f"Publication not found for tracking: {publication_id}")
                return False

            pub.update_ingestion_status(status, error)
            session.add(pub)
            session.commit()
            return True
        finally:
            if close_session:
                session.close()

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
        close_session = False
        if session is None:
            session = Session(engine)
            close_session = True
        try:
            stmt = select(PublicationTracking).where(
                PublicationTracking.download_status == DownloadStatus.PENDING
            )
            if limit:
                stmt = stmt.limit(limit)
            return list(session.exec(stmt))
        finally:
            if close_session:
                session.close()

    def get_publications_for_processing(
        self, limit: int | None = None, session: Session | None = None
    ) -> list[PublicationTracking]:
        """
        Get publications that need to be processed.

        Args:
            limit: Optional limit on number of publications to return
            session: Optional database session to use

        Returns:
            List of publications to process
        """
        close_session = False
        if session is None:
            session = Session(engine)
            close_session = True
        try:
            stmt = select(PublicationTracking).where(
                PublicationTracking.download_status == DownloadStatus.DOWNLOADED,
                PublicationTracking.processing_status == ProcessingStatus.PENDING,
            )
            if limit:
                stmt = stmt.limit(limit)
            return list(session.exec(stmt))
        finally:
            if close_session:
                session.close()

    def get_publications_for_embedding(
        self, limit: int | None = None, session: Session | None = None
    ) -> list[PublicationTracking]:
        """
        Get publications that need to be embedded.

        Args:
            limit: Optional limit on number of publications to return
            session: Optional database session to use

        Returns:
            List of publications to embed
        """
        close_session = False
        if session is None:
            session = Session(engine)
            close_session = True
        try:
            stmt = select(PublicationTracking).where(
                PublicationTracking.processing_status == ProcessingStatus.PROCESSED,
                PublicationTracking.embedding_status == EmbeddingStatus.PENDING,
            )
            if limit:
                stmt = stmt.limit(limit)
            return list(session.exec(stmt))
        finally:
            if close_session:
                session.close()

    def get_publications_for_ingestion(
        self, limit: int | None = None, session: Session | None = None
    ) -> list[PublicationTracking]:
        """
        Get publications that need to be ingested.

        Args:
            limit: Optional limit on number of publications to return
            session: Optional database session to use

        Returns:
            List of publications to ingest
        """
        close_session = False
        if session is None:
            session = Session(engine)
            close_session = True
        try:
            stmt = select(PublicationTracking).where(
                PublicationTracking.embedding_status == EmbeddingStatus.EMBEDDED,
                PublicationTracking.ingestion_status == IngestionStatus.PENDING,
            )
            if limit:
                stmt = stmt.limit(limit)
            return list(session.exec(stmt))
        finally:
            if close_session:
                session.close()

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
        close_session = False
        if session is None:
            session = Session(engine)
            close_session = True
        try:
            stmt = select(PublicationTracking).where(
                PublicationTracking.publication_id == publication_id
            )
            pub = session.exec(stmt).first()

            if not pub:
                return None

            session.refresh(pub)
            session.expunge(pub)

            return {
                "publication_id": pub.publication_id,
                "title": pub.title,
                "download_status": pub.download_status,
                "processing_status": pub.processing_status,
                "embedding_status": pub.embedding_status,
                "ingestion_status": pub.ingestion_status,
                "error_message": pub.error_message,
            }
        finally:
            if close_session:
                session.close()
