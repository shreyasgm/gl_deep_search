"""
Publication tracking utility for ETL pipeline.

This module provides functionality to track publications through the ETL pipeline
stages (discovery, download, processing, embedding, ingestion).
"""

import logging

from sqlmodel import Session, select

from backend.etl.models.publications import GrowthLabPublication, OpenAlexPublication
from backend.etl.models.tracking import (
    DownloadStatus,
    EmbeddingStatus,
    IngestionStatus,
    ProcessingStatus,
    PublicationTracking,
)
from backend.storage.database import engine, ensure_db_initialized

logger = logging.getLogger(__name__)


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
        """
        close_session = False
        if session is None:
            session = Session(engine)
            close_session = True

        try:
            # Check if publication already exists
            stmt = select(PublicationTracking).where(
                PublicationTracking.publication_id == publication.paper_id
            )
            existing = session.exec(stmt).first()

            if existing:
                # Update existing record if content hash differs
                if existing.content_hash != publication.content_hash:
                    existing.title = publication.title
                    existing.authors = publication.authors
                    existing.year = publication.year
                    existing.abstract = publication.abstract
                    existing.source_url = (
                        str(publication.pub_url) if publication.pub_url else ""
                    )
                    existing.content_hash = publication.content_hash
                    existing.file_urls = [str(url) for url in publication.file_urls]
                    session.add(existing)
                    session.commit()
                    logger.info(
                        f"Updated tracking for publication: {publication.paper_id}"
                    )

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

            return tracking

        finally:
            if close_session:
                session.close()

    def add_publications(
        self,
        publications: list[GrowthLabPublication | OpenAlexPublication],
    ) -> list[PublicationTracking]:
        """
        Add multiple publications to tracking.

        Args:
            publications: List of publications to track

        Returns:
            List of tracking records
        """
        with Session(engine) as session:
            tracking_records = []
            for pub in publications:
                tracking = self.add_publication(pub, session=session)
                tracking_records.append(tracking)
            return tracking_records

    def update_download_status(
        self,
        publication_id: str,
        status: DownloadStatus,
        error: str | None = None,
    ) -> bool:
        """
        Update download status for a publication.

        Args:
            publication_id: ID of the publication to update
            status: New download status
            error: Optional error message

        Returns:
            True if successfully updated
        """
        with Session(engine) as session:
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

    def update_processing_status(
        self,
        publication_id: str,
        status: ProcessingStatus,
        error: str | None = None,
    ) -> bool:
        """
        Update processing status for a publication.

        Args:
            publication_id: ID of the publication to update
            status: New processing status
            error: Optional error message

        Returns:
            True if successfully updated
        """
        with Session(engine) as session:
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

    def update_embedding_status(
        self,
        publication_id: str,
        status: EmbeddingStatus,
        error: str | None = None,
    ) -> bool:
        """
        Update embedding status for a publication.

        Args:
            publication_id: ID of the publication to update
            status: New embedding status
            error: Optional error message

        Returns:
            True if successfully updated
        """
        with Session(engine) as session:
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

    def update_ingestion_status(
        self,
        publication_id: str,
        status: IngestionStatus,
        error: str | None = None,
    ) -> bool:
        """
        Update ingestion status for a publication.

        Args:
            publication_id: ID of the publication to update
            status: New ingestion status
            error: Optional error message

        Returns:
            True if successfully updated
        """
        with Session(engine) as session:
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

    def get_publications_for_download(
        self, limit: int | None = None
    ) -> list[PublicationTracking]:
        """
        Get publications that need to be downloaded.

        Args:
            limit: Optional limit on number of publications to return

        Returns:
            List of publications to download
        """
        with Session(engine) as session:
            stmt = select(PublicationTracking).where(
                PublicationTracking.download_status == DownloadStatus.PENDING
            )
            if limit:
                stmt = stmt.limit(limit)
            return list(session.exec(stmt))

    def get_publications_for_processing(
        self, limit: int | None = None
    ) -> list[PublicationTracking]:
        """
        Get publications that need to be processed.

        Args:
            limit: Optional limit on number of publications to return

        Returns:
            List of publications to process
        """
        with Session(engine) as session:
            stmt = select(PublicationTracking).where(
                PublicationTracking.download_status == DownloadStatus.DOWNLOADED,
                PublicationTracking.processing_status == ProcessingStatus.PENDING,
            )
            if limit:
                stmt = stmt.limit(limit)
            return list(session.exec(stmt))

    def get_publications_for_embedding(
        self, limit: int | None = None
    ) -> list[PublicationTracking]:
        """
        Get publications that need to be embedded.

        Args:
            limit: Optional limit on number of publications to return

        Returns:
            List of publications to embed
        """
        with Session(engine) as session:
            stmt = select(PublicationTracking).where(
                PublicationTracking.processing_status == ProcessingStatus.PROCESSED,
                PublicationTracking.embedding_status == EmbeddingStatus.PENDING,
            )
            if limit:
                stmt = stmt.limit(limit)
            return list(session.exec(stmt))

    def get_publications_for_ingestion(
        self, limit: int | None = None
    ) -> list[PublicationTracking]:
        """
        Get publications that need to be ingested.

        Args:
            limit: Optional limit on number of publications to return

        Returns:
            List of publications to ingest
        """
        with Session(engine) as session:
            stmt = select(PublicationTracking).where(
                PublicationTracking.embedding_status == EmbeddingStatus.EMBEDDED,
                PublicationTracking.ingestion_status == IngestionStatus.PENDING,
            )
            if limit:
                stmt = stmt.limit(limit)
            return list(session.exec(stmt))

    def get_publication_status(self, publication_id: str) -> dict | None:
        """
        Get the current status of a publication.

        Args:
            publication_id: ID of the publication to check

        Returns:
            Dictionary with status information or None if not found
        """
        with Session(engine) as session:
            stmt = select(PublicationTracking).where(
                PublicationTracking.publication_id == publication_id
            )
            pub = session.exec(stmt).first()

            if not pub:
                return None

            return {
                "publication_id": pub.publication_id,
                "title": pub.title,
                "download_status": pub.download_status,
                "processing_status": pub.processing_status,
                "embedding_status": pub.embedding_status,
                "ingestion_status": pub.ingestion_status,
                "error_message": pub.error_message,
            }
