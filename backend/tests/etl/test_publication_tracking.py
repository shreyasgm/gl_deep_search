"""
Tests for the publication tracking functionality.

This module tests the PublicationTracker class and related components that are
responsible for tracking publications through the ETL pipeline stages.
"""

import logging
import typing
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import HttpUrl, ValidationError
from sqlmodel import Session, SQLModel, create_engine, select

from backend.etl.models.publications import (
    GrowthLabPublication,
    OpenAlexPublication,
)
from backend.etl.models.tracking import (
    DownloadStatus,
    EmbeddingStatus,
    IngestionStatus,
    ProcessingStatus,
    PublicationTracking,
)
from backend.etl.utils.publication_tracker import ProcessingPlan, PublicationTracker

if typing.TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture


# Configure logger
logger = logging.getLogger(__name__)


@pytest.fixture
def in_memory_db() -> Any:
    """
    Create an in-memory SQLite database for testing.

    Returns:
        SQLAlchemy engine instance with in-memory database
    """
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    # Create all tables
    SQLModel.metadata.create_all(test_engine)

    return test_engine


@pytest.fixture
def mock_publication_tracker(in_memory_db: Any) -> PublicationTracker:
    """
    Create a PublicationTracker instance for testing.

    Args:
        in_memory_db: In-memory database engine

    Returns:
        PublicationTracker instance for testing
    """
    with patch("backend.etl.utils.publication_tracker.engine", in_memory_db):
        with patch("backend.etl.utils.publication_tracker.ensure_db_initialized"):
            tracker = PublicationTracker(ensure_db=False)

            # Mock scrapers
            tracker.growthlab_scraper = MagicMock()
            tracker.openalex_client = MagicMock()

            return tracker


@pytest.fixture
def sample_growthlab_publication() -> GrowthLabPublication:
    """
    Create a sample GrowthLab publication for testing.

    Returns:
        GrowthLabPublication instance
    """
    pub = GrowthLabPublication(
        title="Test GrowthLab Publication",
        authors="John Doe, Jane Smith",
        year=2023,
        abstract="This is a test abstract for a GrowthLab publication",
        pub_url=HttpUrl(
            "https://growthlab.hks.harvard.edu/publications/test-publication"
        ),
        file_urls=[
            HttpUrl("https://growthlab.hks.harvard.edu/files/growthlab/files/test.pdf"),
            HttpUrl(
                "https://growthlab.hks.harvard.edu/files/growthlab/files/test.docx"
            ),
        ],
        source="GrowthLab",
    )
    # Generate stable ID and content hash
    pub.paper_id = pub.generate_id()
    pub.content_hash = pub.generate_content_hash()
    return pub


@pytest.fixture
def sample_openalex_publication() -> OpenAlexPublication:
    """
    Create a sample OpenAlex publication for testing.

    Returns:
        OpenAlexPublication instance
    """
    pub = OpenAlexPublication(
        paper_id="W123456789",
        openalex_id="https://openalex.org/W123456789",
        title="Test OpenAlex Publication",
        authors="Alice Johnson, Bob Williams",
        year=2022,
        abstract="This is a test abstract for an OpenAlex publication",
        pub_url=HttpUrl("https://doi.org/10.1234/test.5678"),
        file_urls=[
            HttpUrl("https://doi.org/10.1234/test.5678"),
            HttpUrl("https://example.org/publications/test-paper.pdf"),
        ],
        source="OpenAlex",
        cited_by_count=42,
    )
    # Generate content hash
    pub.content_hash = pub.generate_content_hash()
    return pub


@pytest.fixture
def sample_tracking_publication(
    sample_growthlab_publication: GrowthLabPublication,
) -> PublicationTracking:
    """
    Create a sample PublicationTracking record for testing.

    Args:
        sample_growthlab_publication: Sample publication to create tracking for

    Returns:
        PublicationTracking instance
    """
    pub = sample_growthlab_publication
    tracking = PublicationTracking(
        publication_id=pub.paper_id,
        source_url=str(pub.pub_url) if pub.pub_url else "",
        title=pub.title,
        authors=pub.authors,
        year=pub.year,
        abstract=pub.abstract,
        content_hash=pub.content_hash,
    )
    tracking.file_urls = [str(url) for url in pub.file_urls]

    return tracking


class TestPublicationTracker:
    """Tests for the PublicationTracker class."""

    def test_initialization(self) -> None:
        """Test that the PublicationTracker initializes correctly."""
        with (
            patch("backend.etl.utils.publication_tracker.engine"),
            patch(
                "backend.etl.utils.publication_tracker.ensure_db_initialized"
            ) as mock_ensure_db,
        ):
            # Test with ensure_db=True (default)
            tracker = PublicationTracker()
            mock_ensure_db.assert_called_once()

            # Test with ensure_db=False
            mock_ensure_db.reset_mock()
            tracker = PublicationTracker(ensure_db=False)
            mock_ensure_db.assert_not_called()

            # Verify attributes
            assert hasattr(tracker, "growthlab_scraper")
            assert hasattr(tracker, "openalex_client")

    @pytest.mark.asyncio
    async def test_discover_publications(
        self,
        mock_publication_tracker: PublicationTracker,
        sample_growthlab_publication: GrowthLabPublication,
        sample_openalex_publication: OpenAlexPublication,
    ) -> None:
        """
        Test the discover_publications method returns publications from both sources.

        Args:
            mock_publication_tracker: Mocked PublicationTracker instance
            sample_growthlab_publication: Sample GrowthLab publication
            sample_openalex_publication: Sample OpenAlex publication
        """
        # Configure mocks
        mock_publication_tracker.growthlab_scraper.extract_and_enrich_publications = AsyncMock(
            return_value=[sample_growthlab_publication]
        )
        mock_publication_tracker.openalex_client.fetch_publications = AsyncMock(
            return_value=[sample_openalex_publication]
        )

        # Call the method
        publications = await mock_publication_tracker.discover_publications()

        # Verify results
        assert len(publications) == 2
        assert publications[0][0] == sample_growthlab_publication
        assert publications[0][1] == "growthlab"
        assert publications[1][0] == sample_openalex_publication
        assert publications[1][1] == "openalex"

        # Verify mock calls
        mock_publication_tracker.growthlab_scraper.extract_and_enrich_publications.assert_called_once()
        mock_publication_tracker.openalex_client.fetch_publications.assert_called_once()

    @pytest.mark.asyncio
    async def test_discover_publications_handles_errors(
        self, mock_publication_tracker: PublicationTracker, caplog: "LogCaptureFixture"
    ) -> None:
        """
        Test that discover_publications handles errors correctly.

        Args:
            mock_publication_tracker: Mocked PublicationTracker instance
            caplog: Fixture for capturing log messages
        """
        # Make the scrapers raise exceptions
        mock_publication_tracker.growthlab_scraper.extract_and_enrich_publications.side_effect = (
            Exception("Test error")
        )

        # Verify the method raises the exception
        with pytest.raises(Exception, match="Test error"):
            await mock_publication_tracker.discover_publications()

        # Verify error logging
        assert "Error discovering publications: Test error" in caplog.text

    def test_generate_processing_plan_new_publication(
        self,
        mock_publication_tracker: PublicationTracker,
        sample_growthlab_publication: GrowthLabPublication,
        in_memory_db: Any,
    ) -> None:
        """
        Test generating a processing plan for a new publication.

        Args:
            mock_publication_tracker: Mocked PublicationTracker instance
            sample_growthlab_publication: Sample publication
            in_memory_db: In-memory database
        """
        with Session(in_memory_db) as session:
            plan = mock_publication_tracker.generate_processing_plan(
                sample_growthlab_publication
            )

            assert plan.publication_id == sample_growthlab_publication.paper_id
            assert plan.needs_download is True
            assert plan.needs_processing is True
            assert plan.needs_embedding is True
            assert plan.needs_ingestion is True
            assert plan.files_to_reprocess == []
            assert plan.reason == "New publication"

    def test_generate_processing_plan_content_changed(
        self,
        mock_publication_tracker: PublicationTracker,
        sample_growthlab_publication: GrowthLabPublication,
        in_memory_db: Any,
    ) -> None:
        """
        Test generating a processing plan when content has changed.

        Args:
            mock_publication_tracker: Mocked PublicationTracker instance
            sample_growthlab_publication: Sample publication
            in_memory_db: In-memory database
        """
        with Session(in_memory_db) as session:
            # Add initial publication
            tracking = PublicationTracking(
                publication_id=sample_growthlab_publication.paper_id,
                source_url=str(sample_growthlab_publication.pub_url),
                title=sample_growthlab_publication.title,
                authors=sample_growthlab_publication.authors,
                year=sample_growthlab_publication.year,
                abstract=sample_growthlab_publication.abstract,
                content_hash="old_hash",
            )
            tracking.file_urls = [
                str(url) for url in sample_growthlab_publication.file_urls
            ]
            session.add(tracking)
            session.commit()
            tracking = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_growthlab_publication.paper_id
                )
            ).first()

            # Generate plan with new content hash
            plan = mock_publication_tracker.generate_processing_plan(
                sample_growthlab_publication, session=session
            )

            assert plan.publication_id == sample_growthlab_publication.paper_id
            assert plan.needs_download is True
            assert plan.needs_processing is True
            assert plan.needs_embedding is True
            assert plan.needs_ingestion is True
            assert plan.files_to_reprocess == [
                str(url) for url in sample_growthlab_publication.file_urls
            ]
            assert plan.reason == "Content hash changed"

    def test_generate_processing_plan_files_changed(
        self,
        mock_publication_tracker: PublicationTracker,
        sample_growthlab_publication: GrowthLabPublication,
        in_memory_db: Any,
    ) -> None:
        """
        Test generating a processing plan when files have changed.

        Args:
            mock_publication_tracker: Mocked PublicationTracker instance
            sample_growthlab_publication: Sample publication
            in_memory_db: In-memory database
        """
        with Session(in_memory_db) as session:
            # Add initial publication with different files
            tracking = PublicationTracking(
                publication_id=sample_growthlab_publication.paper_id,
                source_url=str(sample_growthlab_publication.pub_url),
                title=sample_growthlab_publication.title,
                authors=sample_growthlab_publication.authors,
                year=sample_growthlab_publication.year,
                abstract=sample_growthlab_publication.abstract,
                content_hash=sample_growthlab_publication.content_hash,
            )
            tracking.file_urls = ["https://example.com/old.pdf"]
            session.add(tracking)
            session.commit()
            tracking = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_growthlab_publication.paper_id
                )
            ).first()

            # Generate plan with new files
            plan = mock_publication_tracker.generate_processing_plan(
                sample_growthlab_publication, session=session
            )

            assert plan.publication_id == sample_growthlab_publication.paper_id
            assert plan.needs_download is True
            assert plan.needs_processing is True
            assert plan.needs_embedding is True
            assert plan.needs_ingestion is True
            assert set(plan.files_to_reprocess) == set(
                str(url) for url in sample_growthlab_publication.file_urls
            )
            assert plan.reason == "Files changed"

    def test_generate_processing_plan_status_based(
        self,
        mock_publication_tracker: PublicationTracker,
        sample_growthlab_publication: GrowthLabPublication,
        in_memory_db: Any,
    ) -> None:
        """
        Test generating a processing plan based on current status.

        Args:
            mock_publication_tracker: Mocked PublicationTracker instance
            sample_growthlab_publication: Sample publication
            in_memory_db: In-memory database
        """
        with Session(in_memory_db) as session:
            # Add publication with partial processing
            tracking = PublicationTracking(
                publication_id=sample_growthlab_publication.paper_id,
                source_url=str(sample_growthlab_publication.pub_url),
                title=sample_growthlab_publication.title,
                authors=sample_growthlab_publication.authors,
                year=sample_growthlab_publication.year,
                abstract=sample_growthlab_publication.abstract,
                content_hash=sample_growthlab_publication.content_hash,
                download_status=DownloadStatus.DOWNLOADED,
                processing_status=ProcessingStatus.PROCESSED,
                embedding_status=EmbeddingStatus.PENDING,
                ingestion_status=IngestionStatus.PENDING,
            )
            tracking.file_urls = [
                str(url) for url in sample_growthlab_publication.file_urls
            ]
            session.add(tracking)
            session.commit()
            tracking = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_growthlab_publication.paper_id
                )
            ).first()

            # Generate plan
            plan = mock_publication_tracker.generate_processing_plan(
                sample_growthlab_publication, session=session
            )

            assert plan.publication_id == sample_growthlab_publication.paper_id
            assert plan.needs_download is False
            assert plan.needs_processing is False
            assert plan.needs_embedding is True
            assert plan.needs_ingestion is True
            assert plan.files_to_reprocess == []
            assert plan.reason == "Status check"

    def test_add_publication_new(
        self,
        mock_publication_tracker: PublicationTracker,
        sample_growthlab_publication: GrowthLabPublication,
        in_memory_db: Any,
    ) -> None:
        """
        Test adding a new publication.

        Args:
            mock_publication_tracker: Mocked PublicationTracker instance
            sample_growthlab_publication: Sample publication
            in_memory_db: In-memory database
        """
        with Session(in_memory_db) as session:
            tracking = mock_publication_tracker.add_publication(
                sample_growthlab_publication, session=session
            )
            session.commit()
            # Re-query the object to ensure it's persistent
            tracking = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_growthlab_publication.paper_id
                )
            ).first()

            assert tracking.publication_id == sample_growthlab_publication.paper_id
            assert tracking.title == sample_growthlab_publication.title
            assert tracking.authors == sample_growthlab_publication.authors
            assert tracking.year == sample_growthlab_publication.year
            assert tracking.abstract == sample_growthlab_publication.abstract
            assert tracking.source_url == str(sample_growthlab_publication.pub_url)
            assert tracking.content_hash == sample_growthlab_publication.content_hash
            assert tracking.file_urls == [
                str(url) for url in sample_growthlab_publication.file_urls
            ]
            assert tracking.download_status == DownloadStatus.PENDING
            assert tracking.processing_status == ProcessingStatus.PENDING
            assert tracking.embedding_status == EmbeddingStatus.PENDING
            assert tracking.ingestion_status == IngestionStatus.PENDING

    def test_add_publication_update(
        self,
        mock_publication_tracker: PublicationTracker,
        sample_growthlab_publication: GrowthLabPublication,
        in_memory_db: Any,
    ) -> None:
        """
        Test updating an existing publication.

        Args:
            mock_publication_tracker: Mocked PublicationTracker instance
            sample_growthlab_publication: Sample publication
            in_memory_db: In-memory database
        """
        with Session(in_memory_db) as session:
            # Add initial publication
            tracking = PublicationTracking(
                publication_id=sample_growthlab_publication.paper_id,
                source_url=str(sample_growthlab_publication.pub_url),
                title="Old Title",
                authors="Old Authors",
                year=2022,
                abstract="Old Abstract",
                content_hash="old_hash",
            )
            tracking.file_urls = ["https://example.com/old.pdf"]
            session.add(tracking)
            session.commit()
            # Re-query the object
            tracking = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_growthlab_publication.paper_id
                )
            ).first()

            # Update publication
            updated = mock_publication_tracker.add_publication(
                sample_growthlab_publication, session=session
            )
            session.commit()
            updated = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_growthlab_publication.paper_id
                )
            ).first()

            assert updated.publication_id == sample_growthlab_publication.paper_id
            assert updated.title == sample_growthlab_publication.title
            assert updated.authors == sample_growthlab_publication.authors
            assert updated.year == sample_growthlab_publication.year
            assert updated.abstract == sample_growthlab_publication.abstract
            assert updated.source_url == str(sample_growthlab_publication.pub_url)
            assert updated.content_hash == sample_growthlab_publication.content_hash
            assert updated.file_urls == [
                str(url) for url in sample_growthlab_publication.file_urls
            ]
            assert updated.download_status == DownloadStatus.PENDING
            assert updated.processing_status == ProcessingStatus.PENDING
            assert updated.embedding_status == EmbeddingStatus.PENDING
            assert updated.ingestion_status == IngestionStatus.PENDING

    def test_update_download_status(
        self,
        mock_publication_tracker: PublicationTracker,
        sample_growthlab_publication: GrowthLabPublication,
        in_memory_db: Any,
    ) -> None:
        """
        Test updating download status.

        Args:
            mock_publication_tracker: Mocked PublicationTracker instance
            sample_growthlab_publication: Sample publication
            in_memory_db: In-memory database
        """
        with Session(in_memory_db) as session:
            # Add publication
            tracking = PublicationTracking(
                publication_id=sample_growthlab_publication.paper_id,
                source_url=str(sample_growthlab_publication.pub_url),
                title=sample_growthlab_publication.title,
                authors=sample_growthlab_publication.authors,
                year=sample_growthlab_publication.year,
                abstract=sample_growthlab_publication.abstract,
                content_hash=sample_growthlab_publication.content_hash,
            )
            tracking.file_urls = [
                str(url) for url in sample_growthlab_publication.file_urls
            ]
            session.add(tracking)
            session.commit()
            tracking = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_growthlab_publication.paper_id
                )
            ).first()

            # Update status
            success = mock_publication_tracker.update_download_status(
                sample_growthlab_publication.paper_id,
                DownloadStatus.DOWNLOADED,
                session=session,
            )
            assert success

            # Verify update
            with Session(in_memory_db) as verify_session:
                stmt = select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_growthlab_publication.paper_id
                )
                updated = verify_session.exec(stmt).first()
                assert updated.download_status == DownloadStatus.DOWNLOADED
                assert updated.download_timestamp is not None

    def test_update_processing_status(
        self,
        mock_publication_tracker: PublicationTracker,
        sample_tracking_publication: PublicationTracking,
        in_memory_db: Any,
    ) -> None:
        """
        Test updating processing status.

        Args:
            mock_publication_tracker: Mocked PublicationTracker instance
            sample_tracking_publication: Sample tracking record
            in_memory_db: In-memory database
        """
        with Session(in_memory_db) as session:
            # Add publication
            session.add(sample_tracking_publication)
            session.commit()
            sample_tracking_publication = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_tracking_publication.publication_id
                )
            ).first()

            # Update status
            success = mock_publication_tracker.update_processing_status(
                sample_tracking_publication.publication_id,
                ProcessingStatus.PROCESSED,
                session=session,
            )
            assert success

            # Verify update
            with Session(in_memory_db) as verify_session:
                stmt = select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_tracking_publication.publication_id
                )
                updated = verify_session.exec(stmt).first()
                assert updated.processing_status == ProcessingStatus.PROCESSED
                assert updated.processing_timestamp is not None

    def test_update_embedding_status(
        self,
        mock_publication_tracker: PublicationTracker,
        sample_tracking_publication: PublicationTracking,
        in_memory_db: Any,
    ) -> None:
        """
        Test updating embedding status.

        Args:
            mock_publication_tracker: Mocked PublicationTracker instance
            sample_tracking_publication: Sample tracking record
            in_memory_db: In-memory database
        """
        with Session(in_memory_db) as session:
            # Add publication
            session.add(sample_tracking_publication)
            session.commit()
            sample_tracking_publication = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_tracking_publication.publication_id
                )
            ).first()

            # Update status
            success = mock_publication_tracker.update_embedding_status(
                sample_tracking_publication.publication_id,
                EmbeddingStatus.EMBEDDED,
                session=session,
            )
            assert success

            # Verify update
            with Session(in_memory_db) as verify_session:
                stmt = select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_tracking_publication.publication_id
                )
                updated = verify_session.exec(stmt).first()
                assert updated.embedding_status == EmbeddingStatus.EMBEDDED
                assert updated.embedding_timestamp is not None

    def test_update_ingestion_status(
        self,
        mock_publication_tracker: PublicationTracker,
        sample_tracking_publication: PublicationTracking,
        in_memory_db: Any,
    ) -> None:
        """
        Test updating ingestion status.

        Args:
            mock_publication_tracker: Mocked PublicationTracker instance
            sample_tracking_publication: Sample tracking record
            in_memory_db: In-memory database
        """
        with Session(in_memory_db) as session:
            # Add publication
            session.add(sample_tracking_publication)
            session.commit()
            sample_tracking_publication = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_tracking_publication.publication_id
                )
            ).first()

            # Update status
            success = mock_publication_tracker.update_ingestion_status(
                sample_tracking_publication.publication_id,
                IngestionStatus.INGESTED,
                session=session,
            )
            assert success

            # Verify update
            with Session(in_memory_db) as verify_session:
                stmt = select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_tracking_publication.publication_id
                )
                updated = verify_session.exec(stmt).first()
                assert updated.ingestion_status == IngestionStatus.INGESTED
                assert updated.ingestion_timestamp is not None

    def test_get_publications_for_download(
        self, mock_publication_tracker: PublicationTracker, in_memory_db: Any
    ) -> None:
        """
        Test getting publications for download.

        Args:
            mock_publication_tracker: Mocked PublicationTracker instance
            in_memory_db: In-memory database
        """
        with Session(in_memory_db) as session:
            # Add publications with different statuses
            pub1 = PublicationTracking(
                publication_id="pub1",
                source_url="https://example.com/pub1",
                title="Publication 1",
                authors="Author 1",
                year=2023,
                abstract="Abstract 1",
                content_hash="hash1",
                download_status=DownloadStatus.PENDING,
            )
            pub2 = PublicationTracking(
                publication_id="pub2",
                source_url="https://example.com/pub2",
                title="Publication 2",
                authors="Author 2",
                year=2023,
                abstract="Abstract 2",
                content_hash="hash2",
                download_status=DownloadStatus.DOWNLOADED,
            )
            session.add(pub1)
            session.add(pub2)
            session.commit()
            pub1 = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id == "pub1"
                )
            ).first()
            pub2 = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id == "pub2"
                )
            ).first()

            # Get publications for download
            pubs = mock_publication_tracker.get_publications_for_download(
                session=session
            )
            assert len(pubs) == 1
            assert pubs[0].publication_id == "pub1"

    def test_get_publications_for_processing(
        self, mock_publication_tracker: PublicationTracker, in_memory_db: Any
    ) -> None:
        """
        Test getting publications for processing.

        Args:
            mock_publication_tracker: Mocked PublicationTracker instance
            in_memory_db: In-memory database
        """
        with Session(in_memory_db) as session:
            # Add publications with different statuses
            pub1 = PublicationTracking(
                publication_id="pub1",
                source_url="https://example.com/pub1",
                title="Publication 1",
                authors="Author 1",
                year=2023,
                abstract="Abstract 1",
                content_hash="hash1",
                download_status=DownloadStatus.DOWNLOADED,
                processing_status=ProcessingStatus.PENDING,
            )
            pub2 = PublicationTracking(
                publication_id="pub2",
                source_url="https://example.com/pub2",
                title="Publication 2",
                authors="Author 2",
                year=2023,
                abstract="Abstract 2",
                content_hash="hash2",
                download_status=DownloadStatus.DOWNLOADED,
                processing_status=ProcessingStatus.PROCESSED,
            )
            session.add(pub1)
            session.add(pub2)
            session.commit()
            pub1 = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id == "pub1"
                )
            ).first()
            pub2 = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id == "pub2"
                )
            ).first()

            # Get publications for processing
            pubs = mock_publication_tracker.get_publications_for_processing(
                session=session
            )
            assert len(pubs) == 1
            assert pubs[0].publication_id == "pub1"

    def test_get_publications_for_embedding(
        self, mock_publication_tracker: PublicationTracker, in_memory_db: Any
    ) -> None:
        """
        Test getting publications for embedding.

        Args:
            mock_publication_tracker: Mocked PublicationTracker instance
            in_memory_db: In-memory database
        """
        with Session(in_memory_db) as session:
            # Add publications with different statuses
            pub1 = PublicationTracking(
                publication_id="pub1",
                source_url="https://example.com/pub1",
                title="Publication 1",
                authors="Author 1",
                year=2023,
                abstract="Abstract 1",
                content_hash="hash1",
                download_status=DownloadStatus.DOWNLOADED,
                processing_status=ProcessingStatus.PROCESSED,
                embedding_status=EmbeddingStatus.PENDING,
            )
            pub2 = PublicationTracking(
                publication_id="pub2",
                source_url="https://example.com/pub2",
                title="Publication 2",
                authors="Author 2",
                year=2023,
                abstract="Abstract 2",
                content_hash="hash2",
                download_status=DownloadStatus.DOWNLOADED,
                processing_status=ProcessingStatus.PROCESSED,
                embedding_status=EmbeddingStatus.EMBEDDED,
            )
            session.add(pub1)
            session.add(pub2)
            session.commit()
            pub1 = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id == "pub1"
                )
            ).first()
            pub2 = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id == "pub2"
                )
            ).first()

            # Get publications for embedding
            pubs = mock_publication_tracker.get_publications_for_embedding(
                session=session
            )
            assert len(pubs) == 1
            assert pubs[0].publication_id == "pub1"

    def test_get_publications_for_ingestion(
        self, mock_publication_tracker: PublicationTracker, in_memory_db: Any
    ) -> None:
        """
        Test getting publications for ingestion.

        Args:
            mock_publication_tracker: Mocked PublicationTracker instance
            in_memory_db: In-memory database
        """
        with Session(in_memory_db) as session:
            # Add publications with different statuses
            pub1 = PublicationTracking(
                publication_id="pub1",
                source_url="https://example.com/pub1",
                title="Publication 1",
                authors="Author 1",
                year=2023,
                abstract="Abstract 1",
                content_hash="hash1",
                download_status=DownloadStatus.DOWNLOADED,
                processing_status=ProcessingStatus.PROCESSED,
                embedding_status=EmbeddingStatus.EMBEDDED,
                ingestion_status=IngestionStatus.PENDING,
            )
            pub2 = PublicationTracking(
                publication_id="pub2",
                source_url="https://example.com/pub2",
                title="Publication 2",
                authors="Author 2",
                year=2023,
                abstract="Abstract 2",
                content_hash="hash2",
                download_status=DownloadStatus.DOWNLOADED,
                processing_status=ProcessingStatus.PROCESSED,
                embedding_status=EmbeddingStatus.EMBEDDED,
                ingestion_status=IngestionStatus.INGESTED,
            )
            session.add(pub1)
            session.add(pub2)
            session.commit()
            pub1 = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id == "pub1"
                )
            ).first()
            pub2 = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id == "pub2"
                )
            ).first()

            # Get publications for ingestion
            pubs = mock_publication_tracker.get_publications_for_ingestion(
                session=session
            )
            assert len(pubs) == 1
            assert pubs[0].publication_id == "pub1"

    def test_get_publication_status(
        self,
        mock_publication_tracker: PublicationTracker,
        sample_tracking_publication: PublicationTracking,
        in_memory_db: Any,
    ) -> None:
        """
        Test getting publication status.

        Args:
            mock_publication_tracker: Mocked PublicationTracker instance
            sample_tracking_publication: Sample tracking record
            in_memory_db: In-memory database
        """
        with Session(in_memory_db) as session:
            # Add publication
            session.add(sample_tracking_publication)
            session.commit()
            sample_tracking_publication = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_tracking_publication.publication_id
                )
            ).first()

            # Get status
            status = mock_publication_tracker.get_publication_status(
                sample_tracking_publication.publication_id, session=session
            )
            assert status is not None
            assert (
                status["publication_id"] == sample_tracking_publication.publication_id
            )
            assert status["title"] == sample_tracking_publication.title
            assert status["download_status"] == DownloadStatus.PENDING
            assert status["processing_status"] == ProcessingStatus.PENDING
            assert status["embedding_status"] == EmbeddingStatus.PENDING
            assert status["ingestion_status"] == IngestionStatus.PENDING


class TestPublicationTracking:
    """Tests for the PublicationTracking model."""

    def test_year_validation(self) -> None:
        """Test year validation."""
        # Valid year
        tracking = PublicationTracking(
            publication_id="test123",
            source_url="https://example.com/test",
            title="Test Paper",
            authors="Test Author",
            year=2023,
            abstract="Test abstract",
            content_hash="abc123",
        )
        assert tracking.year == 2023

        # Invalid year
        with pytest.raises(ValidationError):
            PublicationTracking.model_validate(
                {
                    "publication_id": "test123",
                    "source_url": "https://example.com/test",
                    "title": "Test Paper",
                    "authors": "Test Author",
                    "year": 1800,  # Too old
                    "abstract": "Test abstract",
                    "content_hash": "abc123",
                }
            )

    def test_file_urls_property(self) -> None:
        """Test file_urls property."""
        tracking = PublicationTracking(
            publication_id="test123",
            source_url="https://example.com/test",
            title="Test Paper",
            authors="Test Author",
            year=2023,
            abstract="Test abstract",
            content_hash="abc123",
        )
        tracking.file_urls = ["https://example.com/paper.pdf"]
        assert tracking.file_urls == ["https://example.com/paper.pdf"]

    def test_update_download_status(self) -> None:
        """Test update_download_status method."""
        tracking = PublicationTracking(
            publication_id="test123",
            source_url="https://example.com/test",
            title="Test Paper",
            authors="Test Author",
            year=2023,
            abstract="Test abstract",
            content_hash="abc123",
        )
        tracking.update_download_status(DownloadStatus.DOWNLOADED)
        assert tracking.download_status == DownloadStatus.DOWNLOADED
        assert tracking.download_timestamp is not None

    def test_update_processing_status(self) -> None:
        """Test update_processing_status method."""
        tracking = PublicationTracking(
            publication_id="test123",
            source_url="https://example.com/test",
            title="Test Paper",
            authors="Test Author",
            year=2023,
            abstract="Test abstract",
            content_hash="abc123",
        )
        tracking.update_processing_status(ProcessingStatus.PROCESSED)
        assert tracking.processing_status == ProcessingStatus.PROCESSED
        assert tracking.processing_timestamp is not None

    def test_update_embedding_status(self) -> None:
        """Test update_embedding_status method."""
        tracking = PublicationTracking(
            publication_id="test123",
            source_url="https://example.com/test",
            title="Test Paper",
            authors="Test Author",
            year=2023,
            abstract="Test abstract",
            content_hash="abc123",
        )
        tracking.update_embedding_status(EmbeddingStatus.EMBEDDED)
        assert tracking.embedding_status == EmbeddingStatus.EMBEDDED
        assert tracking.embedding_timestamp is not None

    def test_update_ingestion_status(self) -> None:
        """Test update_ingestion_status method."""
        tracking = PublicationTracking(
            publication_id="test123",
            source_url="https://example.com/test",
            title="Test Paper",
            authors="Test Author",
            year=2023,
            abstract="Test abstract",
            content_hash="abc123",
        )
        tracking.update_ingestion_status(IngestionStatus.INGESTED)
        assert tracking.ingestion_status == IngestionStatus.INGESTED
        assert tracking.ingestion_timestamp is not None


class TestProcessingPlan:
    """Tests for the ProcessingPlan class."""

    def test_processing_plan_creation(self) -> None:
        """Test ProcessingPlan creation."""
        plan = ProcessingPlan(
            publication_id="test123",
            needs_download=True,
            needs_processing=True,
            needs_embedding=True,
            needs_ingestion=True,
            files_to_reprocess=["file1.pdf", "file2.pdf"],
            reason="Test reason",
        )
        assert plan.publication_id == "test123"
        assert plan.needs_download is True
        assert plan.needs_processing is True
        assert plan.needs_embedding is True
        assert plan.needs_ingestion is True
        assert plan.files_to_reprocess == ["file1.pdf", "file2.pdf"]
        assert plan.reason == "Test reason"
