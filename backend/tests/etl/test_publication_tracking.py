"""
Integration tests for publication tracking system using real publications.
Unit tests are below starting from line 600

This module tests the complete publication manifest and tracking system
using actual publications from real sources to validate end-to-end functionality.
It tests the core components from Issues #19 (Publication Manifest System)
and #8 (Check for new publications).

These tests require network access and may be slower than unit tests.
"""

import logging
import os
import tempfile
import typing
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from pydantic import HttpUrl, ValidationError
from sqlmodel import Session, SQLModel, create_engine, select

from backend.etl.models.publications import GrowthLabPublication, OpenAlexPublication
from backend.etl.models.tracking import (
    DownloadStatus,
    EmbeddingStatus,
    IngestionStatus,
    ProcessingStatus,
    PublicationTracking,
)
from backend.etl.utils.publication_tracker import ProcessingPlan, PublicationTracker

# Configure logger
logger = logging.getLogger(__name__)


@pytest.fixture
def real_test_db():
    """
    Create a temporary database for real integration tests.

    Returns:
        SQLAlchemy engine instance with temporary database
    """
    # Create temporary database file
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)

    test_engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )

    # Create all tables
    SQLModel.metadata.create_all(test_engine)

    yield test_engine

    # Cleanup
    try:
        os.unlink(db_path)
    except OSError:
        pass


@pytest.fixture
def real_publication_tracker(real_test_db):
    """
    Create a PublicationTracker with real scrapers for integration testing.

    Args:
        real_test_db: Temporary database engine

    Returns:
        PublicationTracker instance with real scrapers
    """
    # Patch the engine to use our test database
    import backend.etl.utils.publication_tracker as tracker_module
    import backend.storage.database as db_module

    original_engine = tracker_module.engine
    original_db_engine = getattr(db_module, "engine", None)

    try:
        # Replace engines with test database
        tracker_module.engine = real_test_db
        if original_db_engine:
            db_module.engine = real_test_db

        # Create tracker without database initialization (we already created tables)
        tracker = PublicationTracker(ensure_db=False)
        yield tracker

    finally:
        # Restore original engines
        tracker_module.engine = original_engine
        if original_db_engine:
            db_module.engine = original_db_engine


@pytest.mark.integration
class TestPublicationTrackingIntegration:
    """Integration tests for the complete publication tracking system."""

    @pytest.mark.asyncio
    async def test_growthlab_publication_discovery_and_tracking(
        self, real_publication_tracker: PublicationTracker, real_test_db
    ):
        """
        Test discovering ONE real GrowthLab publication and tracking it
        through the manifest system.

        This test validates:
        - Real publication discovery from GrowthLab (limited to 1 publication)
        - Publication manifest creation (Issue #19)
        - New publication detection (Issue #8)
        - End-to-end tracking functionality
        """
        logger.info(
            "Starting GrowthLab publication discovery and tracking test (1 publication)"
        )

        # Discover only ONE real publication from GrowthLab for faster testing
        try:
            # Limit scraper to get minimal publications
            scraper = real_publication_tracker.growthlab_scraper
            growthlab_publications = await scraper.extract_and_enrich_publications(
                limit=1
            )

            # Ensure we have at least one publication for testing
            assert len(growthlab_publications) > 0, (
                "No GrowthLab publications discovered"
            )

            # Take ONLY the first publication for testing
            test_publication = growthlab_publications[0]
            logger.info(f"Testing with publication: {test_publication.title}")

            # Validate publication data structure
            assert test_publication.title is not None, "Publication should have a title"
            assert test_publication.paper_id is not None, (
                "Publication should have an ID"
            )
            assert test_publication.source == "GrowthLab", (
                "Publication should be from GrowthLab"
            )

            # Test publication manifest creation (Issue #19)
            with Session(real_test_db) as session:
                # Add publication to tracking manifest
                tracking_record = real_publication_tracker.add_publication(
                    test_publication, session=session
                )

                # Verify manifest entry was created correctly
                assert tracking_record.publication_id == test_publication.paper_id
                assert tracking_record.title == test_publication.title
                assert (
                    tracking_record.source_url == str(test_publication.pub_url)
                    if test_publication.pub_url
                    else ""
                )
                assert tracking_record.content_hash == test_publication.content_hash

                # Verify initial status is PENDING for all stages
                assert tracking_record.download_status == DownloadStatus.PENDING
                assert tracking_record.processing_status == ProcessingStatus.PENDING
                assert tracking_record.embedding_status == EmbeddingStatus.PENDING
                assert tracking_record.ingestion_status == IngestionStatus.PENDING

                # Test new publication detection (Issue #8)
                processing_plan = real_publication_tracker.generate_processing_plan(
                    test_publication, session=session
                )

                # For publication (new or existing), should need all stages initially
                assert processing_plan.publication_id == test_publication.paper_id
                assert processing_plan.needs_download is True
                assert processing_plan.needs_processing is True
                assert processing_plan.needs_embedding is True
                assert processing_plan.needs_ingestion is True
                # Reason could be "New publication" or "Status check"
                # depending on if it was already added
                assert processing_plan.reason in ["New publication", "Status check"]

                # Test status updates through pipeline stages
                logger.info("Testing status updates through ETL pipeline")

                # Update download status
                success = real_publication_tracker.update_download_status(
                    test_publication.paper_id,
                    DownloadStatus.DOWNLOADED,
                    session=session,
                )
                assert success, "Download status update should succeed"

                # Verify download status was updated
                updated_record = session.exec(
                    select(PublicationTracking).where(
                        PublicationTracking.publication_id == test_publication.paper_id
                    )
                ).first()
                assert updated_record.download_status == DownloadStatus.DOWNLOADED
                assert updated_record.download_timestamp is not None
                assert updated_record.download_attempt_count == 1

                # Update processing status
                success = real_publication_tracker.update_processing_status(
                    test_publication.paper_id,
                    ProcessingStatus.PROCESSED,
                    session=session,
                )
                assert success, "Processing status update should succeed"

                # Update embedding status
                success = real_publication_tracker.update_embedding_status(
                    test_publication.paper_id, EmbeddingStatus.EMBEDDED, session=session
                )
                assert success, "Embedding status update should succeed"

                # Update ingestion status
                success = real_publication_tracker.update_ingestion_status(
                    test_publication.paper_id, IngestionStatus.INGESTED, session=session
                )
                assert success, "Ingestion status update should succeed"

                # Verify final status
                final_record = session.exec(
                    select(PublicationTracking).where(
                        PublicationTracking.publication_id == test_publication.paper_id
                    )
                ).first()

                assert final_record.download_status == DownloadStatus.DOWNLOADED
                assert final_record.processing_status == ProcessingStatus.PROCESSED
                assert final_record.embedding_status == EmbeddingStatus.EMBEDDED
                assert final_record.ingestion_status == IngestionStatus.INGESTED

                # Test publication status retrieval
                status_dict = real_publication_tracker.get_publication_status(
                    test_publication.paper_id, session=session
                )
                assert status_dict is not None
                assert status_dict["publication_id"] == test_publication.paper_id
                assert status_dict["title"] == test_publication.title
                assert status_dict["download_status"] == DownloadStatus.DOWNLOADED
                assert status_dict["ingestion_status"] == IngestionStatus.INGESTED

                logger.info(
                    "✅ GrowthLab publication tracking test completed successfully"
                )

        except Exception as e:
            logger.error(f"❌ GrowthLab publication test failed: {str(e)}")
            raise

    @pytest.mark.asyncio
    async def test_openalex_publication_discovery_and_tracking(
        self, real_publication_tracker: PublicationTracker, real_test_db
    ):
        """
        Test discovering ONE real OpenAlex publication and tracking it.

        Validates the manifest system works with OpenAlex data sources.
        """
        logger.info(
            "Starting OpenAlex publication discovery and tracking test (1 publication)"
        )

        try:
            # Discover only ONE real publication from OpenAlex for faster testing
            # Fetch only the first page instead of all pages
            async with aiohttp.ClientSession() as session:
                results, _ = await real_publication_tracker.openalex_client.fetch_page(
                    session, cursor="*"
                )
                if not results:
                    pytest.skip("No OpenAlex publications available")
                # Process only the first result
                raw_publications = (
                    real_publication_tracker.openalex_client.process_results(
                        [results[0]]
                    )
                )
                assert len(raw_publications) > 0, "No OpenAlex publications discovered"
                test_publication = raw_publications[0]
            logger.info(f"Testing with OpenAlex publication: {test_publication.title}")

            # Validate OpenAlex publication structure
            assert test_publication.title is not None, "Publication should have a title"
            assert test_publication.paper_id is not None, (
                "Publication should have an ID"
            )
            assert test_publication.source == "OpenAlex", (
                "Publication should be from OpenAlex"
            )
            assert test_publication.openalex_id is not None, "Should have OpenAlex ID"

            with Session(real_test_db) as session:
                # Test adding OpenAlex publication to manifest
                tracking_record = real_publication_tracker.add_publication(
                    test_publication, session=session
                )

                # Verify OpenAlex-specific fields are tracked
                assert tracking_record.publication_id == test_publication.paper_id
                assert tracking_record.title == test_publication.title
                assert tracking_record.content_hash == test_publication.content_hash

                logger.info(
                    "✅ OpenAlex publication tracking test completed successfully"
                )

        except Exception as e:
            logger.error(f"❌ OpenAlex publication test failed: {str(e)}")
            raise

    @pytest.mark.asyncio
    async def test_publication_update_detection(
        self, real_publication_tracker: PublicationTracker, real_test_db
    ):
        """
        Test the system's ability to detect publication updates (Issue #8).

        Uses ONE real publication, simulates an update, and verifies the system
        correctly identifies what needs to be reprocessed.
        """
        logger.info("Starting publication update detection test (1 publication)")

        try:
            # Get ONE real publication for faster testing
            scraper = real_publication_tracker.growthlab_scraper
            growthlab_publications = await scraper.extract_and_enrich_publications(
                limit=1
            )
            assert len(growthlab_publications) > 0, (
                "Need at least one publication for update test"
            )

            # Use ONLY the first publication
            original_publication = growthlab_publications[0]

            with Session(real_test_db) as session:
                # Add original publication
                original_tracking = real_publication_tracker.add_publication(
                    original_publication, session=session
                )

                # Mark it as fully processed
                real_publication_tracker.update_download_status(
                    original_publication.paper_id,
                    DownloadStatus.DOWNLOADED,
                    session=session,
                )
                real_publication_tracker.update_processing_status(
                    original_publication.paper_id,
                    ProcessingStatus.PROCESSED,
                    session=session,
                )
                real_publication_tracker.update_embedding_status(
                    original_publication.paper_id,
                    EmbeddingStatus.EMBEDDED,
                    session=session,
                )
                real_publication_tracker.update_ingestion_status(
                    original_publication.paper_id,
                    IngestionStatus.INGESTED,
                    session=session,
                )

                # Simulate content change by modifying the content hash
                updated_title = (
                    original_publication.title or "Untitled"
                ) + " [UPDATED]"
                updated_publication = GrowthLabPublication(
                    paper_id=original_publication.paper_id,
                    title=updated_title,  # Simulate title change
                    authors=original_publication.authors,
                    year=original_publication.year,
                    abstract=original_publication.abstract,
                    pub_url=original_publication.pub_url,
                    file_urls=original_publication.file_urls,
                    source=original_publication.source,
                )
                # Generate new content hash for updated publication
                updated_publication.content_hash = (
                    updated_publication.generate_content_hash()
                )

                # Test update detection
                processing_plan = real_publication_tracker.generate_processing_plan(
                    updated_publication, session=session
                )

                # Should detect content change and require full reprocessing
                assert processing_plan.publication_id == original_publication.paper_id
                assert processing_plan.needs_download is True, (
                    "Should need download due to content change"
                )
                assert processing_plan.needs_processing is True, (
                    "Should need processing due to content change"
                )
                assert processing_plan.needs_embedding is True, (
                    "Should need embedding due to content change"
                )
                assert processing_plan.needs_ingestion is True, (
                    "Should need ingestion due to content change"
                )
                assert processing_plan.reason == "Content hash changed", (
                    "Should detect content hash change"
                )

                # Apply the update
                updated_tracking = real_publication_tracker.add_publication(
                    updated_publication, session=session
                )

                # Verify the tracking record was updated correctly
                assert updated_tracking.title == updated_publication.title
                assert updated_tracking.content_hash == updated_publication.content_hash
                # Status should be reset to PENDING due to content change
                assert updated_tracking.download_status == DownloadStatus.PENDING
                assert updated_tracking.processing_status == ProcessingStatus.PENDING

                logger.info(
                    "✅ Publication update detection test completed successfully"
                )

        except Exception as e:
            logger.error(f"❌ Publication update detection test failed: {str(e)}")
            raise

    @pytest.mark.asyncio
    async def test_complete_etl_pipeline_simulation(
        self, real_publication_tracker: PublicationTracker, real_test_db
    ):
        """
        Test a complete ETL pipeline simulation with ONE real publication.

        This test simulates the full workflow with minimal data:
        1. Discovery of ONE publication
        2. Manifest creation and tracking
        3. Status updates through all pipeline stages
        4. Query publications at each stage
        """
        logger.info("Starting complete ETL pipeline simulation test (1 publication)")

        try:
            # Discover publications from GrowthLab source only (faster)
            scraper = real_publication_tracker.growthlab_scraper
            growthlab_publications = await scraper.extract_and_enrich_publications(
                limit=1
            )
            assert len(growthlab_publications) > 0, (
                "Should discover at least one publication"
            )

            # Use only ONE publication for faster testing
            test_publication = growthlab_publications[0]
            discovered_publications = [(test_publication, "growthlab")]

            logger.info(f"Testing complete pipeline with: {test_publication.title}")

            with Session(real_test_db) as session:
                # Add the single publication to tracking manifest
                tracking_records = []
                for pub, source in discovered_publications:  # Only 1 publication
                    tracking = real_publication_tracker.add_publication(
                        pub, session=session
                    )
                    tracking_records.append(tracking)
                    logger.info(f"Added {pub.title} from {source} to manifest")

                # Test querying publications at each stage
                # Initially, all should be pending download
                download_queue = real_publication_tracker.get_publications_for_download(
                    session=session
                )
                assert len(download_queue) == len(tracking_records), (
                    "All publications should need download"
                )

                # No publications should be ready for other stages yet
                processing_queue = (
                    real_publication_tracker.get_publications_for_processing(
                        session=session
                    )
                )
                assert len(processing_queue) == 0, (
                    "No publications ready for processing yet"
                )

                # Simulate processing first publication through download stage
                first_pub = tracking_records[0]
                success = real_publication_tracker.update_download_status(
                    first_pub.publication_id, DownloadStatus.DOWNLOADED, session=session
                )
                assert success, "Download status update should succeed"

                # Now one publication should be ready for processing
                processing_queue = (
                    real_publication_tracker.get_publications_for_processing(
                        session=session
                    )
                )
                assert len(processing_queue) == 1, (
                    "One publication should be ready for processing"
                )
                assert processing_queue[0].publication_id == first_pub.publication_id

                # Continue processing through all stages
                real_publication_tracker.update_processing_status(
                    first_pub.publication_id,
                    ProcessingStatus.PROCESSED,
                    session=session,
                )

                embedding_queue = (
                    real_publication_tracker.get_publications_for_embedding(
                        session=session
                    )
                )
                assert len(embedding_queue) == 1, (
                    "One publication should be ready for embedding"
                )

                real_publication_tracker.update_embedding_status(
                    first_pub.publication_id, EmbeddingStatus.EMBEDDED, session=session
                )

                ingestion_queue = (
                    real_publication_tracker.get_publications_for_ingestion(
                        session=session
                    )
                )
                assert len(ingestion_queue) == 1, (
                    "One publication should be ready for ingestion"
                )

                real_publication_tracker.update_ingestion_status(
                    first_pub.publication_id, IngestionStatus.INGESTED, session=session
                )

                # Verify the publication is fully processed
                final_status = real_publication_tracker.get_publication_status(
                    first_pub.publication_id, session=session
                )
                assert final_status["ingestion_status"] == IngestionStatus.INGESTED

                # Verify queues are updated correctly
                ingestion_queue = (
                    real_publication_tracker.get_publications_for_ingestion(
                        session=session
                    )
                )
                assert len(ingestion_queue) == 0, (
                    "No publications should need ingestion now"
                )

                logger.info(
                    "✅ Complete ETL pipeline simulation test completed successfully"
                )

        except Exception as e:
            logger.error(f"❌ Complete ETL pipeline simulation test failed: {str(e)}")
            raise

    def test_publication_deduplication(
        self, real_publication_tracker: PublicationTracker, real_test_db
    ):
        """
        Test that the manifest system properly handles duplicate publications
        (Issue #19).

        Verifies that adding the same publication twice doesn't create duplicates
        but updates the existing record appropriately.
        """
        logger.info("Starting publication deduplication test")

        try:
            # Create a test publication
            test_publication = GrowthLabPublication(
                paper_id="test_dedup_123",
                title="Test Deduplication Paper",
                authors="Test Author",
                year=2023,
                abstract="Test abstract for deduplication",
                source="GrowthLab",
            )
            test_publication.content_hash = test_publication.generate_content_hash()

            with Session(real_test_db) as session:
                # Add publication first time
                first_tracking = real_publication_tracker.add_publication(
                    test_publication, session=session
                )

                # Count records
                count_query = select(PublicationTracking).where(
                    PublicationTracking.publication_id == test_publication.paper_id
                )
                records = list(session.exec(count_query))
                assert len(records) == 1, (
                    "Should have exactly one record after first add"
                )

                # Add same publication again
                second_tracking = real_publication_tracker.add_publication(
                    test_publication, session=session
                )

                # Should still have only one record
                records = list(session.exec(count_query))
                assert len(records) == 1, (
                    "Should still have exactly one record after duplicate add"
                )

                # Records should be equivalent
                assert first_tracking.publication_id == second_tracking.publication_id
                assert first_tracking.content_hash == second_tracking.content_hash

                logger.info("✅ Publication deduplication test completed successfully")

        except Exception as e:
            logger.error(f"❌ Publication deduplication test failed: {str(e)}")
            raise


@pytest.mark.integration
class TestPublicationTrackingRobustness:
    """Test the robustness and error handling of the publication tracking system."""

    def test_invalid_publication_handling(
        self, real_publication_tracker: PublicationTracker, real_test_db
    ):
        """Test handling of invalid or malformed publication data."""
        logger.info("Starting invalid publication handling test")

        try:
            with Session(real_test_db) as session:
                # Test with missing required fields
                invalid_publication = GrowthLabPublication(
                    title="Test Publication",
                    # Missing other required fields
                )

                # Should handle gracefully
                try:
                    tracking = real_publication_tracker.add_publication(
                        invalid_publication, session=session
                    )
                    # If it succeeds, verify it handles missing data appropriately
                    assert tracking is not None
                except (ValueError, AttributeError) as e:
                    # Expected for invalid data
                    logger.info(f"Correctly handled invalid data: {e}")

                logger.info("✅ Invalid publication handling test completed")

        except Exception as e:
            logger.error(f"❌ Invalid publication handling test failed: {str(e)}")
            raise

    def test_database_error_handling(
        self, real_publication_tracker: PublicationTracker
    ):
        """Test handling of database connection errors."""
        logger.info("Starting database error handling test")

        try:
            # Test with non-existent publication ID
            success = real_publication_tracker.update_download_status(
                "non_existent_id", DownloadStatus.DOWNLOADED
            )
            assert success is False, "Should return False for non-existent publication"

            # Test getting status for non-existent publication
            status = real_publication_tracker.get_publication_status("non_existent_id")
            assert status is None, "Should return None for non-existent publication"

            logger.info("✅ Database error handling test completed")

        except Exception as e:
            logger.error(f"❌ Database error handling test failed: {str(e)}")
            raise


"""
Unit Tests for the publication tracking functionality.

This module tests the PublicationTracker class and related components that are
responsible for tracking publications through the ETL pipeline stages.
"""

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
    Create a PublicationTracker instance for testing with properly configured
    async mocks.

    Args:
        in_memory_db: In-memory database engine

    Returns:
        PublicationTracker instance with mocked scrapers for testing
    """
    with patch("backend.etl.utils.publication_tracker.engine", in_memory_db):
        with patch("backend.etl.utils.publication_tracker.ensure_db_initialized"):
            tracker = PublicationTracker(ensure_db=False)

            # Create mock scrapers - the key is to patch the methods directly
            # rather than trying to assign AsyncMock objects later
            tracker.growthlab_scraper = MagicMock()
            tracker.openalex_client = MagicMock()

            # Pre-configure the async methods as AsyncMock objects
            # This prevents the "Callable has no attribute" errors
            tracker.growthlab_scraper.extract_and_enrich_publications = AsyncMock()
            tracker.openalex_client.fetch_publications = AsyncMock()

            return tracker


@pytest.fixture
def sample_growthlab_publication() -> GrowthLabPublication:
    """
    Create a sample GrowthLab publication for testing.

    Returns:
        GrowthLabPublication instance with realistic test data
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
        OpenAlexPublication instance with realistic test data
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
        # Cast to AsyncMock to help MyPy understand these are AsyncMock objects
        growthlab_mock = cast(
            AsyncMock,
            mock_publication_tracker.growthlab_scraper.extract_and_enrich_publications,
        )
        openalex_mock = cast(
            AsyncMock, mock_publication_tracker.openalex_client.fetch_publications
        )

        # Configure mock return values - MyPy-safe approach
        growthlab_mock.return_value = [sample_growthlab_publication]
        openalex_mock.return_value = [sample_openalex_publication]

        # Call the method under test
        publications = await mock_publication_tracker.discover_publications()

        # Verify results structure and content
        assert len(publications) == 2
        assert publications[0][0] == sample_growthlab_publication
        assert publications[0][1] == "growthlab"
        assert publications[1][0] == sample_openalex_publication
        assert publications[1][1] == "openalex"

        # Verify mock methods were called exactly once
        growthlab_mock.assert_called_once()
        openalex_mock.assert_called_once()

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
        # Cast to AsyncMock to help MyPy understand this is an AsyncMock object
        growthlab_mock = cast(
            AsyncMock,
            mock_publication_tracker.growthlab_scraper.extract_and_enrich_publications,
        )

        # Configure mock to raise exception - MyPy-safe approach
        growthlab_mock.side_effect = Exception("Test error")

        # Verify the method raises the exception
        with pytest.raises(Exception, match="Test error"):
            await mock_publication_tracker.discover_publications()

        # Verify error logging
        assert "Error discovering publications: Test error" in caplog.text

    @pytest.mark.asyncio
    async def test_discover_publications_empty_results(
        self, mock_publication_tracker: PublicationTracker
    ) -> None:
        """
        Test discover_publications with empty results from both sources.
        """
        # Cast to AsyncMock to help MyPy understand these are AsyncMock objects
        growthlab_mock = cast(
            AsyncMock,
            mock_publication_tracker.growthlab_scraper.extract_and_enrich_publications,
        )
        openalex_mock = cast(
            AsyncMock, mock_publication_tracker.openalex_client.fetch_publications
        )

        # Configure mocks to return empty lists
        growthlab_mock.return_value = []
        openalex_mock.return_value = []

        # Call the method
        publications = await mock_publication_tracker.discover_publications()

        # Verify empty results
        assert len(publications) == 0
        assert publications == []

    @pytest.mark.asyncio
    async def test_discover_publications_partial_failure(
        self,
        mock_publication_tracker: PublicationTracker,
        sample_openalex_publication: OpenAlexPublication,
        caplog: "LogCaptureFixture",
    ) -> None:
        """
        Test discover_publications when one source fails but the other succeeds.
        """
        # Cast to AsyncMock to help MyPy understand these are AsyncMock objects
        growthlab_mock = cast(
            AsyncMock,
            mock_publication_tracker.growthlab_scraper.extract_and_enrich_publications,
        )
        openalex_mock = cast(
            AsyncMock, mock_publication_tracker.openalex_client.fetch_publications
        )

        # Configure one source to fail, another to succeed
        growthlab_mock.side_effect = Exception("GrowthLab error")
        openalex_mock.return_value = [sample_openalex_publication]

        # The method should still raise the exception (based on current implementation)
        with pytest.raises(Exception, match="GrowthLab error"):
            await mock_publication_tracker.discover_publications()

    def test_generate_processing_plan_new_publication(
        self,
        mock_publication_tracker: PublicationTracker,
        sample_growthlab_publication: GrowthLabPublication,
        in_memory_db: Any,
    ) -> None:
        """
        Test generating a processing plan for a new publication.
        """
        with Session(in_memory_db) as session:
            plan = mock_publication_tracker.generate_processing_plan(
                sample_growthlab_publication, session=session
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
            tracking_result = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_growthlab_publication.paper_id
                )
            ).first()
            assert tracking_result is not None
            tracking = tracking_result

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
            tracking_result = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_growthlab_publication.paper_id
                )
            ).first()
            assert tracking_result is not None
            tracking = tracking_result

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
            tracking_result = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_growthlab_publication.paper_id
                )
            ).first()
            assert tracking_result is not None
            tracking = tracking_result

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
            tracking_result = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_growthlab_publication.paper_id
                )
            ).first()
            assert tracking_result is not None
            tracking = tracking_result

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
            tracking_result = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_growthlab_publication.paper_id
                )
            ).first()
            assert tracking_result is not None
            tracking = tracking_result

            # Update publication
            updated = mock_publication_tracker.add_publication(
                sample_growthlab_publication, session=session
            )
            session.commit()
            updated_result = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_growthlab_publication.paper_id
                )
            ).first()
            assert updated_result is not None
            updated = updated_result

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
            tracking_result = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_growthlab_publication.paper_id
                )
            ).first()
            assert tracking_result is not None
            tracking = tracking_result

            # Update status
            success = mock_publication_tracker.update_download_status(
                sample_growthlab_publication.paper_id,
                DownloadStatus.DOWNLOADED,
                session=session,
            )
            assert success

            # Verify update
            with Session(in_memory_db) as verify_session:
                stmt: Any = select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_growthlab_publication.paper_id
                )
                updated_result = verify_session.exec(stmt).first()
                assert updated_result is not None
                updated = updated_result
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
            sample_tracking_result = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_tracking_publication.publication_id
                )
            ).first()
            assert sample_tracking_result is not None
            sample_tracking_publication = sample_tracking_result

            # Update status
            success = mock_publication_tracker.update_processing_status(
                sample_tracking_publication.publication_id,
                ProcessingStatus.PROCESSED,
                session=session,
            )
            assert success

            # Verify update
            with Session(in_memory_db) as verify_session:
                stmt: Any = select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_tracking_publication.publication_id
                )
                updated_result = verify_session.exec(stmt).first()
                assert updated_result is not None
                updated = updated_result
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
            sample_tracking_result = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_tracking_publication.publication_id
                )
            ).first()
            assert sample_tracking_result is not None
            sample_tracking_publication = sample_tracking_result

            # Update status
            success = mock_publication_tracker.update_embedding_status(
                sample_tracking_publication.publication_id,
                EmbeddingStatus.EMBEDDED,
                session=session,
            )
            assert success

            # Verify update
            with Session(in_memory_db) as verify_session:
                stmt: Any = select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_tracking_publication.publication_id
                )
                updated_result = verify_session.exec(stmt).first()
                assert updated_result is not None
                updated = updated_result
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
            sample_tracking_result = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_tracking_publication.publication_id
                )
            ).first()
            assert sample_tracking_result is not None
            sample_tracking_publication = sample_tracking_result

            # Update status
            success = mock_publication_tracker.update_ingestion_status(
                sample_tracking_publication.publication_id,
                IngestionStatus.INGESTED,
                session=session,
            )
            assert success

            # Verify update
            with Session(in_memory_db) as verify_session:
                stmt: Any = select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_tracking_publication.publication_id
                )
                updated_result = verify_session.exec(stmt).first()
                assert updated_result is not None
                updated = updated_result
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
            pub1_result = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id == "pub1"
                )
            ).first()
            pub2_result = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id == "pub2"
                )
            ).first()
            assert pub1_result is not None
            assert pub2_result is not None
            pub1 = pub1_result
            pub2 = pub2_result

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
            pub1_result = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id == "pub1"
                )
            ).first()
            pub2_result = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id == "pub2"
                )
            ).first()
            assert pub1_result is not None
            assert pub2_result is not None
            pub1 = pub1_result
            pub2 = pub2_result

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
            pub1_result = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id == "pub1"
                )
            ).first()
            pub2_result = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id == "pub2"
                )
            ).first()
            assert pub1_result is not None
            assert pub2_result is not None
            pub1 = pub1_result
            pub2 = pub2_result

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
            pub1_result = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id == "pub1"
                )
            ).first()
            pub2_result = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id == "pub2"
                )
            ).first()
            assert pub1_result is not None
            assert pub2_result is not None
            pub1 = pub1_result
            pub2 = pub2_result

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
            sample_tracking_result = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id
                    == sample_tracking_publication.publication_id
                )
            ).first()
            assert sample_tracking_result is not None
            sample_tracking_publication = sample_tracking_result

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
