"""
Integration tests for publication tracking system using real publications.

This module tests the complete publication manifest and tracking system
using actual publications from real sources to validate end-to-end functionality.
It tests the core components from Issues #19 (Publication Manifest System)
and #8 (Check for new publications).

These tests require network access and may be slower than unit tests.
"""

import logging
import os
import tempfile

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from backend.etl.models.publications import GrowthLabPublication
from backend.etl.models.tracking import (
    DownloadStatus,
    EmbeddingStatus,
    IngestionStatus,
    ProcessingStatus,
    PublicationTracking,
)
from backend.etl.utils.publication_tracker import PublicationTracker

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
            growthlab_publications = await scraper.extract_and_enrich_publications()

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
            openalex_publications = (
                await real_publication_tracker.openalex_client.fetch_publications()
            )

            # Ensure we have at least one publication
            assert len(openalex_publications) > 0, "No OpenAlex publications discovered"

            # Take ONLY the first publication for testing
            test_publication = openalex_publications[0]
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
            growthlab_publications = await scraper.extract_and_enrich_publications()
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
            growthlab_publications = await scraper.extract_and_enrich_publications()
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
