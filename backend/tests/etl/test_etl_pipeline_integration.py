"""
End-to-end integration tests for the complete ETL pipeline with publication tracking.

These tests verify that the entire pipeline works correctly with publication tracker
integration, ensuring status transitions flow properly through all stages.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from backend.etl.models.publications import GrowthLabPublication
from backend.etl.models.tracking import (
    DownloadStatus,
    EmbeddingStatus,
    ProcessingStatus,
    PublicationTracking,
)
from backend.etl.utils.embeddings_generator import EmbeddingsGenerator
from backend.etl.utils.pdf_processor import PDFProcessor
from backend.etl.utils.publication_tracker import PublicationTracker
from backend.etl.utils.text_chunker import TextChunker
from backend.storage.local import LocalStorage


@pytest.fixture
def test_db():
    """Create a temporary database for testing."""
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    import os

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
def test_tracker(test_db):
    """Create a PublicationTracker with test database."""
    import backend.etl.utils.publication_tracker as tracker_module
    import backend.storage.database as db_module

    original_engine = tracker_module.engine
    original_db_engine = getattr(db_module, "engine", None)

    try:
        tracker_module.engine = test_db
        if original_db_engine:
            db_module.engine = test_db

        tracker = PublicationTracker(ensure_db=False)
        yield tracker

    finally:
        tracker_module.engine = original_engine
        if original_db_engine:
            db_module.engine = original_db_engine


@pytest.fixture
def test_storage():
    """Create temporary directory for test storage."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create directory structure
    (temp_dir / "raw" / "documents" / "growthlab" / "test_pub_123").mkdir(
        parents=True, exist_ok=True
    )
    (temp_dir / "processed" / "documents" / "growthlab" / "test_pub_123").mkdir(
        parents=True, exist_ok=True
    )
    (
        temp_dir / "processed" / "chunks" / "documents" / "growthlab" / "test_pub_123"
    ).mkdir(parents=True, exist_ok=True)
    (
        temp_dir
        / "processed"
        / "embeddings"
        / "documents"
        / "growthlab"
        / "test_pub_123"
    ).mkdir(parents=True, exist_ok=True)

    storage = LocalStorage(base_path=temp_dir)

    yield temp_dir, storage

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_pdf():
    """Get path to sample PDF file."""
    fixtures_dir = Path(__file__).parent / "fixtures" / "pdfs"
    pdf_path = fixtures_dir / "sample1.pdf"
    assert pdf_path.exists(), f"Sample PDF {pdf_path} not found"
    return pdf_path


@pytest.fixture
def test_publication():
    """Create a test publication."""
    pub = GrowthLabPublication(
        paper_id="test_pub_123",
        title="Test Publication",
        authors="Test Author",
        year=2023,
        abstract="Test abstract",
        source="GrowthLab",
    )
    pub.content_hash = pub.generate_content_hash()
    return pub


class TestETLPipelineIntegration:
    """End-to-end integration tests for complete ETL pipeline."""

    def test_complete_pipeline_flow_with_real_files(
        self, test_storage, test_tracker, test_db, sample_pdf, test_publication
    ):
        """
        Test complete pipeline flow with real files.

        Verifies:
        1. Publication registered in tracker
        2. PDF processing updates status
        3. Text chunking updates status
        4. Embeddings generator finds and processes publication
        5. Status transitions correctly through all stages
        """
        temp_dir, storage = test_storage

        # Step 1: Register publication in tracker
        with Session(test_db) as session:
            tracking_record = test_tracker.add_publication(
                test_publication, session=session
            )
            assert tracking_record.publication_id == test_publication.paper_id
            assert tracking_record.processing_status == ProcessingStatus.PENDING

        # Step 2: Copy PDF to raw directory
        pdf_path = storage.get_path("raw/documents/growthlab/test_pub_123/sample.pdf")
        shutil.copy(sample_pdf, pdf_path)

        # Step 3: Process PDF
        processor = PDFProcessor(storage=storage, tracker=test_tracker)
        processed_path = processor.process_pdf(pdf_path)

        assert processed_path is not None
        assert processed_path.exists()

        # Verify processing status updated
        with Session(test_db) as session:
            updated = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id == test_publication.paper_id
                )
            ).first()
            assert updated is not None
            assert updated.processing_status == ProcessingStatus.PROCESSED
            assert updated.processing_timestamp is not None

        # Step 4: Chunk the processed text
        chunker = TextChunker(
            config_path=Path(__file__).parent.parent.parent / "etl" / "config.yaml",
            tracker=test_tracker,
        )
        chunking_results = chunker.process_all_documents(storage=storage)

        assert len(chunking_results) > 0
        assert any(r.status.value == "success" for r in chunking_results)

        # Verify processing status still PROCESSED (chunker should maintain it)
        with Session(test_db) as session:
            updated = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id == test_publication.paper_id
                )
            ).first()
            assert updated is not None
            assert updated.processing_status == ProcessingStatus.PROCESSED

        # Step 5: Generate embeddings
        # First, mark download as complete (embeddings generator expects this)
        with Session(test_db) as session:
            test_tracker.update_download_status(
                test_publication.paper_id,
                DownloadStatus.DOWNLOADED,
                session=session,
            )

        generator = EmbeddingsGenerator(
            config_path=Path(__file__).parent.parent.parent / "etl" / "config.yaml"
        )

        # Mock OpenAI API to avoid actual API calls
        from unittest.mock import AsyncMock

        with patch("backend.etl.utils.embeddings_generator.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            # Create mock embedding response
            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1] * 1536  # 1536 dimensions

            mock_response = MagicMock()
            mock_response.data = [mock_embedding]

            # Use AsyncMock for async method
            async_create = AsyncMock(return_value=mock_response)
            mock_client.embeddings.create = async_create

            # Generate embeddings
            import asyncio

            embedding_results = asyncio.run(
                generator.process_all_documents(storage=storage, tracker=test_tracker)
            )

            # Verify embeddings were generated
            assert len(embedding_results) > 0

        # Verify embedding status updated
        with Session(test_db) as session:
            updated = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id == test_publication.paper_id
                )
            ).first()
            assert updated is not None
            assert updated.embedding_status == EmbeddingStatus.EMBEDDED
            assert updated.embedding_timestamp is not None

    def test_embeddings_query_dependency(self, test_tracker, test_db, test_publication):
        """
        Test that embeddings generator only gets publications with PROCESSED status.

        This is the critical test that would have caught the original bug.
        """
        with Session(test_db) as session:
            # Add publication
            test_tracker.add_publication(test_publication, session=session)

            # Mark as downloaded but NOT processed
            test_tracker.update_download_status(
                test_publication.paper_id, DownloadStatus.DOWNLOADED, session=session
            )
            # Leave processing_status as PENDING

            # Query for publications ready for embedding
            publications = test_tracker.get_publications_for_embedding(session=session)

            # Should return ZERO publications because processing_status != PROCESSED
            assert len(publications) == 0, (
                "Should not return publications with PENDING processing status"
            )

            # Now mark as PROCESSED
            test_tracker.update_processing_status(
                test_publication.paper_id,
                ProcessingStatus.PROCESSED,
                session=session,
            )

            # Query again
            publications = test_tracker.get_publications_for_embedding(session=session)

            # Should now return the publication
            assert len(publications) == 1
            assert publications[0].publication_id == test_publication.paper_id

    def test_component_with_tracker(
        self, test_storage, test_tracker, test_db, sample_pdf
    ):
        """
        Test that components work correctly with tracker when provided.

        Verifies:
        - Components update tracker when tracker is provided
        - No errors when tracker is provided
        """
        temp_dir, storage = test_storage

        # Create test publication
        pub = GrowthLabPublication(
            paper_id="test_pub_123",
            title="Test Publication",
            authors="Test Author",
            year=2023,
            abstract="Test abstract",
            source="GrowthLab",
        )
        pub.content_hash = pub.generate_content_hash()

        with Session(test_db) as session:
            test_tracker.add_publication(pub, session=session)

        # Test PDF processor with tracker
        pdf_path = storage.get_path("raw/documents/growthlab/test_pub_123/sample.pdf")
        shutil.copy(sample_pdf, pdf_path)

        processor = PDFProcessor(storage=storage, tracker=test_tracker)
        processed_path = processor.process_pdf(pdf_path)

        assert processed_path is not None

        # Verify tracker was updated
        with Session(test_db) as session:
            updated = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id == "test_pub_123"
                )
            ).first()
            assert updated is not None
            assert updated.processing_status == ProcessingStatus.PROCESSED

        # Test text chunker with tracker
        chunker = TextChunker(
            config_path=Path(__file__).parent.parent.parent / "etl" / "config.yaml",
            tracker=test_tracker,
        )
        chunking_results = chunker.process_all_documents(storage=storage)

        assert len(chunking_results) > 0

        # Verify tracker still shows PROCESSED
        with Session(test_db) as session:
            updated = session.exec(
                select(PublicationTracking).where(
                    PublicationTracking.publication_id == "test_pub_123"
                )
            ).first()
            assert updated is not None
            assert updated.processing_status == ProcessingStatus.PROCESSED
