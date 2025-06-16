"""
Integration tests for the PDF processor functionality.
"""

import asyncio
import shutil
import tempfile
from pathlib import Path

import pytest
from loguru import logger

from backend.etl.utils.pdf_processor import PDFProcessor
from backend.storage.local import LocalStorage


class TestPDFProcessor:
    """Test suite for PDF processor functionality."""

    @pytest.fixture
    def sample_pdfs(self):
        """Get paths to sample PDF files for testing."""
        # Get the fixtures directory
        fixtures_dir = Path(__file__).parent / "fixtures" / "pdfs"

        # Ensure the fixtures exist
        assert fixtures_dir.exists(), f"Fixtures directory {fixtures_dir} not found"

        # Get the PDF paths
        sample1_path = fixtures_dir / "sample1.pdf"
        sample2_path = fixtures_dir / "sample2.pdf"

        assert sample1_path.exists(), f"Sample PDF {sample1_path} not found"
        assert sample2_path.exists(), f"Sample PDF {sample2_path} not found"

        return [sample1_path, sample2_path]

    @pytest.fixture
    def test_storage(self):
        """Create a temporary directory for test storage."""
        # Create a temporary directory
        temp_dir = Path(tempfile.mkdtemp())

        # Create raw and processed directories
        raw_dir = temp_dir / "raw" / "documents" / "growthlab" / "test_pub"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Create storage instance
        storage = LocalStorage(base_path=temp_dir)

        # Return storage and its temp directory
        yield storage

        # Clean up
        shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_process_single_pdf(self, sample_pdfs, test_storage):
        """Test processing a single PDF file."""
        # Copy the sample PDF to the raw directory
        pdf_path = sample_pdfs[0]
        dest_path = test_storage.get_path("raw/documents/growthlab/test_pub/sample.pdf")
        shutil.copy(pdf_path, dest_path)

        # Create processor instance
        processor = PDFProcessor(
            storage=test_storage,
            concurrency_limit=1,
        )

        # Process the PDF
        result_path = await processor.process_pdf(dest_path)

        # Check that processing was successful
        assert result_path is not None, "PDF processing failed"
        assert result_path.exists(), f"Output file {result_path} not found"

        # Check that the output contains text
        text_content = result_path.read_text()
        assert len(text_content) > 0, "Output file is empty"

        # Basic validation that it contains some expected PDF content
        assert "Page" in text_content, "Output doesn't contain page markers"

        logger.info(f"Processed PDF {dest_path} to {result_path}")
        logger.info(f"Output length: {len(text_content)} characters")

    @pytest.mark.asyncio
    async def test_process_multiple_pdfs(self, sample_pdfs, test_storage):
        """Test processing multiple PDF files asynchronously."""
        # Copy the sample PDFs to the raw directory
        dest_paths = []
        for i, pdf_path in enumerate(sample_pdfs):
            dest_path = test_storage.get_path(
                f"raw/documents/growthlab/test_pub/sample{i + 1}.pdf"
            )
            shutil.copy(pdf_path, dest_path)
            dest_paths.append(dest_path)

        # Create processor instance
        processor = PDFProcessor(
            storage=test_storage,
            concurrency_limit=2,  # Allow concurrent processing
            config_path=None,
        )

        # Process the PDFs
        results = await processor.process_pdfs(
            dest_paths,
            force_reprocess=False,
            show_progress=True,
        )

        # Check that all PDFs were processed successfully
        assert len(results) == len(dest_paths), "Not all PDFs were processed"
        assert all(result is not None for result in results.values()), (
            "Some PDFs failed processing"
        )

        # Check that all output files exist and contain text
        for pdf_path, result_path in results.items():
            assert result_path.exists(), f"Output file {result_path} not found"

            text_content = result_path.read_text()
            assert len(text_content) > 0, f"Output file for {pdf_path} is empty"

            # Basic validation
            assert "Page" in text_content, (
                f"Output for {pdf_path} doesn't contain page markers"
            )

            logger.info(f"Processed PDF {pdf_path} to {result_path}")
            logger.info(f"Output length: {len(text_content)} characters")

    @pytest.mark.asyncio
    async def test_async_concurrency(self, sample_pdfs, test_storage):
        """Test that PDFs are processed concurrently."""
        # Skip this test if only one sample PDF is available
        if len(sample_pdfs) < 2:
            pytest.skip("Need at least 2 sample PDFs for concurrency testing")

        # Copy the sample PDFs to the raw directory multiple times to get more tests
        dest_paths = []
        for i in range(4):  # Create 4 copies (2 from each sample)
            sample_idx = i % len(sample_pdfs)
            dest_path = test_storage.get_path(
                f"raw/documents/growthlab/test_pub/sample{i + 1}.pdf"
            )
            shutil.copy(sample_pdfs[sample_idx], dest_path)
            dest_paths.append(dest_path)

        # Create processor instance with high concurrency
        processor = PDFProcessor(
            storage=test_storage,
            concurrency_limit=4,  # Allow high concurrency
        )

        # Add a timestamp to track start time
        start_time = asyncio.get_event_loop().time()

        # Process the PDFs
        results = await processor.process_pdfs(
            dest_paths,
            force_reprocess=False,
            show_progress=True,
        )

        # Calculate duration
        duration = asyncio.get_event_loop().time() - start_time

        # Check results
        success_count = sum(1 for result in results.values() if result is not None)

        logger.info(
            f"Processed {success_count}/{len(dest_paths)} PDFs "
            f"in {duration:.2f} seconds"
        )
        assert success_count > 0, "No PDFs were processed successfully"

        # Note: We can't assert specific timing as it depends on the test environment,
        # but we log it for manual verification
