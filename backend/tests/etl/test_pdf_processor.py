"""
Tests for the PDF processor functionality.

The integration tests use the dev config (unstructured backend, fast strategy)
so they run without heavy model downloads.

The unit tests exercise the storage-abstraction logic (skip checks,
page cap, upload calls) without touching any real backend.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from loguru import logger

from backend.etl.utils.pdf_processor import PDFProcessor, find_growth_lab_pdfs
from backend.storage.local import LocalStorage

# ---------------------------------------------------------------------------
# Unit tests — fast, no model downloads
# ---------------------------------------------------------------------------


class TestPDFProcessorUnit:
    """Unit tests that mock the PDF extraction backend."""

    @pytest.fixture
    def test_storage(self, tmp_path):
        """Create a LocalStorage rooted at a temp directory."""
        storage = LocalStorage(base_path=tmp_path)
        return storage

    @pytest.fixture
    def processor_with_mock_backend(self, test_storage):
        """Create a PDFProcessor with a mocked extraction backend."""
        processor = PDFProcessor.__new__(PDFProcessor)
        processor.storage = test_storage
        processor.tracker = None
        processor.config = {"min_chars_per_page": 10, "max_pages": 0}
        processor.min_chars_per_page = 10
        processor.max_pages = 0
        processor.processed_root = "processed/documents/growthlab"
        processor._full_config = {
            "ocr": processor.config,
            "processed_root": processor.processed_root,
        }

        # Mock the backend so no model weights are loaded
        mock_backend = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.text = "Extracted text content for testing. " * 20
        mock_result.error = None
        mock_backend.extract.return_value = mock_result
        processor._backend = mock_backend

        return processor

    def test_skip_already_processed(self, test_storage, processor_with_mock_backend):
        """Verify that process_pdf skips when output already exists in storage."""
        processor = processor_with_mock_backend

        # Create a fake PDF in the raw dir
        pdf_dir = test_storage.get_path("raw/documents/growthlab/pub123")
        pdf_dir.mkdir(parents=True)
        pdf_path = pdf_dir / "paper.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")

        # Pre-create the processed output
        out_dir = test_storage.get_path("processed/documents/growthlab/pub123")
        out_dir.mkdir(parents=True)
        out_file = out_dir / "paper.txt"
        out_file.write_text("already done")

        result = processor.process_pdf(pdf_path, force_reprocess=False)

        # Should return the existing output without calling the backend
        assert result is not None
        processor._backend.extract.assert_not_called()

    def test_force_reprocess_ignores_existing(
        self, test_storage, processor_with_mock_backend
    ):
        """Verify that force_reprocess=True re-extracts even if output exists."""
        processor = processor_with_mock_backend

        pdf_dir = test_storage.get_path("raw/documents/growthlab/pub456")
        pdf_dir.mkdir(parents=True)
        pdf_path = pdf_dir / "paper.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")

        # Pre-create the processed output
        out_dir = test_storage.get_path("processed/documents/growthlab/pub456")
        out_dir.mkdir(parents=True)
        (out_dir / "paper.txt").write_text("old text")

        result = processor.process_pdf(pdf_path, force_reprocess=True)

        assert result is not None
        processor._backend.extract.assert_called_once()

    def test_page_cap_skips_long_pdf(self, test_storage, processor_with_mock_backend):
        """Verify that PDFs exceeding max_pages are skipped."""
        processor = processor_with_mock_backend
        processor.max_pages = 5  # very low cap

        pdf_dir = test_storage.get_path("raw/documents/growthlab/pub789")
        pdf_dir.mkdir(parents=True)
        pdf_path = pdf_dir / "big.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")

        # Mock pypdfium2 to report a high page count
        mock_doc = MagicMock()
        mock_doc.__len__ = lambda self: 300
        with patch.dict("sys.modules", {"pypdfium2": MagicMock()}):
            import sys

            sys.modules["pypdfium2"].PdfDocument.return_value = mock_doc
            result = processor.process_pdf(pdf_path)

        assert result is None
        processor._backend.extract.assert_not_called()

    def test_upload_called_after_processing(
        self, test_storage, processor_with_mock_backend
    ):
        """Verify that storage.upload() is called after writing output."""
        processor = processor_with_mock_backend

        # Spy on the upload method
        processor.storage = MagicMock(wraps=test_storage)
        processor.storage.get_path = test_storage.get_path
        processor.storage.ensure_dir = test_storage.ensure_dir
        processor.storage.exists = test_storage.exists

        pdf_dir = test_storage.get_path("raw/documents/growthlab/pub_up")
        pdf_dir.mkdir(parents=True)
        pdf_path = pdf_dir / "paper.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")

        result = processor.process_pdf(pdf_path)
        assert result is not None
        processor.storage.upload.assert_called_once()


class TestFindGrowthLabPdfs:
    """Test the find_growth_lab_pdfs function."""

    def test_finds_pdfs_via_storage_glob(self, tmp_path):
        """Verify that find_growth_lab_pdfs uses storage.glob()."""
        storage = LocalStorage(base_path=tmp_path)

        # Create some PDF files
        pdf_dir = tmp_path / "raw" / "documents" / "growthlab" / "pub1"
        pdf_dir.mkdir(parents=True)
        (pdf_dir / "paper.pdf").write_bytes(b"%PDF-1.4 content")

        pdf_dir2 = tmp_path / "raw" / "documents" / "growthlab" / "pub2"
        pdf_dir2.mkdir(parents=True)
        (pdf_dir2 / "report.pdf").write_bytes(b"%PDF-1.4 content")

        # Also a non-PDF with PDF magic bytes
        (pdf_dir2 / "mystery.bin").write_bytes(b"%PDF-1.4 sneaky")

        results = find_growth_lab_pdfs(storage)
        assert len(results) == 3  # 2 .pdf files + 1 magic-byte match

    def test_returns_empty_when_no_dir(self, tmp_path):
        """Verify empty list when raw dir doesn't exist."""
        storage = LocalStorage(base_path=tmp_path)
        results = find_growth_lab_pdfs(storage)
        assert results == []


# ---------------------------------------------------------------------------
# Integration tests — use dev config (unstructured backend, fast)
# ---------------------------------------------------------------------------


class TestPDFProcessorIntegration:
    """Integration tests that run real PDF extraction via the dev config.

    Uses the unstructured backend (fast strategy) so no heavy model
    downloads are needed.
    """

    @pytest.fixture
    def sample_pdfs(self):
        fixtures_dir = Path(__file__).parent / "fixtures" / "pdfs"
        assert fixtures_dir.exists(), f"Fixtures directory {fixtures_dir} not found"
        sample1_path = fixtures_dir / "sample1.pdf"
        assert sample1_path.exists(), f"Sample PDF {sample1_path} not found"
        return [sample1_path]

    @pytest.fixture
    def test_storage(self):
        temp_dir = Path(tempfile.mkdtemp())
        raw_dir = temp_dir / "raw" / "documents" / "growthlab" / "test_pub"
        raw_dir.mkdir(parents=True, exist_ok=True)
        storage = LocalStorage(base_path=temp_dir)
        yield storage
        shutil.rmtree(temp_dir)

    def test_process_single_pdf(self, sample_pdfs, test_storage, dev_config_path):
        """Test processing a single PDF file with the dev backend."""
        pdf_path = sample_pdfs[0]
        dest_path = test_storage.get_path("raw/documents/growthlab/test_pub/sample.pdf")
        shutil.copy(pdf_path, dest_path)

        processor = PDFProcessor(storage=test_storage, config_path=dev_config_path)
        result_path = processor.process_pdf(dest_path)

        assert result_path is not None, "PDF processing failed"
        assert result_path.exists()

        text_content = result_path.read_text()
        assert len(text_content) > 100, "Output text too short for a real PDF"

        logger.info(f"Processed PDF: {len(text_content)} characters")
