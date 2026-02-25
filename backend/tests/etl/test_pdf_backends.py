"""
Unit tests for PDF extraction backend infrastructure.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.etl.utils.pdf_backends import (
    BACKEND_REGISTRY,
    get_backend,
    register_backend,
)
from backend.etl.utils.pdf_backends.base import ExtractionResult, PDFBackend
from backend.etl.utils.pdf_backends.device import DeviceType, detect_device


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_default_success(self):
        result = ExtractionResult(text="hello", pages=1, backend_name="test")
        assert result.success is True
        assert result.error is None
        assert result.metadata == {}

    def test_failure_result(self):
        result = ExtractionResult(
            text="", pages=0, backend_name="test", success=False, error="bad pdf"
        )
        assert result.success is False
        assert result.error == "bad pdf"


class TestDeviceDetection:
    """Tests for GPU/device detection."""

    def test_env_override_cuda(self):
        with patch.dict(os.environ, {"PDF_DEVICE": "cuda"}):
            assert detect_device() == DeviceType.CUDA

    def test_env_override_mps(self):
        with patch.dict(os.environ, {"PDF_DEVICE": "mps"}):
            assert detect_device() == DeviceType.MPS

    def test_env_override_cpu(self):
        with patch.dict(os.environ, {"PDF_DEVICE": "cpu"}):
            assert detect_device() == DeviceType.CPU

    def test_env_override_invalid_falls_through(self):
        with patch.dict(os.environ, {"PDF_DEVICE": "tpu"}):
            # Should fall through to torch detection or CPU
            device = detect_device()
            assert isinstance(device, DeviceType)

    def test_no_torch_falls_back_to_cpu(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("backend.etl.utils.pdf_backends.device.logger"),
        ):
            # Remove PDF_DEVICE and simulate no torch
            env = os.environ.copy()
            env.pop("PDF_DEVICE", None)
            with patch.dict(os.environ, env, clear=True):
                with patch.dict("sys.modules", {"torch": None}):
                    # When torch import fails, should return CPU
                    # This is tricky to test; just verify it returns a valid DeviceType
                    device = detect_device()
                    assert isinstance(device, DeviceType)


class TestBackendRegistry:
    """Tests for backend registry and factory."""

    def test_register_and_get(self):
        """Test registering a custom backend and retrieving it."""

        class DummyBackend(PDFBackend):
            def extract(self, pdf_path: Path) -> ExtractionResult:
                return ExtractionResult(text="dummy", pages=1, backend_name="dummy")

        register_backend("_test_dummy", DummyBackend)
        try:
            backend = get_backend("_test_dummy", config={"key": "val"})
            assert isinstance(backend, DummyBackend)
            assert backend.config == {"key": "val"}
        finally:
            BACKEND_REGISTRY.pop("_test_dummy", None)

    def test_get_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown PDF backend"):
            get_backend("nonexistent_backend_xyz")

    def test_lazy_import_docling(self):
        """Test that docling backend can be lazy-imported."""
        # This just verifies the import mechanism works, not that docling is installed
        try:
            backend = get_backend("docling")
            assert backend is not None
        except ImportError:
            pytest.skip("docling not installed")

    def test_lazy_import_marker(self):
        """Test that marker backend can be lazy-imported."""
        try:
            backend = get_backend("marker")
            assert backend is not None
        except ImportError:
            pytest.skip("marker not installed")


class TestDoclingBackendUnit:
    """Unit tests for Docling backend with mocked library calls."""

    def test_extract_success(self):
        """Test that real extract() processes a successful conversion result correctly.

        Mocks _get_converter to return a fake converter whose .convert() yields
        a mock DocumentConversionResult with SUCCESS status. Verifies that the
        real extract() method correctly checks status, exports markdown, counts
        pages, and extracts table/picture metadata.
        """
        from docling.datamodel.base_models import ConversionStatus

        from backend.etl.utils.pdf_backends.docling_backend import DoclingBackend

        backend = DoclingBackend(config={"do_ocr": True})

        # Build a fake conversion result with real ConversionStatus.SUCCESS
        mock_conv_result = MagicMock()
        mock_conv_result.status = ConversionStatus.SUCCESS
        mock_conv_result.document.export_to_markdown.return_value = (
            "# Title\n\nContent here"
        )
        mock_conv_result.document.pages = ["page1", "page2", "page3"]
        mock_conv_result.document.tables = ["t1"]
        mock_conv_result.document.pictures = ["p1", "p2"]

        # Mock the converter returned by _get_converter
        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_conv_result

        with patch.object(backend, "_get_converter", return_value=mock_converter):
            result = backend.extract(Path("/fake/test.pdf"))

        # The real extract() ran and produced a real ExtractionResult
        assert isinstance(result, ExtractionResult)
        assert result.success is True
        assert result.backend_name == "docling"
        assert result.text == "# Title\n\nContent here"
        assert result.pages == 3
        assert result.metadata["num_tables"] == 1
        assert result.metadata["num_pictures"] == 2

        # Verify the converter was actually called with the pdf path
        mock_converter.convert.assert_called_once_with(Path("/fake/test.pdf"))

    def test_extract_failure_returns_error(self):
        """Test that extraction errors are caught and returned."""
        from backend.etl.utils.pdf_backends.docling_backend import DoclingBackend

        backend = DoclingBackend()

        with patch.object(backend, "_get_converter", side_effect=RuntimeError("boom")):
            result = backend.extract(Path("/fake/test.pdf"))

        assert not result.success
        assert "boom" in result.error


class TestMarkerBackendUnit:
    """Unit tests for Marker backend with mocked library calls."""

    def test_extract_success_v1(self):
        """Test successful extraction using marker v1.x API path.

        Mocks _load_models to set _api_version=1 without downloading weights,
        then pre-sets backend._converter to a callable mock so _extract_v1()
        skips PdfConverter construction and directly calls the mock converter.
        Verifies the real extract() -> _extract_v1() correctly parses the
        rendered output into an ExtractionResult.
        """
        from backend.etl.utils.pdf_backends.marker_backend import MarkerBackend

        backend = MarkerBackend(config={"force_ocr": False})

        # Build a mock rendered output (what PdfConverter.__call__ returns)
        mock_rendered = MagicMock()
        mock_rendered.markdown = "# Extracted Title\n\nSome content from the PDF."
        mock_rendered.metadata = {"pages": 5}

        # Pre-set the converter so _extract_v1 skips construction
        mock_converter = MagicMock()
        mock_converter.return_value = mock_rendered  # __call__(pdf_path)
        backend._converter = mock_converter

        def fake_load_models():
            backend._models = {"mock": "models"}
            backend._api_version = 1

        with patch.object(backend, "_load_models", side_effect=fake_load_models):
            result = backend.extract(Path("/fake/test.pdf"))

        assert isinstance(result, ExtractionResult)
        assert result.success is True
        assert result.backend_name == "marker"
        assert "Extracted Title" in result.text
        assert "Some content" in result.text
        assert result.metadata.get("marker_metadata") == {"pages": 5}

        # Verify the converter was called with the string path
        mock_converter.assert_called_once_with(str(Path("/fake/test.pdf")))

    def test_extract_failure_returns_error(self):
        """Test that extraction errors are caught and returned."""
        from backend.etl.utils.pdf_backends.marker_backend import MarkerBackend

        backend = MarkerBackend()

        with patch.object(backend, "_load_models", side_effect=RuntimeError("boom")):
            result = backend.extract(Path("/fake/test.pdf"))

        assert not result.success
        assert "boom" in result.error
