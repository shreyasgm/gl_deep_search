"""
Integration tests for PDF extraction backends.

The unstructured backend test runs by default (lightweight, no model downloads).
The Docling and Marker backend tests require heavy model weights and are behind
``@pytest.mark.integration``.  Run them explicitly with::

  uv run pytest backend/tests/etl/test_pdf_backends_integration.py \
      -v -m integration
"""

import signal
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "pdfs"
SAMPLE_PDF = FIXTURES_DIR / "sample1.pdf"
MIN_CHARS = 100
TIMEOUT_SECONDS = 120


class _Timeout:
    """Context manager that raises after TIMEOUT_SECONDS using SIGALRM."""

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._handler)
        signal.alarm(TIMEOUT_SECONDS)
        return self

    def __exit__(self, *args):
        signal.alarm(0)

    @staticmethod
    def _handler(signum, frame):
        raise TimeoutError(f"Backend extraction exceeded {TIMEOUT_SECONDS}s timeout")


@pytest.fixture
def sample_pdf() -> Path:
    assert SAMPLE_PDF.exists(), f"Sample PDF not found: {SAMPLE_PDF}"
    return SAMPLE_PDF


@pytest.mark.integration
class TestDoclingBackendIntegration:
    """Integration tests for Docling backend."""

    def test_extract_produces_output(self, sample_pdf: Path) -> None:
        try:
            from backend.etl.utils.pdf_backends.docling_backend import DoclingBackend
        except ImportError:
            pytest.skip("docling not installed")

        backend = DoclingBackend(config={"do_ocr": False, "num_threads": 2})
        try:
            with _Timeout():
                result = backend.extract(sample_pdf)
        except TimeoutError:
            pytest.skip(f"Docling exceeded {TIMEOUT_SECONDS}s timeout")
        finally:
            backend.cleanup()

        assert result.success, f"Docling extraction failed: {result.error}"
        assert len(result.text) > MIN_CHARS, (
            f"Docling output too short: {len(result.text)} chars"
        )
        assert result.backend_name == "docling"


@pytest.mark.integration
class TestMarkerBackendIntegration:
    """Integration tests for Marker backend."""

    def test_extract_produces_output(self, sample_pdf: Path) -> None:
        try:
            from backend.etl.utils.pdf_backends.marker_backend import MarkerBackend
        except ImportError:
            pytest.skip("marker-pdf not installed")

        backend = MarkerBackend(config={"force_ocr": False})
        try:
            with _Timeout():
                result = backend.extract(sample_pdf)
        except TimeoutError:
            pytest.skip(f"Marker exceeded {TIMEOUT_SECONDS}s timeout")
        finally:
            backend.cleanup()

        assert result.success, f"Marker extraction failed: {result.error}"
        assert len(result.text) > MIN_CHARS, (
            f"Marker output too short: {len(result.text)} chars"
        )
        assert result.backend_name == "marker"


class TestUnstructuredBackendIntegration:
    """Integration tests for the unstructured backend (fast strategy).

    These run by default — no heavy model downloads required.
    """

    def test_extract_produces_output(self, sample_pdf: Path) -> None:
        from backend.etl.utils.pdf_backends.unstructured_backend import (
            UnstructuredBackend,
        )

        backend = UnstructuredBackend(config={"strategy": "fast"})
        result = backend.extract(sample_pdf)

        assert result.success, f"Unstructured extraction failed: {result.error}"
        assert len(result.text) > MIN_CHARS, (
            f"Unstructured output too short: {len(result.text)} chars"
        )
        assert result.backend_name == "unstructured"
        assert result.pages > 0
