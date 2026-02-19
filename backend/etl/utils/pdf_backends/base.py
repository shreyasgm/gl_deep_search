"""
Abstract base class and data models for PDF extraction backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExtractionResult:
    """Result of a PDF text extraction operation."""

    text: str
    pages: int
    backend_name: str
    metadata: dict = field(default_factory=dict)
    success: bool = True
    error: str | None = None


class PDFBackend(ABC):
    """Abstract base class for PDF extraction backends."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def extract(self, pdf_path: Path) -> ExtractionResult:
        """
        Extract text from a single PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            ExtractionResult with extracted text and metadata.
        """

    def extract_batch(self, pdf_paths: list[Path]) -> list[ExtractionResult]:
        """
        Extract text from multiple PDF files.

        Default implementation processes sequentially. Backends with native
        batch support should override this.

        Args:
            pdf_paths: List of paths to PDF files.

        Returns:
            List of ExtractionResult objects.
        """
        return [self.extract(path) for path in pdf_paths]

    def cleanup(self) -> None:  # noqa: B027
        """Release resources (models, GPU memory). Override in subclasses."""
