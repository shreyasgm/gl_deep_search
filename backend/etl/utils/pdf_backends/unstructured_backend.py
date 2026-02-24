"""
Unstructured-based PDF extraction backend.

Uses the unstructured library with strategy="fast" for lightweight local
development. Avoids loading heavy ML models (layout, OCR, recognition)
that Marker and Docling require, making it practical for CPU/MPS iteration.
"""

from pathlib import Path

from loguru import logger

from backend.etl.utils.pdf_backends import register_backend
from backend.etl.utils.pdf_backends.base import ExtractionResult, PDFBackend


class UnstructuredBackend(PDFBackend):
    """PDF extraction backend using unstructured (fast strategy)."""

    def extract(self, pdf_path: Path) -> ExtractionResult:
        """Extract text from a PDF using unstructured's partition_pdf."""
        try:
            from unstructured.partition.pdf import partition_pdf

            strategy = self.config.get("strategy", "fast")
            languages = self.config.get("languages", ["eng"])
            extract_images = self.config.get("extract_images", False)

            logger.debug(
                f"Extracting {pdf_path} with unstructured "
                f"(strategy={strategy}, languages={languages})"
            )

            elements = partition_pdf(
                filename=str(pdf_path),
                strategy=strategy,
                languages=languages,
                extract_image_block_to_payload=extract_images,
            )

            text = "\n\n".join(str(el) for el in elements)

            # Count unique page numbers across elements
            page_numbers: set[int] = set()
            for el in elements:
                if hasattr(el, "metadata") and hasattr(el.metadata, "page_number"):
                    page_num = el.metadata.page_number
                    if page_num is not None:
                        page_numbers.add(page_num)

            return ExtractionResult(
                text=text,
                pages=len(page_numbers),
                backend_name="unstructured",
                metadata={"strategy": strategy, "num_elements": len(elements)},
            )

        except Exception as e:
            logger.error(f"Unstructured extraction failed for {pdf_path}: {e}")
            return ExtractionResult(
                text="",
                pages=0,
                backend_name="unstructured",
                success=False,
                error=str(e),
            )


register_backend("unstructured", UnstructuredBackend)
