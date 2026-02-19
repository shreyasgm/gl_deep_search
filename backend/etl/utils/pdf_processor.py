"""
PDF processing and OCR module for extracting text from PDF documents.

This module uses the unstructured library to extract text from PDF files downloaded
by the Growth Lab scraper or other sources. It handles OCR, layout analysis, and
text extraction in a consistent way.
"""

import hashlib
import logging
from pathlib import Path

import yaml
from tqdm import tqdm
from unstructured.partition.pdf import partition_pdf

from backend.etl.models.tracking import ProcessingStatus
from backend.etl.utils.publication_tracker import PublicationTracker
from backend.storage.base import StorageBase
from backend.storage.factory import get_storage

# Configure logger
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Process PDF files to extract text and metadata."""

    def __init__(
        self,
        storage: StorageBase | None = None,
        config_path: Path | None = None,
        tracker: PublicationTracker | None = None,
    ):
        """
        Initialize the PDF processor.

        Args:
            storage: Storage backend to use (defaults to factory-configured storage)
            config_path: Path to configuration file
            tracker: Optional PublicationTracker instance for updating processing status
        """
        # Storage configuration
        self.storage = storage or get_storage()
        self.tracker = tracker

        # Load configuration or use defaults
        self.config = self._load_config(config_path)

        # Processing settings
        self.ocr_languages = self.config.get("ocr_languages", ["eng"])
        self.min_chars_per_page = self.config.get("min_chars_per_page", 100)
        self.extract_images = self.config.get("extract_images", False)

        # PDF specific processing settings
        self.split_pdf_page = self.config.get("split_pdf_page", True)
        self.split_pdf_allow_failed = self.config.get("split_pdf_allow_failed", True)

        # Storage paths
        self.processed_root = self.config.get(
            "processed_root", "processed/documents/growthlab"
        )

    def _load_config(self, config_path: Path | None) -> dict:
        """Load configuration from YAML file."""
        if not config_path:
            # Look for config in standard location
            config_path = Path(__file__).parent.parent / "config.yaml"

            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                    return config.get("pdf_processor", {})
            except Exception as e:
                logger.warning(
                    f"Error loading PDF processor config: {e}. Using defaults."
                )

        # Default configuration
        return {
            "ocr_languages": ["eng"],
            "min_chars_per_page": 100,
            "extract_images": False,
            "processed_root": "processed/documents/growthlab",
            "split_pdf_page": True,
            "split_pdf_allow_failed": True,
        }

    def _get_processed_path(self, raw_file_path: Path) -> Path:
        """
        Determine the path where processed text should be saved.

        Args:
            raw_file_path: Original PDF file path

        Returns:
            Path where processed text file should be saved
        """
        # Extract relevant parts from the raw path
        # Expected path: data/raw/documents/growthlab/<publication_id>/<filename>.pdf
        try:
            relative_path = raw_file_path.relative_to(
                self.storage.get_path("raw/documents")
            )
            pub_id = relative_path.parts[1]  # growthlab, <publication_id>, <filename>

            # Create processed path:
            # processed/documents/growthlab/<publication_id>/<filename>.txt
            base_name = raw_file_path.stem
            processed_path = self.storage.get_path(
                f"{self.processed_root}/{pub_id}/{base_name}.txt"
            )
            return processed_path
        except ValueError:
            # If we can't determine the proper path, use a hash-based path
            file_hash = hashlib.md5(str(raw_file_path).encode()).hexdigest()[:8]
            return self.storage.get_path(
                f"{self.processed_root}/unknown/{file_hash}.txt"
            )

    def _extract_publication_id(self, pdf_path: Path) -> str | None:
        """
        Extract publication ID from PDF file path.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Publication ID if found, None otherwise
        """
        try:
            relative_path = pdf_path.relative_to(self.storage.get_path("raw/documents"))
            # Expected structure: <source_type>/<publication_id>/<filename>.pdf
            if len(relative_path.parts) >= 2:
                return relative_path.parts[1]
        except (ValueError, IndexError):
            pass
        return None

    def process_pdf(self, pdf_path: Path, force_reprocess: bool = False) -> Path | None:
        """
        Process a single PDF file to extract text.

        Args:
            pdf_path: Path to the PDF file
            force_reprocess: Whether to reprocess even if output already exists

        Returns:
            Path to the processed text file, or None if processing failed
        """
        # Check if PDF exists
        if not pdf_path.exists():
            logger.warning(f"PDF file not found: {pdf_path}")
            return None

        # Extract publication ID for tracking
        publication_id = self._extract_publication_id(pdf_path)

        # Update status to IN_PROGRESS if tracker is available
        if self.tracker and publication_id:
            try:
                self.tracker.update_processing_status(
                    publication_id, ProcessingStatus.IN_PROGRESS
                )
            except Exception as e:
                logger.warning(
                    f"Failed to update processing status to IN_PROGRESS for "
                    f"{publication_id}: {e}"
                )

        # Get output path
        output_path = self._get_processed_path(pdf_path)

        # Check if already processed
        if output_path.exists() and not force_reprocess:
            logger.info(f"PDF already processed: {pdf_path} -> {output_path}")
            # Still update status if tracker is available
            if self.tracker and publication_id:
                try:
                    self.tracker.update_processing_status(
                        publication_id, ProcessingStatus.PROCESSED
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to update processing status for {publication_id}: {e}"
                    )
            return output_path

        # Ensure output directory exists
        self.storage.ensure_dir(output_path.parent)

        try:
            # Use unstructured partition function
            pdf_elements = partition_pdf(
                filename=str(pdf_path),
                extract_images=self.extract_images,
                languages=self.ocr_languages,
                strategy="fast",  # Use fast strategy by default
                infer_table_structure=True,
                chunking_strategy="by_title",
                hi_res_model_name=None,  # Use default model
                split_pdf_page=self.split_pdf_page,
                split_pdf_allow_failed=self.split_pdf_allow_failed,
            )

            # Build text content from elements, preserving structural information
            text_content: list[str] = []
            for element in pdf_elements:
                # Extract element text and metadata
                element_text = str(element)

                # Add metadata if available (based on unstructured element types)
                if hasattr(element, "metadata") and element.metadata:
                    # Add page number if available
                    if (
                        hasattr(element.metadata, "page_number")
                        and element.metadata.page_number is not None
                    ):
                        page_num = element.metadata.page_number
                        if not text_content or not text_content[-1].startswith(
                            f"--- Page {page_num} ---"
                        ):
                            text_content.append(f"--- Page {page_num} ---")

                    # For table elements, add special formatting
                    if getattr(element, "category", "") == "Table":
                        text_content.append("--- Table Start ---")
                        text_content.append(element_text)
                        text_content.append("--- Table End ---")
                    # For title elements, add formatting
                    elif getattr(element, "category", "") == "Title":
                        text_content.append(f"# {element_text}")
                    # For list items, add formatting
                    elif getattr(element, "category", "") == "ListItem":
                        text_content.append(f"• {element_text}")
                    else:
                        text_content.append(element_text)
                else:
                    text_content.append(element_text)

            # Join all text content with line breaks
            full_text = "\n\n".join(text_content)

            # Skip if the extracted text is too short (likely a failed extraction)
            if len(full_text) < self.min_chars_per_page:
                logger.warning(
                    f"Extracted text too short ({len(full_text)} chars): {pdf_path}"
                )
                return None

            # Write extracted text to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_text)

            logger.info(f"Successfully processed PDF: {pdf_path} -> {output_path}")

            # Update status to PROCESSED on success
            if self.tracker and publication_id:
                try:
                    self.tracker.update_processing_status(
                        publication_id, ProcessingStatus.PROCESSED
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to update processing status to PROCESSED for "
                        f"{publication_id}: {e}"
                    )

            return output_path

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")

            # Update status to FAILED on error
            if self.tracker and publication_id:
                try:
                    self.tracker.update_processing_status(
                        publication_id, ProcessingStatus.FAILED, error=str(e)
                    )
                except Exception as tracker_error:
                    logger.warning(
                        f"Failed to update processing status to FAILED for "
                        f"{publication_id}: {tracker_error}"
                    )

            return None

    def process_pdfs(
        self,
        pdf_paths: list[Path],
        force_reprocess: bool = False,
        show_progress: bool = True,
    ) -> dict[Path, Path | None]:
        """
        Process multiple PDF files to extract text.

        Args:
            pdf_paths: List of paths to PDF files
            force_reprocess: Whether to reprocess even if output already exists
            show_progress: Whether to show a progress bar

        Returns:
            Dictionary mapping input PDF paths to output text paths (None for failures)
        """
        results = {}

        if show_progress:
            for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
                result = self.process_pdf(pdf_path, force_reprocess)
                results[pdf_path] = result

                # Log result
                status = "success" if result else "failed"
                logger.info(f"Processed PDF: {pdf_path} - {status}")
        else:
            # Process without progress updates
            for pdf_path in pdf_paths:
                try:
                    result = self.process_pdf(pdf_path, force_reprocess)
                    results[pdf_path] = result
                except Exception as e:
                    logger.error(f"Error processing {pdf_path}: {e}")
                    results[pdf_path] = None

        # Calculate success rate
        success_count = sum(1 for result in results.values() if result is not None)
        total_count = len(results)
        success_rate = success_count / total_count if total_count > 0 else 0

        logger.info(
            f"PDF processing complete: "
            f"{success_count}/{total_count} "
            f"successful ({success_rate:.1%})"
        )

        return results


def find_growth_lab_pdfs(storage: StorageBase | None = None) -> list[Path]:
    """
    Find all Growth Lab PDF files in the raw storage.

    Args:
        storage: Storage backend to use

    Returns:
        List of PDF file paths
    """
    storage = storage or get_storage()
    raw_path = storage.get_path("raw/documents/growthlab")

    # Ensure the path exists
    if not raw_path.exists():
        logger.warning(f"Growth Lab documents directory not found: {raw_path}")
        return []

    # Find all PDF files recursively
    pdf_files = list(raw_path.glob("**/*.pdf"))

    # Also check files without .pdf extension for PDF magic bytes.
    # This catches files that were downloaded before the Content-Type
    # rename fix, or where the server didn't return a Content-Type header.
    pdf_file_set = set(pdf_files)
    for f in raw_path.glob("**/*"):
        if f.is_file() and f.suffix.lower() != ".pdf" and f not in pdf_file_set:
            try:
                with open(f, "rb") as fh:
                    if fh.read(5) == b"%PDF-":
                        pdf_files.append(f)
                        logger.info(f"Found PDF with wrong extension: {f}")
            except (OSError, PermissionError):
                pass

    logger.info(f"Found {len(pdf_files)} Growth Lab PDF files")
    return pdf_files


def process_growth_lab_pdfs(
    storage: StorageBase | None = None,
    force_reprocess: bool = False,
    config_path: Path | None = None,
) -> dict[Path, Path | None]:
    """
    Process all Growth Lab PDF files to extract text.

    Args:
        storage: Storage backend to use
        force_reprocess: Whether to reprocess even if output already exists
        config_path: Path to configuration file

    Returns:
        Dictionary mapping input PDF paths to output text paths (None for failures)
    """
    # Get storage
    storage = storage or get_storage()

    # Find all PDF files
    pdf_files = find_growth_lab_pdfs(storage)

    if not pdf_files:
        logger.warning("No Growth Lab PDF files found to process")
        return {}

    # Create processor and process PDFs
    processor = PDFProcessor(
        storage=storage,
        config_path=config_path,
    )

    results = processor.process_pdfs(
        pdf_files,
        force_reprocess=force_reprocess,
        show_progress=True,
    )

    return results
