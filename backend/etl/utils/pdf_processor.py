"""
PDF processing and OCR module for extracting text from PDF documents.

This module delegates to configurable backends (Marker primary, Docling fallback)
for actual PDF text extraction. The PDFProcessor class is a facade that handles
file management, status tracking, and output writing.
"""

import hashlib
import logging
from pathlib import Path

import yaml
from tqdm import tqdm

from backend.etl.models.tracking import ProcessingStatus
from backend.etl.utils.pdf_backends import get_backend
from backend.etl.utils.pdf_backends.base import PDFBackend
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
        self._full_config = self._load_config(config_path)
        self.config = self._full_config.get("ocr", {})

        # Processing settings
        self.min_chars_per_page = self.config.get("min_chars_per_page", 100)

        # Storage paths
        self.processed_root = self._full_config.get(
            "processed_root", "processed/documents/growthlab"
        )

        # Page cap — skip extremely long PDFs that would OOM the backend
        self.max_pages = self.config.get("max_pages", 0)  # 0 = no limit

        # Initialize the extraction backend
        self._backend = self._init_backend()

    def _load_config(self, config_path: Path | None) -> dict:
        """Load file_processing configuration from YAML file."""
        if not config_path:
            config_path = Path(__file__).parent.parent / "config.yaml"

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                return config.get("file_processing", {})
        except Exception as e:
            logger.warning(f"Error loading PDF processor config: {e}. Using defaults.")

        return {
            "ocr": {
                "default_model": "marker",
                "min_chars_per_page": 100,
            },
            "processed_root": "processed/documents/growthlab",
        }

    def _init_backend(self) -> PDFBackend:
        """Initialize the configured PDF extraction backend with fallback.

        Tries the configured default backend first (marker). If that fails
        to import, falls back to docling.
        """
        backend_name = self.config.get("default_model", "marker")
        fallback_name = "docling" if backend_name == "marker" else "marker"

        try:
            backend_config = self.config.get(backend_name, {})
            logger.info(f"Initializing PDF backend: {backend_name}")
            return get_backend(backend_name, config=backend_config)
        except (ImportError, ValueError) as e:
            logger.warning(
                f"Failed to initialize {backend_name}: {e}. "
                f"Falling back to {fallback_name}."
            )
            fallback_config = self.config.get(fallback_name, {})
            return get_backend(fallback_name, config=fallback_config)

    def _get_processed_relative(self, raw_file_path: Path) -> str:
        """Return the storage-relative path for the processed output.

        Derives the source directory (growthlab, openalex, …) from the input
        path so that each source's outputs stay in their own tree.

        E.g. ``"processed/documents/growthlab/<pub_id>/<stem>.txt"``
             ``"processed/documents/openalex/<pub_id>/<stem>.txt"``
        """
        try:
            relative_path = raw_file_path.relative_to(
                self.storage.get_path("raw/documents")
            )
            # parts[0] = source dir (growthlab, openalex, …)
            # parts[1] = publication_id
            source = relative_path.parts[0]
            pub_id = relative_path.parts[1]
            base_name = raw_file_path.stem
            return f"processed/documents/{source}/{pub_id}/{base_name}.txt"
        except (ValueError, IndexError):
            file_hash = hashlib.md5(str(raw_file_path).encode()).hexdigest()[:8]
            return f"processed/documents/unknown/{file_hash}.txt"

    def _get_processed_path(self, raw_file_path: Path) -> Path:
        """
        Determine the path where processed text should be saved.

        Args:
            raw_file_path: Original PDF file path

        Returns:
            Path where processed text file should be saved
        """
        return self.storage.get_path(self._get_processed_relative(raw_file_path))

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

        # Get output paths
        output_relative = self._get_processed_relative(pdf_path)
        output_path = self.storage.get_path(output_relative)

        # Check if already processed (uses GCS existence check for cloud)
        if self.storage.exists(output_relative) and not force_reprocess:
            logger.info(f"PDF already processed: {pdf_path} -> {output_relative}")
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

        # Page cap — skip extremely long PDFs
        if self.max_pages > 0:
            try:
                import pypdfium2 as pdfium

                pdf_doc = pdfium.PdfDocument(pdf_path)
                num_pages = len(pdf_doc)
                pdf_doc.close()
                if num_pages > self.max_pages:
                    error_msg = (
                        f"PDF has {num_pages} pages, exceeds max_pages={self.max_pages}"
                    )
                    logger.warning(f"Skipping {pdf_path}: {error_msg}")
                    if self.tracker and publication_id:
                        try:
                            self.tracker.update_processing_status(
                                publication_id,
                                ProcessingStatus.FAILED,
                                error=error_msg,
                            )
                        except Exception as tracker_error:
                            logger.warning(
                                f"Failed to update processing status for "
                                f"{publication_id}: {tracker_error}"
                            )
                    return None
            except Exception as e:
                logger.debug(f"Could not count pages for {pdf_path}: {e}")

        # Ensure output directory exists
        self.storage.ensure_dir(output_path.parent)

        try:
            # Delegate extraction to the configured backend
            result = self._backend.extract(pdf_path)

            if not result.success:
                logger.warning(
                    f"Backend extraction failed for {pdf_path}: {result.error}"
                )
                # Update tracker to FAILED
                if self.tracker and publication_id:
                    try:
                        self.tracker.update_processing_status(
                            publication_id,
                            ProcessingStatus.FAILED,
                            error=result.error or "Extraction failed",
                        )
                    except Exception as tracker_error:
                        logger.warning(
                            f"Failed to update processing status to FAILED for "
                            f"{publication_id}: {tracker_error}"
                        )
                return None

            full_text = result.text

            # Skip if the extracted text is too short (likely a failed extraction)
            if len(full_text) < self.min_chars_per_page:
                logger.warning(
                    f"Extracted text too short ({len(full_text)} chars): {pdf_path}"
                )
                return None

            # Write extracted text to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_text)

            # Upload to remote storage (no-op for local)
            self.storage.upload(output_relative)

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


def find_pdfs(
    storage: StorageBase | None = None,
    source: str | None = None,
) -> list[Path]:
    """Find PDF files in raw storage, optionally filtered by source.

    Args:
        storage: Storage backend to use.
        source: Restrict to a specific source directory (e.g. ``"growthlab"``,
            ``"openalex"``).  When ``None``, searches all subdirectories of
            ``raw/documents/``.

    Returns:
        List of local PDF file paths ready for processing.
    """
    storage = storage or get_storage()

    base_dir = f"raw/documents/{source}" if source else "raw/documents"
    if not storage.exists(base_dir):
        logger.warning(f"Documents directory not found in storage: {base_dir}")
        return []

    # Find .pdf files
    pdf_relatives = storage.glob(f"{base_dir}/**/*.pdf")
    pdf_files: list[Path] = []
    for rel in pdf_relatives:
        local_path = storage.download(rel)
        pdf_files.append(local_path)

    # Also check files without .pdf extension for PDF magic bytes
    all_relatives = storage.glob(f"{base_dir}/**/*")
    pdf_rel_set = set(pdf_relatives)
    for rel in all_relatives:
        if rel in pdf_rel_set or rel.lower().endswith(".pdf"):
            continue
        local_path = storage.download(rel)
        if local_path.is_file():
            try:
                with open(local_path, "rb") as fh:
                    if fh.read(5) == b"%PDF-":
                        pdf_files.append(local_path)
                        logger.info(f"Found PDF with wrong extension: {rel}")
            except (OSError, PermissionError):
                pass

    logger.info(f"Found {len(pdf_files)} PDF files in {base_dir}")
    return pdf_files


def find_growth_lab_pdfs(storage: StorageBase | None = None) -> list[Path]:
    """
    Find all Growth Lab PDF files in the raw storage.

    Uses ``storage.glob()`` so that cloud deployments can discover files
    in GCS without relying on local filesystem state, then downloads
    each PDF to the local cache so backends can open them.

    Args:
        storage: Storage backend to use

    Returns:
        List of local PDF file paths ready for processing
    """
    storage = storage or get_storage()

    # Check if the raw documents directory exists in storage
    if not storage.exists("raw/documents/growthlab"):
        logger.warning("Growth Lab documents directory not found in storage")
        return []

    # Find all PDF files via storage glob
    pdf_relatives = storage.glob("raw/documents/growthlab/**/*.pdf")
    pdf_files: list[Path] = []
    for rel in pdf_relatives:
        local_path = storage.download(rel)
        pdf_files.append(local_path)

    # Also check files without .pdf extension for PDF magic bytes
    all_relatives = storage.glob("raw/documents/growthlab/**/*")
    pdf_rel_set = set(pdf_relatives)
    for rel in all_relatives:
        if rel in pdf_rel_set or rel.lower().endswith(".pdf"):
            continue
        local_path = storage.download(rel)
        if local_path.is_file():
            try:
                with open(local_path, "rb") as fh:
                    if fh.read(5) == b"%PDF-":
                        pdf_files.append(local_path)
                        logger.info(f"Found PDF with wrong extension: {rel}")
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
