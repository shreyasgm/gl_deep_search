"""
ETL Pipeline Orchestrator for Growth Lab Deep Search.

This module orchestrates the complete ETL pipeline, executing components in sequence
with proper error handling, monitoring, and data validation.
"""

import argparse
import asyncio
import copy
import json
import sys
import tempfile
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from loguru import logger

from backend.etl.scrapers.growthlab import GrowthLabScraper
from backend.etl.utils.embeddings_generator import EmbeddingsGenerator
from backend.etl.utils.gl_file_downloader import FileDownloader
from backend.etl.utils.pdf_processor import PDFProcessor, find_pdfs
from backend.etl.utils.profiling import log_component_metrics, profile_operation
from backend.etl.utils.publication_tracker import PublicationTracker
from backend.etl.utils.text_chunker import TextChunker
from backend.storage.factory import get_storage

# Valid data source names for --sources flag
VALID_SOURCES = {"growthlab", "openalex", "lectures"}


class ComponentStatus(Enum):
    """Status of a pipeline component."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ComponentResult:
    """Result of a component execution."""

    component_name: str
    status: ComponentStatus
    start_time: float | None = None
    end_time: float | None = None
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    output_files: list[Path] = field(default_factory=list)

    @property
    def duration(self) -> float | None:
        """Calculate execution duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


@dataclass
class OrchestrationConfig:
    """Configuration for the orchestration pipeline."""

    # Global settings
    config_path: Path
    storage_type: str | None = None
    log_level: str = "INFO"
    dry_run: bool = False
    dev_mode: bool = False

    # Data sources to include (growthlab, openalex, lectures)
    sources: list[str] = field(default_factory=lambda: ["growthlab"])

    # Scraper settings
    skip_scraping: bool = False
    scraper_concurrency: int = 2
    scraper_delay: float = 2.0
    scraper_limit: int | None = None  # Limit number of publications to scrape

    # File downloader settings
    download_concurrency: int = 3
    download_limit: int | None = None
    overwrite_files: bool = False
    min_file_size: int = 1024
    max_file_size: int = 100_000_000

    # PDF processor settings
    force_reprocess: bool = False
    ocr_language: list[str] = field(default_factory=lambda: ["eng"])
    extract_images: bool = False
    min_chars_per_page: int = 100

    # Lecture transcripts settings
    transcripts_input: Path | None = None
    transcripts_limit: int | None = None
    max_tokens: int | None = None


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Deep-merge *overlay* into a copy of *base*.

    For nested dicts, values are merged recursively.
    For all other types, the overlay value wins.
    """
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class ETLOrchestrator:
    """Main orchestrator for the ETL pipeline."""

    def __init__(self, config: OrchestrationConfig):
        """Initialize the orchestrator with configuration."""
        self.config = config
        self._resolved_config_path: Path | None = None
        self.storage = get_storage(storage_type=config.storage_type)
        self.results: list[ComponentResult] = []

        # Create unified tracker instance for entire pipeline
        self.tracker = PublicationTracker()

        # Set up logging
        self._log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        self._configure_logger()

        # Load ETL configuration (applies dev overlay if --dev)
        self.etl_config = self._load_etl_config()

    def _configure_logger(self) -> None:
        """Configure loguru. Called on init and after components that may
        import third-party libraries which reset the global logger
        (e.g. scidownl calls logger.remove() at import time)."""
        logger.remove()
        logger.add(
            sys.stdout,
            level=self.config.log_level,
            format=self._log_format,
        )

    def _load_etl_config(self) -> dict[str, Any]:
        """Load ETL configuration from YAML file.

        When dev mode is active, deep-merges config.dev.yaml on top of the
        base config and writes the resolved result to a temp file so that
        downstream components (which load config by path) see the merged
        values automatically.
        """
        try:
            with open(self.config.config_path) as f:
                base_config = yaml.safe_load(f)
        except Exception as e:
            logger.error(
                f"Failed to load ETL config from {self.config.config_path}: {e}"
            )
            raise

        if not self.config.dev_mode:
            return base_config

        # Locate dev overlay relative to the base config file
        dev_overlay_path = self.config.config_path.parent / "config.dev.yaml"
        if not dev_overlay_path.exists():
            logger.warning(
                f"Dev mode requested but {dev_overlay_path} not found — "
                "running with base config only"
            )
            return base_config

        with open(dev_overlay_path) as f:
            dev_overlay = yaml.safe_load(f) or {}

        merged = _deep_merge(base_config, dev_overlay)
        logger.info(f"Dev mode: merged {dev_overlay_path.name} on top of base config")

        # Write resolved config to a temp file so components that load by
        # path see the merged values.  The file is kept alive for the
        # lifetime of the orchestrator via self._resolved_config_path.
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix="etl_config_resolved_",
            delete=False,
        )
        yaml.safe_dump(merged, tmp, default_flow_style=False, sort_keys=False)
        tmp.close()
        self._resolved_config_path = Path(tmp.name)
        self.config.config_path = self._resolved_config_path
        logger.debug(f"Resolved config written to {self._resolved_config_path}")

        return merged

    def _path_exists(self, path: Path) -> bool:
        """Check if a path exists via the local filesystem.

        Prefer ``self.storage.exists(relative_path)`` for new code.
        """
        return path.exists()

    async def run_pipeline(self) -> list[ComponentResult]:
        """Execute the complete ETL pipeline."""
        logger.info("Starting ETL pipeline orchestration")
        logger.info(f"Active sources: {', '.join(self.config.sources)}")

        # Enable expandable CUDA memory segments to reduce fragmentation.
        # Uses setdefault so user/sbatch overrides are respected.
        import os

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        if self.config.dry_run:
            logger.info("DRY RUN MODE - No actual processing will be performed")
            return await self._simulate_pipeline()

        # Build component list based on active sources
        sources = self.config.sources
        components: list[tuple[str, Any]] = []

        if "growthlab" in sources:
            components.append(("Growth Lab Scraper", self._run_scraper))
            components.append(("Growth Lab File Downloader", self._run_file_downloader))

        if "openalex" in sources:
            components.append(("OpenAlex Scraper", self._run_openalex_scraper))
            components.append(
                ("OpenAlex File Downloader", self._run_openalex_downloader)
            )

        # PDF processing, chunking, and embeddings always run
        components.append(("PDF Processor", self._run_pdf_processor))

        if "lectures" in sources:
            components.append(
                ("Lecture Transcripts Processor", self._run_lecture_transcripts)
            )

        components.append(("Text Chunker", self._run_text_chunker))
        components.append(("Embeddings Generator", self._run_embeddings_generator))

        for component_name, component_func in components:
            result = await self._execute_component(component_name, component_func)
            self.results.append(result)

            # Re-configure logger after each component in case a third-party
            # library (e.g. scidownl) called logger.remove() during import
            self._configure_logger()

            # Stop on critical failures (scrapers)
            if result.status == ComponentStatus.FAILED and component_name in [
                "Growth Lab Scraper",
                "OpenAlex Scraper",
            ]:
                logger.error(
                    f"Critical component {component_name} failed, stopping pipeline"
                )
                break

        # Generate final report
        self._generate_report()

        return self.results

    async def _execute_component(self, name: str, func) -> ComponentResult:
        """Execute a single component with error handling and monitoring."""
        result = ComponentResult(component_name=name, status=ComponentStatus.PENDING)

        logger.info(f"Starting component: {name}")
        result.start_time = time.time()
        result.status = ComponentStatus.RUNNING

        try:
            await func(result)
            # Only set to COMPLETED if component hasn't already set a different status
            if result.status == ComponentStatus.RUNNING:
                result.status = ComponentStatus.COMPLETED
                logger.info(f"Component {name} completed successfully")
            else:
                logger.info(
                    f"Component {name} finished with status: {result.status.value}"
                )

        except Exception as e:
            result.status = ComponentStatus.FAILED
            result.error = str(e)
            logger.error(f"Component {name} failed: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")

        finally:
            result.end_time = time.time()
            if result.duration:
                logger.info(f"Component {name} took {result.duration:.2f} seconds")

        return result

    async def _run_scraper(self, result: ComponentResult) -> None:
        """Execute the Growth Lab scraper component."""
        if self.config.skip_scraping:
            result.status = ComponentStatus.SKIPPED
            logger.info("Skipping scraping as requested")
            return

        with profile_operation("Initialize scraper", include_resources=True):
            scraper = GrowthLabScraper(
                config_path=self.config.config_path,
                concurrency_limit=self.config.scraper_concurrency,
                tracker=self.tracker,
            )

        # Pass limit directly to scraper to avoid fetching unnecessary pages
        with profile_operation("Scrape publications", include_resources=True):
            publications = await scraper.update_publications(
                storage=self.storage, limit=self.config.scraper_limit
            )

        result.metrics = {
            "publications_scraped": len(publications),
            "total_file_urls": sum(
                len(pub.file_urls) for pub in publications if pub.file_urls
            ),
        }

        # Set output files
        output_path = self.storage.get_path("intermediate/growth_lab_publications.csv")
        result.output_files = [output_path]

        logger.info(
            f"Scraped {len(publications)} publications with "
            f"{result.metrics['total_file_urls']} file URLs"
        )
        log_component_metrics("Growth Lab Scraper", result.metrics)

    def _report_file_url_analysis(
        self, publications: list, publications_with_files: list
    ) -> None:
        """Log a pre-download analysis of file URLs from scraped publications.

        Reports unique URL count, file type distribution from URL stems,
        and potential PDF count to help diagnose scraping/download issues.
        """
        # Collect all file URLs
        all_urls: list[str] = []
        for pub in publications_with_files:
            for url in pub.file_urls:
                all_urls.append(str(url))

        unique_urls = set(all_urls)
        duplicate_urls = len(all_urls) - len(unique_urls)

        # Analyze file types from URL path stems
        ext_counter: Counter[str] = Counter()
        for url in unique_urls:
            path = urlparse(url).path
            # Get the last segment of the URL path
            stem = path.rsplit("/", 1)[-1] if "/" in path else path
            # Extract extension if present
            if "." in stem:
                ext = "." + stem.rsplit(".", 1)[-1].lower()
                # Truncate very long "extensions" (likely not real extensions)
                if len(ext) > 10:
                    ext = "(no extension)"
            else:
                ext = "(no extension)"
            ext_counter[ext] += 1

        # Report
        logger.info("=" * 60)
        logger.info("PRE-DOWNLOAD FILE URL ANALYSIS")
        logger.info("=" * 60)
        logger.info(
            f"Total publications: {len(publications)} | "
            f"With file URLs: {len(publications_with_files)} "
            f"({len(publications_with_files) / max(1, len(publications)) * 100:.0f}%)"
        )
        logger.info(
            f"Total file URLs: {len(all_urls)} | "
            f"Unique: {len(unique_urls)} | "
            f"Duplicates: {duplicate_urls}"
        )
        logger.info("File type distribution (from URL path):")
        for ext, count in ext_counter.most_common():
            logger.info(f"  {ext:20s} {count:4d} URLs")
        logger.info(
            "NOTE: URL extensions are hints only. Actual file type is "
            "determined by Content-Type header after download."
        )
        logger.info("=" * 60)

    async def _run_file_downloader(self, result: ComponentResult) -> None:
        """Execute the file downloader component."""
        with profile_operation("Initialize file downloader", include_resources=True):
            downloader = FileDownloader(
                storage=self.storage,
                config_path=self.config.config_path,
                concurrency_limit=self.config.download_concurrency,
                publication_tracker=self.tracker,
            )

        # Get publications from scraper output
        pub_relative = "intermediate/growth_lab_publications.csv"
        if not self.storage.exists(pub_relative):
            raise FileNotFoundError(f"Publications file not found: {pub_relative}")
        # Ensure it's available locally for CSV loading
        publications_path = self.storage.download(pub_relative)

        # Load publications from CSV
        with profile_operation("Load publications from CSV"):
            from backend.etl.scrapers.growthlab import GrowthLabScraper

            scraper = GrowthLabScraper(config_path=self.config.config_path)
            publications = scraper.load_from_csv(publications_path)

        if not publications:
            logger.warning("No publications found in CSV file")
            result.status = ComponentStatus.SKIPPED
            return

        # Filter to publications with file URLs
        publications_with_files = [p for p in publications if p.file_urls]

        if not publications_with_files:
            logger.warning("No publications with file URLs found")
            result.status = ComponentStatus.SKIPPED
            return

        # Report file URL analysis before downloading
        self._report_file_url_analysis(publications, publications_with_files)

        # Apply download limit if specified
        publications_to_download = publications_with_files
        if self.config.download_limit:
            publications_to_download = publications_with_files[
                : self.config.download_limit
            ]
            logger.info(
                f"Limiting downloads to {len(publications_to_download)} "
                f"publications (from {len(publications_with_files)} total)"
            )

        # Download files
        with profile_operation("Download files", include_resources=True):
            download_results = await downloader.download_publications(
                publications_to_download,
                overwrite=self.config.overwrite_files,
                limit=self.config.download_limit,
                progress_bar=True,
            )

        # Calculate metrics
        successful_downloads = sum(
            1 for r in download_results if r.get("success", False)
        )
        total_size = sum(r.get("file_size", 0) or 0 for r in download_results)

        result.metrics = {
            "total_downloads_attempted": len(download_results),
            "successful_downloads": successful_downloads,
            "failed_downloads": len(download_results) - successful_downloads,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "publications_processed": len(publications_to_download),
        }

        logger.info(
            f"Downloaded {successful_downloads}/{len(download_results)} files "
            f"({total_size / (1024 * 1024):.2f} MB) from "
            f"{len(publications_to_download)} publications"
        )
        log_component_metrics("File Downloader", result.metrics)

    async def _run_openalex_scraper(self, result: ComponentResult) -> None:
        """Execute the OpenAlex scraper component."""
        if self.config.skip_scraping:
            result.status = ComponentStatus.SKIPPED
            logger.info("Skipping OpenAlex scraping as requested")
            return

        from backend.etl.scrapers.openalex import OpenAlexClient

        with profile_operation("Initialize OpenAlex scraper", include_resources=True):
            client = OpenAlexClient(config_path=self.config.config_path)

        with profile_operation("Fetch OpenAlex publications", include_resources=True):
            publications = await client.update_publications(
                storage=self.storage,
            )

        result.metrics = {
            "publications_scraped": len(publications),
        }

        output_path = self.storage.get_path("intermediate/openalex_publications.csv")
        result.output_files = [output_path]

        logger.info(f"Scraped {len(publications)} OpenAlex publications")
        log_component_metrics("OpenAlex Scraper", result.metrics)

    async def _run_openalex_downloader(self, result: ComponentResult) -> None:
        """Execute the OpenAlex file downloader component."""
        from backend.etl.utils.oa_file_downloader import (
            OpenAlexFileDownloader,
        )

        pub_relative = "intermediate/openalex_publications.csv"
        if not self.storage.exists(pub_relative):
            raise FileNotFoundError(
                f"OpenAlex publications file not found: {pub_relative}"
            )
        publications_path = self.storage.download(pub_relative)

        with profile_operation(
            "Initialize OpenAlex downloader", include_resources=True
        ):
            from backend.etl.scrapers.openalex import OpenAlexClient

            client = OpenAlexClient(config_path=self.config.config_path)
            publications = client.load_from_csv(publications_path)

        if not publications:
            logger.warning("No OpenAlex publications found in CSV")
            result.status = ComponentStatus.SKIPPED
            return

        # Apply download limit
        pubs_to_download = publications
        if self.config.download_limit:
            pubs_to_download = publications[: self.config.download_limit]
            logger.info(
                f"Limiting OpenAlex downloads to {len(pubs_to_download)} "
                f"publications (from {len(publications)} total)"
            )

        with profile_operation(
            "Initialize OpenAlex file downloader", include_resources=True
        ):
            downloader = OpenAlexFileDownloader(
                storage=self.storage,
                concurrency_limit=self.config.download_concurrency,
                config_path=self.config.config_path,
            )

        with profile_operation("Download OpenAlex files", include_resources=True):
            download_results = await downloader.download_publications(
                pubs_to_download,
                overwrite=self.config.overwrite_files,
                limit=self.config.download_limit,
                progress_bar=True,
            )

        successful = sum(1 for r in download_results if r.get("success", False))
        total_size = sum(r.get("file_size", 0) or 0 for r in download_results)
        oa_downloads = sum(1 for r in download_results if r.get("open_access", False))

        result.metrics = {
            "total_downloads_attempted": len(download_results),
            "successful_downloads": successful,
            "open_access_downloads": oa_downloads,
            "failed_downloads": len(download_results) - successful,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
        }

        logger.info(
            f"Downloaded {successful}/{len(download_results)} OpenAlex files "
            f"({oa_downloads} via open access, "
            f"{total_size / (1024 * 1024):.2f} MB)"
        )
        log_component_metrics("OpenAlex File Downloader", result.metrics)

    async def _run_pdf_processor(self, result: ComponentResult) -> None:
        """Execute the PDF processor component."""
        with profile_operation("Initialize PDF processor", include_resources=True):
            processor = PDFProcessor(
                storage=self.storage,
                config_path=self.config.config_path,
                tracker=self.tracker,
            )

        # Find PDFs across all active sources that produce PDFs
        pdf_sources = [s for s in self.config.sources if s in ("growthlab", "openalex")]
        with profile_operation("Find PDF files"):
            pdf_files: list[Path] = []
            for source in pdf_sources:
                pdf_files.extend(find_pdfs(self.storage, source=source))

        if not pdf_files:
            logger.warning("No PDF files found, skipping PDF processing")
            result.status = ComponentStatus.SKIPPED
            return

        # Process each PDF file
        with profile_operation("Process all PDFs", include_resources=True):
            processed_files = []
            for pdf_path in pdf_files:
                try:
                    output_path = processor.process_pdf(
                        pdf_path, force_reprocess=self.config.force_reprocess
                    )
                    if output_path:
                        processed_files.append(
                            {"output_path": output_path, "success": True}
                        )
                    else:
                        processed_files.append({"output_path": None, "success": False})
                except Exception as e:
                    logger.error(f"Error processing PDF {pdf_path}: {e}")
                    processed_files.append({"output_path": None, "success": False})

        # Release PDF backend models and GPU memory before next stage
        processor._backend.cleanup()

        result.metrics = {
            "files_processed": len(processed_files),
            "successful_extractions": sum(
                1 for pf in processed_files if pf.get("success", False)
            ),
            "failed_extractions": sum(
                1 for pf in processed_files if not pf.get("success", False)
            ),
        }

        result.output_files = [
            Path(pf["output_path"]) for pf in processed_files if pf.get("output_path")
        ]

        logger.info(f"Processed {len(processed_files)} PDF files")
        log_component_metrics("PDF Processor", result.metrics)

    async def _run_lecture_transcripts(self, result: ComponentResult) -> None:
        """Execute the lecture transcripts processor component.

        Processes raw lecture transcript .txt files through the OpenAI
        cleaning and metadata extraction pipeline, then copies the cleaned
        transcript text into ``processed/documents/lecture_transcripts/`` so
        the downstream text chunker and embeddings generator pick it up.
        """
        from backend.etl.scripts.run_lecture_transcripts import (
            process_single_transcript,
        )

        transcripts_input = self.config.transcripts_input or self.storage.get_path(
            "raw/lecture_transcripts"
        )

        if not self.storage.exists("raw/lecture_transcripts"):
            logger.warning(
                "No lecture transcripts directory found, skipping transcript processing"
            )
            result.status = ComponentStatus.SKIPPED
            return

        # Find raw transcript files
        transcript_files = sorted(Path(transcripts_input).glob("*.txt"))
        if not transcript_files:
            logger.warning("No transcript .txt files found, skipping")
            result.status = ComponentStatus.SKIPPED
            return

        # Apply limit
        if self.config.transcripts_limit:
            transcript_files = transcript_files[: self.config.transcripts_limit]
            logger.info(f"Limiting to {len(transcript_files)} transcripts (limit set)")

        logger.info(f"Found {len(transcript_files)} lecture transcripts to process")

        # Output directories
        output_dir = str(self.storage.get_path("processed/lecture_transcripts"))
        intermediate_dir = str(
            self.storage.get_path("intermediate/lecture_transcripts")
        )
        # Also write cleaned text to processed/documents/ for chunker
        docs_output = self.storage.get_path("processed/documents/lecture_transcripts")
        self.storage.ensure_dir(docs_output)

        successful = 0
        for transcript_path in transcript_files:
            try:
                ok = process_single_transcript(
                    transcript_path,
                    output_dir,
                    intermediate_dir,
                    max_tokens=self.config.max_tokens,
                )
                if ok:
                    successful += 1
                    # Copy cleaned text to processed/documents/ so chunker finds it
                    stem = transcript_path.stem
                    cleaned_path = (
                        Path(intermediate_dir) / f"lecture_{stem}_cleaned.txt"
                    )
                    # Fallback: try extracting lecture number like the script does
                    if not cleaned_path.exists():
                        try:
                            num = int("".join(filter(str.isdigit, stem)))
                            cleaned_path = (
                                Path(intermediate_dir)
                                / f"lecture_{num:02d}_cleaned.txt"
                            )
                        except (ValueError, TypeError):
                            pass
                    if cleaned_path.exists():
                        dest = docs_output / f"{stem}.txt"
                        dest.write_text(cleaned_path.read_text(encoding="utf-8"))
                        logger.debug(f"Copied cleaned transcript to {dest}")
            except Exception as e:
                logger.error(f"Error processing transcript {transcript_path.name}: {e}")

        result.metrics = {
            "transcripts_found": len(transcript_files),
            "transcripts_processed": successful,
            "transcripts_failed": len(transcript_files) - successful,
        }

        logger.info(
            f"Processed {successful}/{len(transcript_files)} lecture transcripts"
        )
        log_component_metrics("Lecture Transcripts", result.metrics)

    async def _run_text_chunker(self, result: ComponentResult) -> None:
        """Execute the text chunker component."""
        # Check if chunking is enabled in config
        chunking_config = self.etl_config.get("file_processing", {}).get("chunking", {})
        if not chunking_config.get("enabled", True):
            logger.info("Text chunking is disabled in configuration")
            result.status = ComponentStatus.SKIPPED
            return

        # Check if we have processed text files to chunk
        if not self.storage.exists("processed/documents"):
            logger.warning("No processed documents found, skipping text chunking")
            result.status = ComponentStatus.SKIPPED
            return

        # Find processed text files via storage glob
        with profile_operation("Find text files"):
            text_relatives = self.storage.glob("processed/documents/**/*.txt")

        if not text_relatives:
            logger.warning("No processed text files found, skipping text chunking")
            result.status = ComponentStatus.SKIPPED
            return

        # Initialize text chunker
        with profile_operation("Initialize text chunker", include_resources=True):
            chunker = TextChunker(
                config_path=self.config.config_path, tracker=self.tracker
            )

        # Process all documents
        try:
            with profile_operation("Chunk all documents", include_resources=True):
                chunking_results = chunker.process_all_documents(storage=self.storage)

            # Calculate metrics
            successful_chunks = sum(
                r.total_chunks for r in chunking_results if r.status.value == "success"
            )
            failed_documents = sum(
                1 for r in chunking_results if r.status.value == "failed"
            )
            total_processing_time = sum(r.processing_time for r in chunking_results)

            result.metrics = {
                "documents_processed": len(chunking_results),
                "successful_documents": len(chunking_results) - failed_documents,
                "failed_documents": failed_documents,
                "total_chunks_created": successful_chunks,
                "total_processing_time": total_processing_time,
                "average_chunks_per_document": successful_chunks
                / max(1, len(chunking_results) - failed_documents),
            }

            # Set output files - the chunks.json files created
            chunk_relatives = self.storage.glob("processed/chunks/**/chunks.json")
            result.output_files = [self.storage.get_path(r) for r in chunk_relatives]

            logger.info(
                f"Text chunking completed: {successful_chunks} chunks created from "
                f"{len(chunking_results) - failed_documents} documents"
            )
            log_component_metrics("Text Chunker", result.metrics)

            # Mark as failed only if documents were attempted but none
            # succeeded. If 0 were processed (all skipped/already chunked),
            # that's a successful resume, not a failure.
            if successful_chunks == 0 and len(chunking_results) > 0:
                result.status = ComponentStatus.FAILED
                result.error = "No documents were successfully chunked"

        except Exception as e:
            result.status = ComponentStatus.FAILED
            result.error = str(e)
            logger.error(f"Text chunking failed: {e}")
            raise

    async def _run_embeddings_generator(self, result: ComponentResult) -> None:
        """Execute the embeddings generator component."""
        # Check if embedding is enabled in config
        embedding_config = self.etl_config.get("file_processing", {}).get(
            "embedding", {}
        )
        if not embedding_config:
            logger.info("Embedding generation is not configured, skipping")
            result.status = ComponentStatus.SKIPPED
            return

        # Check if we have chunks to embed
        if not self.storage.exists("processed/chunks"):
            logger.warning("No chunks found, skipping embeddings generation")
            result.status = ComponentStatus.SKIPPED
            return

        # Find chunk files via storage glob
        with profile_operation("Find chunk files"):
            chunk_relatives = self.storage.glob("processed/chunks/**/chunks.json")

        if not chunk_relatives:
            logger.warning("No chunk files found, skipping embeddings generation")
            result.status = ComponentStatus.SKIPPED
            return

        # Initialize embeddings generator
        with profile_operation(
            "Initialize embeddings generator", include_resources=True
        ):
            generator = EmbeddingsGenerator(config_path=self.config.config_path)

        # Process all documents
        try:
            with profile_operation("Generate all embeddings", include_resources=True):
                embedding_results = await generator.process_all_documents(
                    storage=self.storage, tracker=self.tracker
                )

            # Calculate metrics
            from backend.etl.utils.embeddings_generator import (
                EmbeddingGenerationStatus,
            )

            successful_embeddings = sum(
                r.total_embeddings
                for r in embedding_results
                if r.status == EmbeddingGenerationStatus.SUCCESS
            )
            failed_documents = sum(
                1
                for r in embedding_results
                if r.status == EmbeddingGenerationStatus.FAILED
            )
            total_api_calls = sum(r.api_calls for r in embedding_results)
            total_tokens = sum(r.total_tokens for r in embedding_results)
            total_processing_time = sum(r.processing_time for r in embedding_results)

            result.metrics = {
                "documents_processed": len(embedding_results),
                "successful_documents": len(embedding_results) - failed_documents,
                "failed_documents": failed_documents,
                "total_embeddings_created": successful_embeddings,
                "total_api_calls": total_api_calls,
                "total_tokens": total_tokens,
                "total_processing_time": total_processing_time,
                "average_embeddings_per_document": successful_embeddings
                / max(1, len(embedding_results) - failed_documents),
            }

            # Set output files - the embeddings.parquet files created
            emb_relatives = self.storage.glob(
                "processed/embeddings/**/embeddings.parquet"
            )
            result.output_files = [self.storage.get_path(r) for r in emb_relatives]

            logger.info(
                f"Embeddings generation completed: {successful_embeddings} embeddings "
                f"created from {len(embedding_results) - failed_documents} documents "
                f"({total_api_calls} API calls)"
            )
            log_component_metrics("Embeddings Generator", result.metrics)

            # Mark as failed only if documents were attempted but none
            # succeeded. If 0 were processed (all skipped/already
            # embedded), that's a successful resume, not a failure.
            if successful_embeddings == 0 and len(embedding_results) > 0:
                result.status = ComponentStatus.FAILED
                result.error = "No embeddings were successfully generated"

        except Exception as e:
            result.status = ComponentStatus.FAILED
            result.error = str(e)
            logger.error(f"Embeddings generation failed: {e}")
            raise
        finally:
            generator.cleanup()

    async def _simulate_pipeline(self) -> list[ComponentResult]:
        """Simulate pipeline execution for dry run mode."""
        sources = self.config.sources
        components: list[str] = []
        if "growthlab" in sources:
            components += ["Growth Lab Scraper", "Growth Lab File Downloader"]
        if "openalex" in sources:
            components += ["OpenAlex Scraper", "OpenAlex File Downloader"]
        components.append("PDF Processor")
        if "lectures" in sources:
            components.append("Lecture Transcripts Processor")
        components += ["Text Chunker", "Embeddings Generator"]

        results = []
        for component in components:
            result = ComponentResult(
                component_name=component,
                status=ComponentStatus.COMPLETED,
                start_time=time.time(),
                end_time=time.time() + 1.0,
                metrics={"simulated": True},
            )
            results.append(result)
            logger.info(f"[DRY RUN] Would execute: {component}")

        return results

    def _generate_report(self) -> None:
        """Generate a final execution report."""
        logger.info("=" * 60)
        logger.info("ETL PIPELINE EXECUTION REPORT")
        logger.info("=" * 60)

        total_duration = 0.0
        for result in self.results:
            status_color = {
                ComponentStatus.COMPLETED: "green",
                ComponentStatus.FAILED: "red",
                ComponentStatus.SKIPPED: "yellow",
            }.get(result.status, "white")

            duration_str = f"{result.duration:.2f}s" if result.duration else "N/A"
            logger.info(
                f"{result.component_name}: {result.status.value} ({duration_str})"
            )

            if result.metrics:
                for key, value in result.metrics.items():
                    logger.info(f"  {key}: {value}")

            if result.error:
                logger.error(f"  Error: {result.error}")

            if result.duration:
                total_duration += result.duration

        logger.info(f"Total execution time: {total_duration:.2f}s")

        # Save detailed report
        report_path = self.storage.get_path("reports/etl_execution_report.json")
        self.storage.ensure_dir(report_path.parent)

        report_data = {
            "timestamp": int(time.time()),
            "total_duration": total_duration,
            "components": [
                {
                    "name": r.component_name,
                    "status": r.status.value,
                    "duration": r.duration,
                    "error": r.error,
                    "metrics": r.metrics,
                    "output_files": [str(f) for f in r.output_files],
                }
                for r in self.results
            ],
        }

        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Detailed report saved to: {report_path}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="ETL Pipeline Orchestrator for Growth Lab Deep Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (Growth Lab only, default)
  python -m backend.etl.orchestrator

  # Run OpenAlex pipeline end-to-end in dev mode
  python -m backend.etl.orchestrator --dev --sources openalex --download-limit 5

  # Run all sources
  python -m backend.etl.orchestrator --sources all

  # Run OpenAlex + lectures, skip scraping (use cached CSV)
  python -m backend.etl.orchestrator --sources openalex lectures --skip-scraping

  # Dry run to preview actions
  python -m backend.etl.orchestrator --dry-run --sources all

  # Lightweight dev mode (small embedding model + fast PDF extraction)
  python -m backend.etl.orchestrator --dev --skip-scraping --download-limit 3
        """,
    )

    # Global parameters
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("backend/etl/config.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--storage-type",
        choices=["local", "cloud"],
        help="Storage backend (local or cloud)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview actions without execution"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help=(
            "Enable dev mode: merges config.dev.yaml overrides on top of "
            "the base config for lightweight local iteration "
            "(small embedding model + fast PDF extraction)"
        ),
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["growthlab"],
        help=(
            "Data sources to include: growthlab, openalex, lectures, or all. "
            "Multiple can be specified. (default: growthlab)"
        ),
    )

    # Scraper parameters
    parser.add_argument(
        "--skip-scraping",
        action="store_true",
        help="Skip scraping and use existing publication data",
    )
    parser.add_argument(
        "--scraper-concurrency",
        type=int,
        default=2,
        help="Concurrent requests limit for scraper",
    )
    parser.add_argument(
        "--scraper-delay",
        type=float,
        default=2.0,
        help="Delay between scraper requests in seconds",
    )
    parser.add_argument(
        "--scraper-limit",
        type=int,
        help="Maximum number of publications to scrape (for testing)",
    )

    # File downloader parameters
    parser.add_argument(
        "--download-concurrency", type=int, default=3, help="Concurrent downloads"
    )
    parser.add_argument(
        "--download-limit", type=int, help="Maximum files to download (for testing)"
    )
    parser.add_argument(
        "--overwrite-files",
        action="store_true",
        help="Overwrite existing downloaded files",
    )
    parser.add_argument(
        "--min-file-size",
        type=int,
        default=1024,
        help="Minimum file size threshold in bytes",
    )
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=100_000_000,
        help="Maximum file size threshold in bytes",
    )

    # PDF processor parameters
    parser.add_argument(
        "--force-reprocess", action="store_true", help="Reprocess existing PDF files"
    )
    parser.add_argument(
        "--ocr-language", nargs="+", default=["eng"], help="OCR language codes"
    )
    parser.add_argument(
        "--extract-images", action="store_true", help="Extract images from PDFs"
    )
    parser.add_argument(
        "--min-chars-per-page",
        type=int,
        default=100,
        help="Minimum characters for valid extraction",
    )

    # Lecture transcripts parameters
    parser.add_argument(
        "--transcripts-input", type=Path, help="Input directory for raw transcripts"
    )
    parser.add_argument(
        "--transcripts-limit", type=int, help="Limit number of transcripts to process"
    )
    parser.add_argument(
        "--max-tokens", type=int, help="Token limit for transcript processing"
    )

    return parser


async def main() -> None:
    """Main entry point for the orchestrator."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Expand "all" in sources list
    sources = args.sources
    if "all" in sources:
        sources = sorted(VALID_SOURCES)
    else:
        invalid = set(sources) - VALID_SOURCES
        if invalid:
            parser.error(
                f"Invalid source(s): {', '.join(invalid)}. "
                f"Valid: {', '.join(sorted(VALID_SOURCES))}, all"
            )

    # Create configuration from arguments
    config = OrchestrationConfig(
        config_path=args.config,
        storage_type=args.storage_type,
        log_level=args.log_level,
        dry_run=args.dry_run,
        dev_mode=args.dev,
        sources=sources,
        skip_scraping=args.skip_scraping,
        scraper_concurrency=args.scraper_concurrency,
        scraper_delay=args.scraper_delay,
        scraper_limit=args.scraper_limit,
        download_concurrency=args.download_concurrency,
        download_limit=args.download_limit,
        overwrite_files=args.overwrite_files,
        min_file_size=args.min_file_size,
        max_file_size=args.max_file_size,
        force_reprocess=args.force_reprocess,
        ocr_language=args.ocr_language,
        extract_images=args.extract_images,
        min_chars_per_page=args.min_chars_per_page,
        transcripts_input=args.transcripts_input,
        transcripts_limit=args.transcripts_limit,
        max_tokens=args.max_tokens,
    )

    # Run orchestration
    orchestrator = ETLOrchestrator(config)
    try:
        results = await orchestrator.run_pipeline()
    finally:
        # Clean up resolved temp config if dev mode created one
        tmp_cfg = orchestrator._resolved_config_path
        if tmp_cfg and tmp_cfg.exists():
            tmp_cfg.unlink()

    # Exit with appropriate code
    failed_components = [r for r in results if r.status == ComponentStatus.FAILED]
    if failed_components:
        logger.error(
            f"Pipeline completed with {len(failed_components)} failed components"
        )
        sys.exit(1)
    else:
        logger.info("Pipeline completed successfully")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
