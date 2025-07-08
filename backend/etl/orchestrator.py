"""
ETL Pipeline Orchestrator for Growth Lab Deep Search.

This module orchestrates the complete ETL pipeline, executing components in sequence
with proper error handling, monitoring, and data validation.
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from backend.etl.scrapers.growthlab import GrowthLabScraper
from backend.etl.utils.gl_file_downloader import FileDownloader
from backend.etl.utils.pdf_processor import PDFProcessor
from backend.storage.factory import get_storage


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

    # Scraper settings
    skip_scraping: bool = False
    scraper_concurrency: int = 2
    scraper_delay: float = 2.0

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


class ETLOrchestrator:
    """Main orchestrator for the ETL pipeline."""

    def __init__(self, config: OrchestrationConfig):
        """Initialize the orchestrator with configuration."""
        self.config = config
        self.storage = get_storage(storage_type=config.storage_type)
        self.results: list[ComponentResult] = []

        # Set up logging
        logger.remove()
        logger.add(
            sys.stdout,
            level=config.log_level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            ),
        )

        # Load ETL configuration
        self.etl_config = self._load_etl_config()

    def _load_etl_config(self) -> dict[str, Any]:
        """Load ETL configuration from YAML file."""
        try:
            with open(self.config.config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(
                f"Failed to load ETL config from {self.config.config_path}: {e}"
            )
            raise

    def _path_exists(self, path: Path) -> bool:
        """
        Check if a path exists (helper method since storage base doesn't have exists).
        """
        return path.exists()

    async def run_pipeline(self) -> list[ComponentResult]:
        """Execute the complete ETL pipeline."""
        logger.info("Starting ETL pipeline orchestration")

        if self.config.dry_run:
            logger.info("DRY RUN MODE - No actual processing will be performed")
            return await self._simulate_pipeline()

        # Execute components in sequence
        components = [
            ("Growth Lab Scraper", self._run_scraper),
            ("Growth Lab File Downloader", self._run_file_downloader),
            ("PDF Processor", self._run_pdf_processor),
            ("Lecture Transcripts Processor", self._run_lecture_transcripts),
        ]

        for component_name, component_func in components:
            result = await self._execute_component(component_name, component_func)
            self.results.append(result)

            # Stop on critical failures
            if result.status == ComponentStatus.FAILED and component_name in [
                "Growth Lab Scraper"
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
            result.status = ComponentStatus.COMPLETED
            logger.info(f"Component {name} completed successfully")

        except Exception as e:
            result.status = ComponentStatus.FAILED
            result.error = str(e)
            logger.error(f"Component {name} failed: {e}")

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

        scraper = GrowthLabScraper(
            config_path=self.config.config_path,
            concurrency_limit=self.config.scraper_concurrency,
        )

        publications = await scraper.update_publications(storage=self.storage)

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

    async def _run_file_downloader(self, result: ComponentResult) -> None:
        """Execute the file downloader component."""
        downloader = FileDownloader(
            storage=self.storage,
            config_path=self.config.config_path,
            concurrency_limit=self.config.download_concurrency,
        )

        # Get publications from scraper output
        publications_path = self.storage.get_path(
            "intermediate/growth_lab_publications.csv"
        )
        if not self._path_exists(publications_path):
            raise FileNotFoundError(f"Publications file not found: {publications_path}")

        # For now, create a placeholder implementation since the file downloader
        # methods don't exist yet. This would need to be implemented properly.
        logger.warning(
            "File downloader integration not yet implemented in orchestrator"
        )

        # Simulate download results
        download_results: list[dict] = []
        successful_downloads = 0
        total_size = 0

        result.metrics = {
            "total_downloads_attempted": len(download_results),
            "successful_downloads": successful_downloads,
            "failed_downloads": len(download_results) - successful_downloads,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
        }

        logger.info(
            f"Downloaded {successful_downloads}/{len(download_results)} files "
            f"({total_size / (1024 * 1024):.2f} MB)"
        )

    async def _run_pdf_processor(self, result: ComponentResult) -> None:
        """Execute the PDF processor component."""
        processor = PDFProcessor(
            storage=self.storage, config_path=self.config.config_path
        )

        # Find downloaded PDF files
        documents_path = self.storage.get_path("raw/documents/growthlab")
        if not self._path_exists(documents_path):
            logger.warning("No downloaded documents found, skipping PDF processing")
            result.status = ComponentStatus.SKIPPED
            return

        # Process PDFs - find all PDF files in the directory
        pdf_files = []
        for pdf_path in documents_path.rglob("*.pdf"):
            if pdf_path.is_file():
                pdf_files.append(pdf_path)

        # Process each PDF file
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

    async def _run_lecture_transcripts(self, result: ComponentResult) -> None:
        """Execute the lecture transcripts processor component."""
        # This is a placeholder - the actual implementation would need to be
        # adapted from the existing run_lecture_transcripts.py script

        transcripts_input = self.config.transcripts_input or self.storage.get_path(
            "raw/lecture_transcripts"
        )

        if not self._path_exists(transcripts_input):
            logger.warning(
                "No lecture transcripts found, skipping transcript processing"
            )
            result.status = ComponentStatus.SKIPPED
            return

        # For now, just simulate processing
        result.metrics = {
            "transcripts_processed": 0,
            "note": "Lecture transcript processing not yet implemented in orchestrator",
        }

        logger.info("Lecture transcript processing skipped - not yet implemented")
        result.status = ComponentStatus.SKIPPED

    async def _simulate_pipeline(self) -> list[ComponentResult]:
        """Simulate pipeline execution for dry run mode."""
        components = [
            "Growth Lab Scraper",
            "Growth Lab File Downloader",
            "PDF Processor",
            "Lecture Transcripts Processor",
        ]

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
  # Run full pipeline
  python -m backend.etl.orchestrator --config config.yaml

  # Run with custom parameters
  python -m backend.etl.orchestrator --config config.yaml \\
    --download-concurrency 5 --download-limit 100

  # Skip scraping and use existing data
  python -m backend.etl.orchestrator --skip-scraping --overwrite-files

  # Dry run to preview actions
  python -m backend.etl.orchestrator --dry-run
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

    # Create configuration from arguments
    config = OrchestrationConfig(
        config_path=args.config,
        storage_type=args.storage_type,
        log_level=args.log_level,
        dry_run=args.dry_run,
        skip_scraping=args.skip_scraping,
        scraper_concurrency=args.scraper_concurrency,
        scraper_delay=args.scraper_delay,
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
    results = await orchestrator.run_pipeline()

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
