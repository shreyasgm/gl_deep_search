#!/usr/bin/env python
"""
Script to run the PDF processor for Growth Lab publications.
"""

import argparse
import logging
import sys
from pathlib import Path

from loguru import logger

from backend.etl.utils.pdf_processor import process_growth_lab_pdfs
from backend.storage.factory import get_storage


def setup_logging():
    """Set up logging configuration."""
    # Remove default loggers
    logger.remove()

    # Add a new sink to stderr with better formatting
    logger.add(
        sys.stderr,
        level="INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    # Configure standard logging to use loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


class InterceptHandler(logging.Handler):
    """Handler to intercept standard logging and redirect to loguru."""

    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def main():
    """Main entry point for the PDF processor script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process PDFs from Growth Lab publications"
    )
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--storage-type",
        type=str,
        choices=["local", "cloud"],
        help="Storage type to use",
    )
    parser.add_argument(
        "--force-reprocess", action="store_true", help="Force reprocessing of all PDFs"
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable progress display"
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging()

    # Get storage
    storage = get_storage(config_path=args.config, storage_type=args.storage_type)

    # Run the PDF processor
    logger.info("Starting Growth Lab PDF processor")

    results = process_growth_lab_pdfs(
        storage=storage,
        force_reprocess=args.force_reprocess,
        config_path=Path(args.config) if args.config else None,
    )

    # Generate summary
    total = len(results)
    if total == 0:
        logger.warning("No PDF files were processed")
        return 0

    successful = sum(1 for result in results.values() if result is not None)
    success_rate = successful / total

    logger.info(
        f"PDF processing complete: {successful}/{total} successful ({success_rate:.1%})"
    )

    # Return success/failure
    return 0 if successful == total else 1


if __name__ == "__main__":
    try:
        # Run the main function
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Error running PDF processor: {e}")
        sys.exit(1)
