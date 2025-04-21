#!/usr/bin/env python
"""
Script to run the file downloader for Growth Lab publications.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from loguru import logger

from backend.etl.utils.file_downloader import download_growthlab_files
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


async def main():
    """Main entry point for the file downloader script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Download files for Growth Lab publications"
    )
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--publication-data", type=str, help="Path to publication CSV data"
    )
    parser.add_argument(
        "--storage-type",
        type=str,
        choices=["local", "cloud"],
        help="Storage type to use",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of publications to process (for testing)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=3, help="Maximum concurrent downloads"
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging()

    # Get storage
    storage = get_storage(config_path=args.config, storage_type=args.storage_type)

    # Get publication data path
    publication_data_path = None
    if args.publication_data:
        publication_data_path = Path(args.publication_data)

    # Run the downloader
    logger.info("Starting Growth Lab file downloader")
    results = await download_growthlab_files(
        storage=storage,
        publication_data_path=publication_data_path,
        overwrite=args.overwrite,
        limit=args.limit,
        concurrency=args.concurrency,
        config_path=Path(args.config) if args.config else None,
    )

    # Generate summary
    successful = len([r for r in results if r["success"]])
    total = len(results)

    logger.info(f"File download complete: {successful}/{total} successful")

    # Return success/failure
    return 0 if successful == total else 1


if __name__ == "__main__":
    try:
        # Create event loop for Windows compatibility
        if sys.platform == "win32":
            loop = asyncio.ProactorEventLoop()
            asyncio.set_event_loop(loop)

        # Run the main function
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Error running file downloader: {e}")
        sys.exit(1)
