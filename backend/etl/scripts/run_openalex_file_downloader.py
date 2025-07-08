"""
Script to download files from OpenAlex publication DOIs.

This script will:
1. Load publications data from a CSV file
2. Try to find open access versions of the papers
3. Fall back to scidownl for downloading non-open access papers
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from loguru import logger

from backend.etl.utils.oa_file_downloader import download_openalex_files
from backend.storage.factory import get_storage


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download files for OpenAlex publications."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Path to the publication CSV file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Directory to save downloaded files",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit the number of publications to process (for testing)",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=3,
        help="Maximum concurrent downloads",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def setup_logging(verbose=False):
    """Set up logging configuration."""
    log_level = "DEBUG" if verbose else "INFO"

    # Configure loguru
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    # Configure standard logging to use loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            logger_opt = logger.opt(depth=6, exception=record.exc_info)
            logger_opt.log(record.levelname, record.getMessage())

    logging.basicConfig(handlers=[InterceptHandler()], level=0)


async def main():
    """Main entry point for the downloader script."""
    args = parse_args()
    setup_logging(args.verbose)

    # TEMPORARILY DISABLED: OpenAlex file downloader is being skipped due to issues
    logger.warning("OpenAlex File Downloader is temporarily disabled")
    logger.warning(
        "This component is being skipped to allow ETL orchestration "
        "development to continue"
    )
    logger.warning(
        "To re-enable, remove the skip logic in run_openalex_file_downloader.py"
    )

    # Return empty results to maintain compatibility with any calling code
    return 0

    # Get storage
    storage = get_storage()

    # Resolve paths
    input_path = Path(args.input) if args.input else None
    config_path = Path(args.config) if args.config else None

    # Log configuration
    logger.info("OpenAlex File Downloader")
    logger.info(f"Input file: {input_path or 'default'}")
    logger.info(f"Publication limit: {args.limit or 'none'}")
    logger.info(f"Concurrency: {args.concurrency}")
    logger.info(f"Overwrite: {args.overwrite}")

    try:
        # Run the downloader
        results = await download_openalex_files(
            storage=storage,
            publication_data_path=input_path,
            overwrite=args.overwrite,
            limit=args.limit,
            concurrency=args.concurrency,
            config_path=config_path,
        )

        # Count successes and failures
        successes = sum(1 for r in results if r["success"])
        oa_downloads = sum(1 for r in results if r.get("open_access", False))
        scidownl_downloads = sum(1 for r in results if r.get("source") == "scidownl")
        cached = sum(1 for r in results if r.get("cached", False))
        failures = sum(1 for r in results if not r["success"])

        # Log summary
        logger.info("=" * 50)
        logger.info("Download Summary:")
        logger.info(f"Total files processed: {len(results)}")
        logger.info(f"Successfully downloaded: {successes}")
        logger.info(f"  - Via open access: {oa_downloads}")
        logger.info(f"  - Via scidownl: {scidownl_downloads}")
        logger.info(f"Used cached files: {cached}")
        logger.info(f"Failed: {failures}")
        logger.info("=" * 50)

        return 0
    except Exception as e:
        logger.error(f"Error running downloader: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
