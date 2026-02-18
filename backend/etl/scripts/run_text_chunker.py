#!/usr/bin/env python
"""
Script to run the text chunker for Growth Lab publications.

Processes all documents in data/processed/documents/ and generates
token-based chunks in data/processed/chunks/.
"""

import argparse
import logging
import sys

from loguru import logger

from backend.etl.utils.text_chunker import ChunkingStatus, TextChunker
from backend.storage.factory import get_storage


class InterceptHandler(logging.Handler):
    """Handler to intercept standard logging and redirect to loguru."""

    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging():
    """Set up logging configuration."""
    logger.remove()
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
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


def main():
    """Main entry point for the text chunker script."""
    parser = argparse.ArgumentParser(
        description="Chunk processed documents into token-limited segments"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="backend/etl/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--storage-type",
        type=str,
        choices=["local", "cloud"],
        help="Storage type to use",
    )

    args = parser.parse_args()
    setup_logging()

    storage = get_storage(config_path=args.config, storage_type=args.storage_type)

    logger.info("Starting text chunking")

    try:
        chunker = TextChunker(config_path=args.config, tracker=None)
        results = chunker.process_all_documents(storage=storage)

        total = len(results)
        if total == 0:
            logger.warning("No documents were processed")
            return 0

        successful = sum(1 for r in results if r.status == ChunkingStatus.SUCCESS)
        failed = sum(1 for r in results if r.status == ChunkingStatus.FAILED)
        total_chunks = sum(r.total_chunks for r in results)
        total_time = sum(r.processing_time for r in results)

        logger.info("=" * 60)
        logger.info("Text Chunking Summary")
        logger.info("=" * 60)
        logger.info(f"Documents processed: {total}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"Total chunks generated: {total_chunks}")
        logger.info(f"Total processing time: {total_time:.2f}s")

        if failed > 0:
            logger.warning("Failed documents:")
            for r in results:
                if r.status == ChunkingStatus.FAILED:
                    logger.warning(f"  - {r.document_id}: {r.error_message}")

        return 0 if failed == 0 else 1

    except Exception as e:
        logger.exception(f"Error chunking documents: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Error running text chunker: {e}")
        sys.exit(1)
