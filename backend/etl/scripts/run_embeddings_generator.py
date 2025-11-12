#!/usr/bin/env python
"""
Script to run the embeddings generator for Growth Lab publications.
"""

import argparse
import logging
import sys
from pathlib import Path

from loguru import logger

from backend.etl.utils.embeddings_generator import run_embeddings_generator
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
    """Main entry point for the embeddings generator script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate embeddings for processed documents"
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
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of documents to process",
    )
    parser.add_argument(
        "--document-id",
        type=str,
        action="append",
        dest="document_ids",
        help="Process specific document ID (can be specified multiple times)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually generating embeddings",
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging()

    # Get storage
    storage = get_storage(config_path=args.config, storage_type=args.storage_type)

    # Dry run mode
    if args.dry_run:
        from backend.etl.utils.publication_tracker import PublicationTracker

        tracker = PublicationTracker()

        if args.document_ids:
            logger.info(f"Would process {len(args.document_ids)} specific documents:")
            for doc_id in args.document_ids:
                pub = tracker.get_publication(doc_id)
                if pub:
                    logger.info(f"  - {doc_id}: {pub.title}")
                else:
                    logger.warning(f"  - {doc_id}: NOT FOUND")
        else:
            pubs = tracker.get_publications_for_embedding(limit=args.limit)
            logger.info(f"Would process {len(pubs)} documents ready for embedding")
            for pub in pubs[:10]:  # Show first 10
                logger.info(f"  - {pub.publication_id}: {pub.title}")
            if len(pubs) > 10:
                logger.info(f"  ... and {len(pubs) - 10} more")

        return 0

    # Run the embeddings generator
    logger.info("Starting embeddings generation")

    try:
        results = run_embeddings_generator(
            config_path=Path(args.config),
            storage=storage,
            limit=args.limit,
            document_ids=args.document_ids,
        )

        # Generate summary
        total = len(results)
        if total == 0:
            logger.warning("No documents were processed for embedding generation")
            return 0

        from backend.etl.utils.embeddings_generator import (
            EmbeddingGenerationStatus,
        )

        successful = sum(
            1
            for result in results
            if result.status == EmbeddingGenerationStatus.SUCCESS
        )
        failed = sum(
            1 for result in results if result.status == EmbeddingGenerationStatus.FAILED
        )

        total_embeddings = sum(result.total_embeddings for result in results)
        total_api_calls = sum(result.api_calls for result in results)
        total_time = sum(result.processing_time for result in results)

        success_rate = successful / total if total > 0 else 0

        logger.info("=" * 60)
        logger.info("Embeddings Generation Summary")
        logger.info("=" * 60)
        logger.info(f"Documents processed: {total}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Success rate: {success_rate:.1%}")
        logger.info(f"Total embeddings generated: {total_embeddings}")
        logger.info(f"Total API calls: {total_api_calls}")
        logger.info(f"Total processing time: {total_time:.2f}s")

        if failed > 0:
            logger.warning("Failed documents:")
            for result in results:
                if result.status == EmbeddingGenerationStatus.FAILED:
                    logger.warning(f"  - {result.document_id}: {result.error_message}")

        # Return success/failure
        return 0 if failed == 0 else 1

    except Exception as e:
        logger.exception(f"Error generating embeddings: {e}")
        return 1


if __name__ == "__main__":
    try:
        # Run the main function
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Error running embeddings generator: {e}")
        sys.exit(1)
