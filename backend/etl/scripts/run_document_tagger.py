#!/usr/bin/env python
"""
Script to run the document tagger for Growth Lab publications.

Reads processed chunks, generates structured metadata tags via LLM (using
title + abstract from the tracker, falling back to chunk sampling), and
injects the tags into each chunk's metadata for downstream Qdrant filtering.

Pipeline position: after run_text_chunker.py, before run_embeddings_generator.py.

Usage:
    uv run python backend/etl/scripts/run_document_tagger.py --config backend/etl/config.dev.yaml
    uv run python backend/etl/scripts/run_document_tagger.py --config backend/etl/config.yaml --max-docs 5
"""

import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv
from loguru import logger

from backend.etl.models.tracking import TaggingStatus
from backend.etl.utils.document_tagger import DocumentTagger
from backend.storage.factory import get_storage


class InterceptHandler(logging.Handler):
    """Redirect standard logging to loguru."""

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
    """Configure loguru for console output."""
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


async def async_main(args: argparse.Namespace) -> int:
    """Async entry point."""
    storage = get_storage(config_path=args.config, storage_type=args.storage_type)

    # Try to load the tracker; degrade gracefully if unavailable
    tracker = None
    try:
        from backend.etl.utils.publication_tracker import PublicationTracker

        tracker = PublicationTracker()
    except Exception as e:
        logger.warning(
            f"Could not initialise PublicationTracker: {e}. "
            "Continuing without tracker (tags won't be written to SQLite)."
        )

    logger.info("Starting document tagging")

    tagger = DocumentTagger(config_path=args.config, tracker=tracker)
    results = await tagger.process_all_documents(
        storage=storage,
        max_docs=args.max_docs,
        overwrite=args.overwrite,
    )

    total = len(results)
    if total == 0:
        logger.warning(
            "No documents were tagged — nothing to process or all already tagged"
        )
        return 0

    successful = sum(1 for r in results if r.status == TaggingStatus.TAGGED)
    failed = sum(1 for r in results if r.status == TaggingStatus.FAILED)
    total_chunks = sum(r.chunks_tagged for r in results)
    total_time = sum(r.processing_time for r in results)
    abstract_used = sum(1 for r in results if r.text_source == "abstract")
    chunks_used = sum(1 for r in results if r.text_source == "chunks")

    logger.info("=" * 60)
    logger.info("Document Tagging Summary")
    logger.info("=" * 60)
    logger.info(f"Documents processed:   {total}")
    logger.info(f"  Successful:          {successful}")
    logger.info(f"  Failed:              {failed}")
    logger.info(f"Total chunks tagged:   {total_chunks}")
    logger.info(
        f"Text source — abstract: {abstract_used}, chunk samples: {chunks_used}"
    )
    logger.info(f"Total processing time: {total_time:.2f}s")

    if failed > 0:
        logger.warning("Failed documents:")
        for r in results:
            if r.status == TaggingStatus.FAILED:
                logger.warning(f"  - {r.publication_id}: {r.error_message}")

    return 0 if failed == 0 else 1


def main():
    """CLI entry point."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description=(
            "Tag processed document chunks with structured metadata "
            "using an LLM for downstream Qdrant payload filtering."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="backend/etl/config.yaml",
        help="Path to configuration file (default: backend/etl/config.yaml)",
    )
    parser.add_argument(
        "--storage-type",
        type=str,
        choices=["local", "cloud"],
        help="Storage type to use",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help=(
            "Maximum number of documents to tag. "
            "Overrides file_processing.tagging.max_docs from config when provided."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Re-tag documents that have already been tagged, "
            "overwriting existing document_tags in chunks.json."
        ),
    )

    args = parser.parse_args()
    setup_logging()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Error running document tagger: {e}")
        sys.exit(1)
