#!/usr/bin/env python
"""
Script to generate embeddings for all chunked documents, bypassing the
broken publication tracker. Discovers document IDs directly from the
chunks directory.
"""

import asyncio
import logging
import sys
from pathlib import Path

from loguru import logger

from backend.etl.utils.embeddings_generator import (
    EmbeddingGenerationStatus,
    EmbeddingsGenerator,
)
from backend.storage.factory import get_storage


class InterceptHandler(logging.Handler):
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


def discover_document_ids(storage) -> list[str]:
    """Find all document IDs that have chunks.json files."""
    chunks_base = storage.get_path("processed/chunks")
    doc_ids = []
    for chunks_file in chunks_base.rglob("chunks.json"):
        doc_ids.append(chunks_file.parent.name)
    return sorted(doc_ids)


async def run(config_path: str):
    storage = get_storage(config_path=config_path, storage_type="local")
    generator = EmbeddingsGenerator(config_path=Path(config_path))

    doc_ids = discover_document_ids(storage)
    logger.info(f"Found {len(doc_ids)} documents with chunks: {doc_ids}")

    results = []
    for doc_id in doc_ids:
        logger.info(f"Generating embeddings for {doc_id}")
        result = await generator.generate_embeddings_for_document(
            document_id=doc_id, storage=storage
        )
        results.append(result)
        if result.status == EmbeddingGenerationStatus.SUCCESS:
            logger.info(
                f"  OK: {result.total_embeddings} embeddings, "
                f"{result.api_calls} API calls, {result.processing_time:.1f}s"
            )
        else:
            logger.error(f"  FAILED: {result.error_message}")

    # Summary
    successful = sum(
        1 for r in results if r.status == EmbeddingGenerationStatus.SUCCESS
    )
    failed = sum(1 for r in results if r.status == EmbeddingGenerationStatus.FAILED)
    total_embeddings = sum(r.total_embeddings for r in results)

    logger.info("=" * 60)
    logger.info("Embeddings Generation Summary")
    logger.info("=" * 60)
    logger.info(f"Documents: {len(results)} ({successful} OK, {failed} failed)")
    logger.info(f"Total embeddings: {total_embeddings}")

    if failed > 0:
        for r in results:
            if r.status == EmbeddingGenerationStatus.FAILED:
                logger.error(f"  FAILED: {r.document_id}: {r.error_message}")

    return 0 if failed == 0 else 1


def main():
    setup_logging()
    config_path = "backend/etl/config.yaml"
    exit_code = asyncio.run(run(config_path))
    sys.exit(exit_code)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Error: {e}")
        sys.exit(1)
