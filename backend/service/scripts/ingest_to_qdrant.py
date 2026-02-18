#!/usr/bin/env python
"""
Ingest embeddings from Parquet files into Qdrant.

Reads embeddings + chunk metadata from the ETL output directories and
publication metadata from the tracker DB, then upserts everything into
the ``gl_chunks`` Qdrant collection.

Usage:
    uv run python -m backend.service.scripts.ingest_to_qdrant
"""

import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path

import pyarrow.parquet as pq
from fastembed import SparseTextEmbedding
from loguru import logger
from qdrant_client import models
from sqlmodel import Session, select

from backend.etl.models.tracking import PublicationTracking
from backend.service.config import ServiceSettings, get_settings
from backend.service.qdrant_service import QdrantService
from backend.storage.database import engine

# ---------------------------------------------------------------------------
# Logging setup (mirrors run_embeddings_direct.py)
# ---------------------------------------------------------------------------


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


def setup_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

EMBEDDINGS_ROOT = Path("data/processed/embeddings/documents/growthlab")


def deterministic_uuid(chunk_id: str) -> str:
    """Generate a deterministic UUID5 from a chunk_id."""
    namespace = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
    return str(uuid.uuid5(namespace, chunk_id))


def discover_documents(root: Path) -> list[Path]:
    """Return document dirs that have both parquet and metadata."""
    docs: list[Path] = []
    if not root.exists():
        logger.warning(f"Embeddings root not found: {root}")
        return docs
    for d in sorted(root.iterdir()):
        has_parquet = (d / "embeddings.parquet").exists()
        has_meta = (d / "metadata.json").exists()
        if d.is_dir() and has_parquet and has_meta:
            docs.append(d)
    return docs


def load_publication_metadata(doc_ids: list[str]) -> dict[str, PublicationTracking]:
    """Load publication records from the tracker DB."""
    lookup: dict[str, PublicationTracking] = {}
    with Session(engine) as sess:
        stmt = select(PublicationTracking).where(
            PublicationTracking.publication_id.in_(doc_ids)  # type: ignore[attr-defined]
        )
        for pub in sess.exec(stmt).all():
            sess.expunge(pub)
            lookup[pub.publication_id] = pub
    return lookup


def build_points_for_document(
    doc_dir: Path,
    pub: PublicationTracking | None,
    sparse_model: SparseTextEmbedding,
) -> list[models.PointStruct]:
    """Build Qdrant PointStruct objects with dense + BM25 sparse vectors."""
    # Read parquet
    table = pq.read_table(doc_dir / "embeddings.parquet")
    chunk_ids: list[str] = table.column("chunk_id").to_pylist()
    embeddings: list[list[float]] = table.column("embedding").to_pylist()

    # Read metadata.json
    with open(doc_dir / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)

    document_id = meta["document_id"]

    # Build chunk_id -> chunk metadata lookup
    chunk_lookup: dict[str, dict] = {}
    for chunk in meta.get("chunks", []):
        chunk_lookup[chunk["chunk_id"]] = chunk

    # Collect text content for BM25 sparse embedding
    texts: list[str] = []
    for cid in chunk_ids:
        chunk_meta = chunk_lookup.get(cid, {})
        texts.append(chunk_meta.get("text_content", ""))

    # Generate BM25 sparse vectors for all chunks in this document
    sparse_vecs = list(sparse_model.embed(texts))

    points: list[models.PointStruct] = []
    for cid, dense_vec, sparse_vec in zip(
        chunk_ids, embeddings, sparse_vecs, strict=False
    ):
        chunk_meta = chunk_lookup.get(cid, {})
        payload: dict = {
            "chunk_id": cid,
            "document_id": document_id,
            "text_content": chunk_meta.get("text_content", ""),
            "page_numbers": chunk_meta.get("page_numbers", []),
            "section_title": chunk_meta.get("section_title"),
            "chunk_index": chunk_meta.get("chunk_index", 0),
            "token_count": chunk_meta.get("token_count", 0),
        }
        # Attach publication metadata if available
        if pub:
            payload["document_title"] = pub.title
            try:
                payload["document_authors"] = pub.authors
            except (json.JSONDecodeError, TypeError):
                # Old records may store authors as plain string
                raw = pub.authors_json or ""
                payload["document_authors"] = [raw] if raw else []
            payload["document_year"] = pub.year
            payload["document_abstract"] = pub.abstract
            payload["document_url"] = pub.source_url
        else:
            payload["document_title"] = None
            payload["document_authors"] = []
            payload["document_year"] = None
            payload["document_abstract"] = None
            payload["document_url"] = None

        points.append(
            models.PointStruct(
                id=deterministic_uuid(cid),
                vector={
                    "dense": dense_vec,
                    "bm25": models.SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist(),
                    ),
                },
                payload=payload,
            )
        )
    return points


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------


async def ingest(settings: ServiceSettings) -> int:
    """Run the full ingestion pipeline. Returns 0 on success, 1 on failure."""
    qdrant = QdrantService(settings)
    await qdrant.connect()

    try:
        # Delete old collection (schema changed: unnamed -> named vectors + BM25)
        collection = settings.qdrant_collection
        if await qdrant.client.collection_exists(collection):
            logger.info(f"Deleting old collection '{collection}' for re-creation")
            await qdrant.client.delete_collection(collection)

        # Create collection with named dense + sparse vectors
        await qdrant.ensure_collection(
            name=collection,
            vector_size=settings.embedding_dimensions,
        )

        # Load BM25 sparse model
        logger.info("Loading BM25 sparse embedding model...")
        sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

        # Discover documents
        doc_dirs = discover_documents(EMBEDDINGS_ROOT)
        if not doc_dirs:
            logger.error("No documents found with embeddings")
            return 1

        doc_ids = [d.name for d in doc_dirs]
        logger.info(f"Found {len(doc_dirs)} documents: {doc_ids}")

        # Load publication metadata in bulk
        pub_lookup = load_publication_metadata(doc_ids)
        matched = len(pub_lookup)
        total = len(doc_ids)
        logger.info(f"Loaded publication metadata for {matched}/{total} docs")

        # Build and upsert points (now with BM25 sparse vectors)
        all_points: list[models.PointStruct] = []
        for doc_dir in doc_dirs:
            doc_id = doc_dir.name
            pub = pub_lookup.get(doc_id)
            if not pub:
                logger.warning(f"No publication metadata for {doc_id}")
            points = build_points_for_document(doc_dir, pub, sparse_model)
            all_points.extend(points)
            logger.info(f"  {doc_id}: {len(points)} points")

        logger.info(f"Total points to upsert: {len(all_points)}")
        await qdrant.upsert_points(collection, all_points)

        # Verify
        info = await qdrant.collection_info()
        logger.info(f"Collection '{collection}': {info.points_count} points")

        return 0

    finally:
        await qdrant.close()


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    exit_code = asyncio.run(ingest(settings))
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
