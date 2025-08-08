"""
Qdrant Cloud Upload System for Growth Lab Deep Search

This module reads embedding files (embeddings.json) and uploads them to Qdrant Cloud
as points in collections. Each embedding becomes a point with metadata payload.
"""

import json
import os
import uuid
from pathlib import Path
from typing import Any

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Constants for text-embedding-3-small model
VECTOR_DIMENSION = 1536  # text-embedding-3-small produces 1536-dimensional vectors
DEFAULT_COLLECTION_NAME = "gl_deep_search"
BATCH_SIZE = 100  # Upload points in batches

# Load Qdrant credentials from environment variables
api_key = os.getenv("QDRANT_API_KEY")
url = os.getenv("QDRANT_URL")


def create_qdrant_client(api_key: str, url: str) -> QdrantClient | None:
    """Create and return a Qdrant Cloud client."""
    if QdrantClient is None:
        logger.error("qdrant-client package not installed.")
        return None

    try:
        client = QdrantClient(
            url=url,
            api_key=api_key,
        )
        # Test connection
        collections = client.get_collections()
        logger.info(
            f"""Successfully connected to Qdrant Cloud.\n
            Found {len(collections.collections)} collections."""
        )
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant Cloud: {e}")
        return None


def create_collection_if_not_exists(client: QdrantClient, collection_name: str) -> bool:
    """Create a collection if it doesn't exist."""
    try:
        # Check if collection exists
        collections = client.get_collections()
        existing_names = [col.name for col in collections.collections]

        if collection_name in existing_names:
            logger.info(f"Collection '{collection_name}' already exists.")
            return True

        # Create new collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=VECTOR_DIMENSION,
                distance=Distance.COSINE,  # Cosine similarity for text embeddings
            ),
        )
        logger.info(
            f"Created collection '{collection_name}' with {VECTOR_DIMENSION} dims."
        )
        return True

    except Exception as e:
        logger.error(f"Failed to create collection '{collection_name}': {e}")
        return False


def load_embeddings(embeddings_path: Path) -> dict[str, Any]:
    """Load embeddings from an embeddings.json file."""
    try:
        with open(embeddings_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load embeddings from {embeddings_path}: {e}")
        return {}


def convert_embeddings_to_points(embeddings_data: dict[str, Any]) -> list[PointStruct]:
    """Convert embeddings data to Qdrant points."""
    points = []

    document_id = embeddings_data.get("document_id", "unknown")
    source_path = embeddings_data.get("source_path", "")
    embeddings = embeddings_data.get("embeddings", [])

    for embedding_info in embeddings:
        # Generate unique ID for this point
        point_id = str(uuid.uuid4())

        # Extract vector (the actual embedding)
        vector = embedding_info.get("embedding", [])
        if not vector or len(vector) != VECTOR_DIMENSION:
            chunk_id = embedding_info.get("chunk_id", "unknown")
            logger.warning(
                f"""Invalid embedding dimension for chunk {chunk_id}.
                Expected {VECTOR_DIMENSION}, got {len(vector)}"""
            )
            continue

        # Create payload with all metadata
        payload = {
            "document_id": document_id,
            "source_path": source_path,
            "chunk_id": embedding_info.get("chunk_id"),
            "chunk_index": embedding_info.get("chunk_index"),
            "character_start": embedding_info.get("character_start"),
            "character_end": embedding_info.get("character_end"),
            "page_numbers": embedding_info.get("page_numbers", []),
            "section_title": embedding_info.get("section_title"),
            "metadata": embedding_info.get("metadata", {}),
        }

        # Create point
        point = PointStruct(id=point_id, vector=vector, payload=payload)
        points.append(point)

    logger.info(
        f"Converted {len(points)} embeddings to points for document {document_id}"
    )
    return points


def upload_points_batch(
    client: QdrantClient, collection_name: str, points: list[PointStruct]
) -> bool:
    """Upload a batch of points to Qdrant collection."""
    try:
        client.upsert(collection_name=collection_name, points=points)
        logger.info(f"Uploaded batch of {len(points)} points to '{collection_name}'")
        return True
    except Exception as e:
        logger.error(f"Failed to upload batch of {len(points)} points: {e}")
        return False


def upload_document_embeddings(
    embeddings_path: Path,
    client: QdrantClient,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> bool:
    """Upload embeddings from a single document to Qdrant."""
    logger.info(f"Uploading embeddings from {embeddings_path}")

    # Load embeddings data
    embeddings_data = load_embeddings(embeddings_path)
    if not embeddings_data:
        return False

    # Convert to points
    points = convert_embeddings_to_points(embeddings_data)
    if not points:
        logger.warning(f"No valid points to upload from {embeddings_path}")
        return False

    # Upload in batches
    success_count = 0
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i : i + BATCH_SIZE]
        if upload_points_batch(client, collection_name, batch):
            success_count += len(batch)
        else:
            logger.error(f"Failed to upload batch {i // BATCH_SIZE + 1}")

    logger.info(f"Uploaded {success_count}/{len(points)} points from {embeddings_path}")
    return success_count == len(points)


def upload_all_embeddings(
    embeddings_dir: Path,
    qdrant_url: str,
    qdrant_api_key: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> None:
    """Upload all embeddings.json files in a directory to Qdrant."""
    logger.info(f"Starting batch upload to Qdrant collection '{collection_name}'")

    # Create Qdrant client
    client = create_qdrant_client(qdrant_api_key, qdrant_url)
    if not client:
        return

    # Create collection if needed
    if not create_collection_if_not_exists(client, collection_name):
        return

    # Find all embeddings files
    embeddings_files = list(embeddings_dir.rglob("embeddings.json"))
    logger.info(f"Found {len(embeddings_files)} embeddings.json files")

    # Upload each file
    successful_uploads = 0
    for embeddings_path in embeddings_files:
        if upload_document_embeddings(embeddings_path, client, collection_name):
            successful_uploads += 1
        else:
            logger.error(f"Failed to upload {embeddings_path}")

    ratio = successful_uploads / len(embeddings_files) if embeddings_files else 0

    logger.info(f"Upload complete: {ratio} documents uploaded successfully")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Upload embeddings to Qdrant Cloud.")
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        required=True,
        help="Directory containing embeddings.json files.",
    )
    parser.add_argument(
        "--qdrant_url",
        type=str,
        default=os.getenv("QDRANT_URL"),
        help="Qdrant Cloud cluster URL.",
    )
    parser.add_argument(
        "--qdrant_api_key",
        type=str,
        default=os.getenv("QDRANT_API_KEY"),
        help="Qdrant Cloud API key.",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help="Name of the Qdrant collection.",
    )

    args = parser.parse_args()

    if not args.qdrant_url:
        logger.error("Set QDRANT_URL environment variable or use --qdrant_url.")
        return

    if not args.qdrant_api_key:
        logger.error("Set QDRANT_API_KEY environment variable or use --qdrant_api_key.")
        return

    upload_all_embeddings(
        Path(args.embeddings_dir),
        args.qdrant_url,
        args.qdrant_api_key,
        args.collection_name,
    )


if __name__ == "__main__":
    main()
