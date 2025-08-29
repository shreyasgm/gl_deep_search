"""
Test Script for End-to-End Embedding and Search Pipeline

This script demonstrates the complete workflow:
1. Creates chunks from lecture transcript
2. Generates embeddings using OpenAI
3. Uploads to Qdrant Cloud
4. Tests search functionality with sample queries

Run this script to test the entire pipeline with your lecture transcript data.
"""

import json
import time
import uuid
from pathlib import Path
from typing import Any

from loguru import logger
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Configuration
DEFAULT_MODEL = "text-embedding-3-small"
VECTOR_DIMENSION = 1536
COLLECTION_NAME = "test_lecture_collection"
CHUNK_SIZE = 800  # Characters per chunk
CHUNK_OVERLAP = 100  # Overlap between chunks


def create_chunks_from_transcript(
    transcript_path: Path, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[dict[str, Any]]:
    """Create text chunks from the lecture transcript."""
    logger.info(f"Loading transcript from {transcript_path}")

    with open(transcript_path, encoding="utf-8") as f:
        data = json.load(f)

    transcript_text = data.get("transcript", "")
    lecture_number = data.get("lecture_number", 0)
    title = data.get("title", "Unknown")

    if not transcript_text:
        logger.error("No transcript text found in the file")
        return []

    # Simple chunking by character count with overlap
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(transcript_text):
        end = min(start + chunk_size, len(transcript_text))

        # Try to break at sentence boundary if possible
        if end < len(transcript_text):
            # Look for sentence ending within the last 100 characters
            for i in range(end, max(end - 100, start), -1):
                if transcript_text[i] in ".!?":
                    end = i + 1
                    break

        chunk_text = transcript_text[start:end].strip()

        if len(chunk_text) < 50:  # Skip very short chunks
            break

        chunk = {
            "chunk_id": f"lecture_{lecture_number:02d}_chunk_{chunk_index:04d}",
            "text_content": chunk_text,
            "chunk_index": chunk_index,
            "character_start": start,
            "character_end": end,
            "source_document": f"lecture_{lecture_number:02d}",
            "lecture_title": title,
            "metadata": {
                "lecture_number": lecture_number,
                "title": title,
                "chunk_type": "transcript",
            },
        }

        chunks.append(chunk)
        chunk_index += 1

        # Move to next chunk with overlap
        start = max(start + chunk_size - overlap, end)

    logger.info(f"Created {len(chunks)} chunks from transcript")
    return chunks


def embed_chunks(
    chunks: list[dict[str, Any]], api_key: str, model: str = DEFAULT_MODEL
) -> list[dict[str, Any]]:
    """Generate embeddings for text chunks."""
    logger.info(f"Generating embeddings for {len(chunks)} chunks")

    client = OpenAI(api_key=api_key)
    embeddings = []

    # Process in batches of 50 for safety
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [chunk["text_content"] for chunk in batch]

        try:
            response = client.embeddings.create(
                model=model, input=texts, encoding_format="float"
            )

            # Map embeddings back to chunks
            embedding_map = {item.index: item.embedding for item in response.data}

            for j, chunk in enumerate(batch):
                embedding = embedding_map.get(j)
                if embedding:
                    embeddings.append({**chunk, "embedding": embedding})
                else:
                    logger.warning(f"No embedding for chunk {chunk['chunk_id']}")

        except Exception as e:
            logger.error(f"Failed to embed batch {i // batch_size + 1}: {e}")

    logger.info(f"Generated {len(embeddings)} embeddings successfully")
    return embeddings


def upload_to_qdrant(
    embeddings: list[dict[str, Any]],
    qdrant_url: str,
    qdrant_api_key: str,
    collection_name: str = COLLECTION_NAME,
) -> bool:
    """Upload embeddings to Qdrant collection."""
    logger.info(
        f"Uploading {len(embeddings)} embeddings to collection '{collection_name}'"
    )

    try:
        # Create client
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

        # Delete collection if it exists (for clean testing)
        try:
            client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection '{collection_name}'")
        except:
            pass  # Collection might not exist

        # Create collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=VECTOR_DIMENSION, distance=Distance.COSINE
            ),
        )
        logger.info(f"Created collection '{collection_name}'")

        # Convert to Qdrant points
        points = []
        for embedding_data in embeddings:
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding_data["embedding"],
                payload={
                    "chunk_id": embedding_data["chunk_id"],
                    "text_content": embedding_data["text_content"],
                    "chunk_index": embedding_data["chunk_index"],
                    "character_start": embedding_data["character_start"],
                    "character_end": embedding_data["character_end"],
                    "source_document": embedding_data["source_document"],
                    "lecture_title": embedding_data["lecture_title"],
                    "metadata": embedding_data["metadata"],
                },
            )
            points.append(point)

        # Upload in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            client.upsert(collection_name=collection_name, points=batch)

        logger.info("Successfully uploaded all points to Qdrant")
        return True

    except Exception as e:
        logger.error(f"Failed to upload to Qdrant: {e}")
        return False


def search_qdrant(
    query: str,
    openai_api_key: str,
    qdrant_url: str,
    qdrant_api_key: str,
    collection_name: str = COLLECTION_NAME,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Search for similar chunks in Qdrant using a text query."""
    logger.info(f"Searching for: '{query}'")

    try:
        # Generate embedding for the query
        openai_client = OpenAI(api_key=openai_api_key)
        response = openai_client.embeddings.create(
            model=DEFAULT_MODEL, input=[query], encoding_format="float"
        )
        query_vector = response.data[0].embedding
        logger.info("Generated query embedding")

        # Search in Qdrant
        qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
        )

        results = []
        for scored_point in search_result:
            result = {
                "score": scored_point.score,
                "chunk_id": scored_point.payload.get("chunk_id"),
                "text_content": scored_point.payload.get("text_content"),
                "chunk_index": scored_point.payload.get("chunk_index"),
                "lecture_title": scored_point.payload.get("lecture_title"),
                "metadata": scored_point.payload.get("metadata"),
            }
            results.append(result)

        logger.info(f"Found {len(results)} results")
        return results

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []


def main():
    """Run the complete test pipeline."""
    import os

    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    # Configuration - load from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    # Path to your transcript file
    transcript_path = Path(
        "c:/Users/sas5844/OneDrive - Harvard University/Documents/Deep Search/"
        / "gl_deep_search/data/processed/lecture_transcripts/lecture_00_processed.json"
    )

    if not openai_api_key:
        logger.error("openai_api_key not found in environment variables")
        return

    if not all([qdrant_url, qdrant_api_key]):
        logger.error("qdrant_url and qdrant_api_key not found in environment variables")
        return

    if not transcript_path.exists():
        logger.error(f"Transcript file not found: {transcript_path}")
        return

    logger.info("Starting end-to-end test pipeline")

    # Step 1: Create chunks
    logger.info("=" * 50)
    logger.info("STEP 1: Creating chunks from transcript")
    chunks = create_chunks_from_transcript(transcript_path)

    if not chunks:
        logger.error("No chunks created, stopping pipeline")
        return

    # Step 2: Generate embeddings
    logger.info("=" * 50)
    logger.info("STEP 2: Generating embeddings")
    start_time = time.time()
    embeddings = embed_chunks(chunks, openai_api_key)

    if not embeddings:
        logger.error("No embeddings generated, stopping pipeline")
        return

    # Step 3: Upload to Qdrant
    logger.info("=" * 50)
    logger.info("STEP 3: Uploading to Qdrant")
    start_time = time.time()
    upload_success = upload_to_qdrant(embeddings, qdrant_url, qdrant_api_key)

    if not upload_success:
        logger.error("Upload failed, stopping pipeline")
        return

    # Step 4: Test searches
    logger.info("=" * 50)
    logger.info("STEP 4: Testing search functionality")

    test_queries = [
        "What is economic complexity?",
        "How do growth rates differ across regions?",
        "What causes inequality in developing countries?",
        "What role do institutions play in development?",
        "Why aren't developing countries catching up?",
    ]

    # Store all search results
    search_results = {}

    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n--- Test Query {i}: {query} ---")
        start_time = time.time()
        results = search_qdrant(query, openai_api_key, qdrant_url, qdrant_api_key)
        search_time = time.time() - start_time

        logger.info(f"Search took {search_time:.3f} seconds")

        if results:
            logger.info("Top 3 results:")
            for j, result in enumerate(results[:3], 1):
                logger.info(f"  {j}. Score: {result['score']:.4f}")
                logger.info(f"     Chunk: {result['chunk_id']}")
                logger.info(f"     Text: {result['text_content'][:200]}...")
                logger.info("")

            # Store results for this query
            search_results[query] = {
                "search_time_seconds": round(search_time, 3),
                "total_results": len(results),
                "results": [
                    {
                        "rank": j + 1,
                        "score": round(result["score"], 4),
                        "chunk_id": result["chunk_id"],
                        "chunk_index": result["chunk_index"],
                        "text_content": result["text_content"],
                        "lecture_title": result["lecture_title"],
                        "metadata": result["metadata"],
                    }
                    for j, result in enumerate(results)
                ],
            }
        else:
            logger.warning("No results found")
            search_results[query] = {
                "search_time_seconds": round(search_time, 3),
                "total_results": 0,
                "results": [],
            }

    # Save search results to file
    output_dir = Path("data/processed/test_vector_search")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"search_results_{timestamp}.json"

    # Add metadata to the results
    final_results = {
        "test_metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": DEFAULT_MODEL,
            "collection_name": COLLECTION_NAME,
            "total_chunks": len(chunks),
            "total_embeddings": len(embeddings),
        },
        "queries": search_results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Search results saved to: {output_file}")

    logger.info("=" * 50)
    logger.info("Test pipeline completed successfully!")


if __name__ == "__main__":
    main()
