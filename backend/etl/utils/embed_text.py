"""
Embedding System for Growth Lab Deep Search

This module reads chunked text files (chunks.json) and generates vector embeddings
using OpenAI's text-embedding-3-small model. Embeddings are saved as embeddings.json
in the same directory as the input chunks file.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

from loguru import logger
from openai import OpenAI

DEFAULT_MODEL = "text-embedding-3-small"
MAX_BATCH_SIZE = 100  # OpenAI allows up to 2048, but keep small for safety


def load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    """Load chunks from a chunks.json file."""
    with open(chunks_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["chunks"]


def save_embeddings(
    embeddings: list[dict[str, Any]], output_path: Path, metadata: dict[str, Any]
) -> None:
    """Save embeddings to a JSON file."""
    result = {
        "document_id": metadata.get("document_id"),
        "source_path": metadata.get("source_path"),
        "total_chunks": metadata.get("total_chunks"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "embeddings": embeddings,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved embeddings to {output_path}")


def get_openai_embeddings(
    texts: list[str], api_key: str, model: str = DEFAULT_MODEL
) -> list[list[float] | None]:
    """Get embeddings for a list of texts using OpenAI SDK."""
    client = OpenAI(api_key=api_key)
    try:
        response = client.embeddings.create(
            model=model, input=texts, encoding_format="float"
        )
        # The response.data is a list of objects with 'embedding' and 'index'
        # Ensure order matches input
        embedding_map = {item.index: item.embedding for item in response.data}
        return [embedding_map.get(i) for i in range(len(texts))]
    except Exception as e:
        logger.error(f"OpenAI embedding batch failed: {e}")
        return [None] * len(texts)


def embed_document_chunks(
    chunks_path: Path,
    output_path: Path,
    openai_api_key: str,
    model: str = DEFAULT_MODEL,
) -> None:
    """Embed all chunks in a single document using batch requests."""
    logger.info(f"Embedding chunks from {chunks_path}")
    chunks_data = None
    try:
        with open(chunks_path, encoding="utf-8") as f:
            chunks_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load chunks file: {e}")
        return
    chunks = chunks_data.get("chunks", [])
    metadata = {
        "document_id": chunks_data.get("document_id"),
        "source_path": chunks_data.get("source_path"),
        "total_chunks": chunks_data.get("total_chunks"),
    }
    # Prepare texts and chunk info for batching
    chunk_infos = [
        {
            "chunk_id": chunk.get("chunk_id"),
            "text": chunk.get("text_content", ""),
            "chunk_index": chunk.get("chunk_index"),
            "character_start": chunk.get("character_start"),
            "character_end": chunk.get("character_end"),
            "page_numbers": chunk.get("page_numbers"),
            "section_title": chunk.get("section_title"),
            "metadata": chunk.get("metadata"),
        }
        for chunk in chunks
        if chunk.get("text_content", "").strip()
    ]
    if not chunk_infos:
        logger.warning(f"No non-empty chunks to embed in {chunks_path}")
        save_embeddings([], output_path, metadata)
        return
    embeddings = []
    # Batch in groups of MAX_BATCH_SIZE
    for i in range(0, len(chunk_infos), MAX_BATCH_SIZE):
        batch = chunk_infos[i : i + MAX_BATCH_SIZE]
        texts = [c["text"] for c in batch]
        batch_embeddings = get_openai_embeddings(texts, openai_api_key, model)
        for info, emb in zip(batch, batch_embeddings, strict=False):
            if emb is not None:
                embeddings.append(
                    {
                        "chunk_id": info["chunk_id"],
                        "embedding": emb,
                        "chunk_index": info["chunk_index"],
                        "character_start": info["character_start"],
                        "character_end": info["character_end"],
                        "page_numbers": info["page_numbers"],
                        "section_title": info["section_title"],
                        "metadata": info["metadata"],
                    }
                )
            else:
                logger.error(f"Failed to embed chunk {info['chunk_id']}")
    save_embeddings(embeddings, output_path, metadata)


def embed_all_documents(
    chunks_dir: Path, openai_api_key: str, model: str = DEFAULT_MODEL
) -> None:
    """Batch process all chunks.json files in a directory tree."""
    logger.info(f"Batch embedding for directory: {chunks_dir}")
    chunks_files = list(chunks_dir.rglob("chunks.json"))
    logger.info(f"Found {len(chunks_files)} chunks.json files.")
    for chunks_path in chunks_files:
        output_path = chunks_path.parent / "embeddings.json"
        if output_path.exists():
            logger.info(f"Embeddings already exist for {chunks_path}, skipping.")
            continue
        embed_document_chunks(chunks_path, output_path, openai_api_key, model)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Embed text chunks using OpenAI API.")
    parser.add_argument(
        "--chunks_dir",
        type=str,
        required=True,
        help="Directory containing chunks.json files.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key.",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, help="OpenAI embedding model."
    )
    args = parser.parse_args()
    if not args.api_key:
        logger.error("OpenAI API key is required.")
        return
    embed_all_documents(Path(args.chunks_dir), args.api_key, args.model)


if __name__ == "__main__":
    main()
