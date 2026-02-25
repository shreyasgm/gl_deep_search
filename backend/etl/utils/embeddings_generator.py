"""
Embeddings Generation System for Growth Lab Deep Search.

This module generates vector embeddings from text chunks using either:
- OpenRouter API (Qwen3-Embedding-8B hosted remotely), or
- A local SentenceTransformer model (e.g. Qwen3-Embedding-8B on GPU)

Results are stored in Parquet format for efficient vector database ingestion.
"""

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from loguru import logger
from openai import AsyncOpenAI, OpenAIError, RateLimitError


class EmbeddingGenerationStatus(Enum):
    """Status of embedding generation operation."""

    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class ChunkEmbedding:
    """Represents an embedding for a single text chunk."""

    chunk_id: str  # Reference to original chunk
    embedding_vector: list[float]  # The embedding vector
    model: str  # Model used for generation
    dimensions: int  # Vector dimensionality
    created_at: datetime  # Generation timestamp

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["created_at"] = self.created_at.isoformat()
        return result


@dataclass
class EmbeddingResult:
    """Result of embedding generation for a single document."""

    document_id: str
    source_path: Path
    embeddings: list[ChunkEmbedding]
    total_embeddings: int
    processing_time: float
    api_calls: int
    total_tokens: int
    status: EmbeddingGenerationStatus
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        result = asdict(self)
        result["source_path"] = str(self.source_path)
        result["embeddings"] = [emb.to_dict() for emb in self.embeddings]
        result["status"] = self.status.value
        return result


class EmbeddingsGenerator:
    """
    Main embeddings generation system.

    Supports two providers:
    - ``openrouter``: calls OpenRouter's OpenAI-compatible API
      (default model: ``qwen/qwen3-embedding-8b``)
    - ``sentence_transformer``: local inference via SentenceTransformer

    Expected Directory Structure
    ============================
    Input (chunks from text chunker):
        processed/chunks/{content_type}/{source_type}/{doc_id}/chunks.json

        Examples:
        - processed/chunks/documents/growthlab/gl_url_123/chunks.json
        - processed/chunks/documents/openalex/oa_work_456/chunks.json

    Output (embeddings):
        processed/embeddings/{content_type}/{source_type}/{doc_id}/embeddings.parquet
        processed/embeddings/{content_type}/{source_type}/{doc_id}/metadata.json

    This structure mirrors the text chunker's output and maintains consistency
    across the ETL pipeline. The path resolution logic uses recursive glob
    patterns to handle any nesting level automatically.
    """

    def __init__(self, config_path: Path):
        """Initialize embeddings generator with configuration."""
        self.config_path = config_path
        self.config = self._load_config()

        # Load embedding configuration
        emb_config = self.config.get("file_processing", {}).get("embedding", {})

        # Defaults matching config.yaml
        defaults = {
            "model": "sentence_transformer",
            "dimensions": 1024,
            "batch_size": 32,
            "max_retries": 3,
            "retry_delays": [1, 2, 4],
            "timeout": 30,
            "rate_limit_delay": 0.1,
        }

        # Merge configuration
        merged = {**defaults, **emb_config}

        # Validate and set configuration
        self.model_provider = merged["model"]
        self.dimensions = merged["dimensions"]
        self.batch_size = merged["batch_size"]
        self.max_retries = merged["max_retries"]
        self.retry_delays = merged["retry_delays"]
        self.timeout = merged["timeout"]
        self.rate_limit_delay = merged["rate_limit_delay"]

        # Initialize embedding backend
        if self.model_provider == "openrouter":
            api_base_url = emb_config.get(
                "api_base_url", "https://openrouter.ai/api/v1"
            )
            api_key = emb_config.get("api_key", os.environ.get("EMBEDDING_API_KEY"))
            self.model_name = emb_config.get("model_name", "qwen/qwen3-embedding-8b")
            self.client = AsyncOpenAI(api_key=api_key, base_url=api_base_url)
        elif self.model_provider == "sentence_transformer":
            from sentence_transformers import SentenceTransformer

            self.model_name = emb_config.get("model_name", "Qwen/Qwen3-Embedding-8B")
            self.st_model = SentenceTransformer(self.model_name, trust_remote_code=True)
        else:
            raise ValueError(
                f"Unsupported embedding model provider: {self.model_provider}"
            )

        logger.info(
            f"EmbeddingsGenerator initialized with {self.model_provider} "
            f"(model: {self.model_name}, dims: {self.dimensions}, "
            f"batch_size: {self.batch_size})"
        )

    def cleanup(self) -> None:
        """Release the embedding model and free GPU memory."""
        from backend.etl.utils.gpu_memory import release_gpu_memory

        if hasattr(self, "st_model"):
            del self.st_model
        release_gpu_memory()
        logger.info("EmbeddingsGenerator cleaned up and GPU memory released")

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            raise

    async def generate_embeddings_for_document(
        self,
        document_id: str,
        storage,
    ) -> EmbeddingResult:
        """
        Generate embeddings for a single document's chunks.

        Args:
            document_id: Unique identifier for the document
            storage: Storage abstraction for path resolution

        Returns:
            EmbeddingResult with status and generated embeddings
        """
        start_time = time.time()

        try:
            # Load chunks from JSON
            chunks_path = self._resolve_chunks_path(document_id, storage)
            if not chunks_path or not chunks_path.exists():
                error_msg = f"Chunks file not found for {document_id}"
                logger.warning(error_msg)
                return EmbeddingResult(
                    document_id=document_id,
                    source_path=chunks_path or Path("unknown"),
                    embeddings=[],
                    total_embeddings=0,
                    processing_time=time.time() - start_time,
                    api_calls=0,
                    total_tokens=0,
                    status=EmbeddingGenerationStatus.FAILED,
                    error_message=error_msg,
                )

            with open(chunks_path, encoding="utf-8") as f:
                chunks_file = json.load(f)

            # Handle both formats: array of chunks or dict with "chunks" key
            if isinstance(chunks_file, dict) and "chunks" in chunks_file:
                chunks_data = chunks_file["chunks"]
            elif isinstance(chunks_file, list):
                chunks_data = chunks_file
            else:
                chunks_data = []

            if not chunks_data:
                error_msg = f"No chunks found in {chunks_path}"
                logger.warning(error_msg)
                return EmbeddingResult(
                    document_id=document_id,
                    source_path=chunks_path,
                    embeddings=[],
                    total_embeddings=0,
                    processing_time=time.time() - start_time,
                    api_calls=0,
                    total_tokens=0,
                    status=EmbeddingGenerationStatus.FAILED,
                    error_message=error_msg,
                )

            logger.info(f"Generating embeddings for {len(chunks_data)} chunks")

            # Extract text content and chunk IDs
            texts = [chunk["text_content"] for chunk in chunks_data]
            chunk_ids = [chunk["chunk_id"] for chunk in chunks_data]

            # Generate embeddings in batches
            embeddings, api_calls, total_tokens = await self._generate_embeddings_batch(
                texts
            )

            if not embeddings:
                error_msg = "Failed to generate any embeddings"
                logger.error(error_msg)
                return EmbeddingResult(
                    document_id=document_id,
                    source_path=chunks_path,
                    embeddings=[],
                    total_embeddings=0,
                    processing_time=time.time() - start_time,
                    api_calls=api_calls,
                    total_tokens=0,
                    status=EmbeddingGenerationStatus.FAILED,
                    error_message=error_msg,
                )

            # Create ChunkEmbedding objects
            chunk_embeddings = [
                ChunkEmbedding(
                    chunk_id=chunk_ids[i],
                    embedding_vector=embeddings[i],
                    model=self.model_name,
                    dimensions=self.dimensions,
                    created_at=datetime.now(),
                )
                for i in range(len(embeddings))
            ]

            # Save embeddings to Parquet + JSON
            self._save_embeddings(
                document_id=document_id,
                chunks_data=chunks_data,
                chunk_embeddings=chunk_embeddings,
                storage=storage,
            )

            processing_time = time.time() - start_time
            logger.info(
                f"Generated {len(chunk_embeddings)} embeddings in "
                f"{processing_time:.2f}s ({api_calls} API calls, "
                f"{total_tokens} tokens)"
            )

            return EmbeddingResult(
                document_id=document_id,
                source_path=chunks_path,
                embeddings=chunk_embeddings,
                total_embeddings=len(chunk_embeddings),
                processing_time=processing_time,
                api_calls=api_calls,
                total_tokens=total_tokens,
                status=EmbeddingGenerationStatus.SUCCESS,
            )

        except Exception as e:
            logger.error(f"Error generating embeddings for {document_id}: {e}")
            return EmbeddingResult(
                document_id=document_id,
                source_path=Path("unknown"),
                embeddings=[],
                total_embeddings=0,
                processing_time=time.time() - start_time,
                api_calls=0,
                total_tokens=0,
                status=EmbeddingGenerationStatus.FAILED,
                error_message=str(e),
            )

    async def _generate_embeddings_batch(
        self,
        texts: list[str],
    ) -> tuple[list[list[float]], int, int]:
        """
        Generate embeddings for a list of texts using batching and retry logic.

        Args:
            texts: List of text strings to embed

        Returns:
            Tuple of (embeddings list, api_calls count, total_tokens used)
        """
        # Sentence-transformer local inference (no API calls)
        if self.model_provider == "sentence_transformer":
            import numpy as np

            from backend.etl.utils.gpu_memory import release_gpu_memory

            batch_size = self.batch_size
            max_oom_retries = 3

            for oom_attempt in range(max_oom_retries + 1):
                try:
                    vectors = self.st_model.encode(
                        texts,
                        batch_size=batch_size,
                        show_progress_bar=True,
                        normalize_embeddings=True,
                    )
                    break  # success
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    if oom_attempt >= max_oom_retries:
                        logger.error(
                            f"OOM after {max_oom_retries} retries "
                            f"(batch_size={batch_size}), giving up"
                        )
                        raise
                    new_batch_size = max(1, batch_size // 2)
                    logger.warning(
                        f"CUDA OOM at batch_size={batch_size}, "
                        f"retrying with batch_size={new_batch_size} "
                        f"(attempt {oom_attempt + 1}/{max_oom_retries})"
                    )
                    release_gpu_memory()
                    batch_size = new_batch_size

            # Truncate to configured dimensions (MRL)
            if vectors.shape[1] > self.dimensions:
                vectors = vectors[:, : self.dimensions]
                # Re-normalize after truncation
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = vectors / norms
            return [v.tolist() for v in vectors], 0, 0

        # OpenRouter API path (OpenAI-compatible)
        import numpy as np

        embeddings = []
        api_calls = 0
        total_tokens = 0

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            # Retry logic with exponential backoff
            for attempt in range(self.max_retries):
                try:
                    logger.debug(
                        f"Generating embeddings for batch {i // self.batch_size + 1} "
                        f"(size: {len(batch)}, attempt: {attempt + 1})"
                    )

                    response = await self.client.embeddings.create(
                        model=self.model_name,
                        input=batch,
                        timeout=self.timeout,
                    )

                    api_calls += 1
                    total_tokens += response.usage.total_tokens

                    # Extract embeddings from response
                    batch_embeddings = [item.embedding for item in response.data]
                    embeddings.extend(batch_embeddings)

                    # Rate limiting between batches
                    if i + self.batch_size < len(texts):
                        await asyncio.sleep(self.rate_limit_delay)

                    break  # Success, exit retry loop

                except RateLimitError as e:
                    logger.warning(f"Rate limit hit, attempt {attempt + 1}: {e}")
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delays[attempt]
                        logger.info(f"Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"Max retries reached for batch starting at index {i}"
                        )
                        raise

                except OpenAIError as e:
                    logger.error(f"OpenRouter API error on attempt {attempt + 1}: {e}")
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delays[attempt]
                        logger.info(f"Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"Max retries reached for batch starting at index {i}"
                        )
                        raise

                except Exception as e:
                    logger.error(f"Unexpected error generating embeddings: {e}")
                    raise

        # MRL truncation: trim to configured dimensions and re-normalize
        if embeddings and len(embeddings[0]) > self.dimensions:
            vectors = np.array(embeddings)[:, : self.dimensions]
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / norms
            embeddings = [v.tolist() for v in vectors]

        return embeddings, api_calls, total_tokens

    def _save_embeddings(
        self,
        document_id: str,
        chunks_data: list[dict],
        chunk_embeddings: list[ChunkEmbedding],
        storage,
    ) -> None:
        """
        Save embeddings to Parquet file and metadata to JSON.

        Args:
            document_id: Document identifier
            chunks_data: Original chunk data with metadata
            chunk_embeddings: Generated embeddings
            storage: Storage abstraction for path resolution
        """
        try:
            # Create output directory
            output_dir = self._resolve_output_dir(document_id, storage)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Check if embeddings already exist (resume capability)
            embeddings_file = output_dir / "embeddings.parquet"
            metadata_file = output_dir / "metadata.json"

            # Also check remote storage for existence
            output_rel = self._resolve_output_relative(document_id, storage)
            if output_rel and storage and hasattr(storage, "exists"):
                emb_rel = f"{output_rel}/embeddings.parquet"
                meta_rel = f"{output_rel}/metadata.json"
                if storage.exists(emb_rel) and storage.exists(meta_rel):
                    logger.info(
                        f"Embeddings already exist for {document_id}. Skipping save."
                    )
                    return

            if embeddings_file.exists() and metadata_file.exists():
                logger.info(
                    f"Embeddings already exist for {document_id} at {output_dir}. "
                    "Skipping save."
                )
                return

            # Prepare data for Parquet (embeddings only)
            embeddings_df = pd.DataFrame(
                {
                    "chunk_id": [emb.chunk_id for emb in chunk_embeddings],
                    "embedding": [emb.embedding_vector for emb in chunk_embeddings],
                }
            )

            # Save embeddings to Parquet
            table = pa.Table.from_pandas(embeddings_df)
            pq.write_table(table, embeddings_file, compression="snappy")

            logger.info(
                f"Saved {len(chunk_embeddings)} embeddings to {embeddings_file}"
            )

            # Save metadata to JSON (chunk text, page numbers, sections, etc.)
            metadata = {
                "document_id": document_id,
                "total_chunks": len(chunks_data),
                "embedding_model": self.model_name,
                "embedding_dimensions": self.dimensions,
                "created_at": datetime.now().isoformat(),
                "chunks": chunks_data,  # Full chunk metadata
            }

            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved metadata to {metadata_file}")

            # Upload to remote storage (no-op for local)
            if output_rel and storage and hasattr(storage, "upload"):
                storage.upload(output_rel)

        except Exception as e:
            logger.error(f"Failed to save embeddings for {document_id}: {e}")
            raise

    def _resolve_output_relative(self, document_id: str, storage) -> str | None:
        """Resolve the storage-relative output directory for embeddings."""
        if not storage or not hasattr(storage, "glob"):
            return None
        try:
            pattern = f"processed/chunks/**/{document_id}/chunks.json"
            matches = storage.glob(pattern)
            if matches:
                chunks_rel = matches[0]
                # Replace "chunks" segment with "embeddings" and drop "chunks.json"
                parts = Path(chunks_rel).parts
                try:
                    idx = list(parts).index("chunks")
                    new_parts = (
                        list(parts[:idx]) + ["embeddings"] + list(parts[idx + 1 : -1])
                    )
                    return str(Path(*new_parts))
                except ValueError:
                    pass
        except Exception:
            pass
        return f"processed/embeddings/{document_id}"

    def _resolve_chunks_path(self, document_id: str, storage) -> Path | None:
        """Resolve path to chunks.json for a document.

        Uses ``storage.glob()`` to find the file in both local and cloud
        storage, then ``storage.download()`` to ensure it's available
        locally.
        """
        if storage and hasattr(storage, "glob") and callable(storage.glob):
            try:
                pattern = f"processed/chunks/**/{document_id}/chunks.json"
                matches = storage.glob(pattern)
                if matches:
                    if len(matches) > 1:
                        logger.warning(
                            f"Multiple chunks found for {document_id}, "
                            f"using: {matches[0]}"
                        )
                    # Download to local cache and return local path
                    return storage.download(matches[0])
            except Exception as e:
                logger.debug(f"Error resolving chunks path via storage: {e}")

        # Fallback: search in local data directory
        base_dir = self.config.get("runtime", {}).get("local_storage_path", "data/")
        base_path = Path(base_dir)
        if not base_path.is_absolute():
            base_path = self.config_path.parent / base_path

        chunks_base = base_path / "processed" / "chunks"
        if chunks_base.exists():
            pattern = f"**/{document_id}/chunks.json"
            matches = list(chunks_base.glob(pattern))
            if matches:
                return matches[0]

        return None

    def _resolve_output_dir(self, document_id: str, storage) -> Path:
        """Resolve output directory for embeddings."""
        if storage and hasattr(storage, "get_path") and callable(storage.get_path):
            try:
                embeddings_base = storage.get_path("processed/embeddings")
                if isinstance(embeddings_base, Path):
                    # Mirror the chunks directory structure
                    chunks_path = self._resolve_chunks_path(document_id, storage)
                    if chunks_path:
                        # Extract source type from chunks path
                        # e.g., processed/chunks/documents/growthlab/doc_id
                        #       -> processed/embeddings/documents/growthlab/doc_id
                        relative_parts = []
                        for part in chunks_path.parent.parts:
                            if part == "chunks":
                                break
                            relative_parts.append(part)
                        # Take parts after "chunks"
                        chunks_parts = chunks_path.parent.parts
                        start_idx = chunks_parts.index("chunks") + 1
                        relative_parts = list(chunks_parts[start_idx:])

                        output_dir = embeddings_base
                        for part in relative_parts:
                            output_dir = output_dir / part

                        return output_dir
            except Exception as e:
                logger.debug(f"Error resolving output dir via storage: {e}")

        # Fallback: mirror chunks structure in data/processed/embeddings
        base_dir = self.config.get("runtime", {}).get("local_storage_path", "data/")
        base_path = Path(base_dir)
        if not base_path.is_absolute():
            base_path = self.config_path.parent / base_path

        embeddings_base = base_path / "processed" / "embeddings"

        # Try to find chunks path to mirror structure
        chunks_path = self._resolve_chunks_path(document_id, storage)
        if chunks_path:
            chunks_parts = chunks_path.parent.parts
            try:
                start_idx = chunks_parts.index("chunks") + 1
                relative_parts = list(chunks_parts[start_idx:])
                output_dir = embeddings_base
                for part in relative_parts:
                    output_dir = output_dir / part
                return output_dir
            except ValueError:
                pass

        # Last resort: flat structure
        return embeddings_base / document_id

    async def process_all_documents(
        self,
        storage,
        limit: int | None = None,
        document_ids: list[str] | None = None,
        tracker=None,
    ) -> list[EmbeddingResult]:
        """
        Process all eligible documents for embedding generation.

        Args:
            storage: Storage abstraction
            limit: Optional limit on number of documents to process
            document_ids: Optional list of specific document IDs to process
            tracker: Optional PublicationTracker instance (for testing)

        Returns:
            List of EmbeddingResult objects
        """
        from backend.etl.utils.publication_tracker import PublicationTracker

        if tracker is None:
            tracker = PublicationTracker()
        results: list[EmbeddingResult] = []

        try:
            if document_ids:
                # Process specific documents by ID
                # Get publications for embedding and filter by document_ids
                all_publications = tracker.get_publications_for_embedding()
                publications = [
                    pub
                    for pub in all_publications
                    if pub.publication_id in document_ids
                ]

                # Warn about any missing documents
                found_ids = {pub.publication_id for pub in publications}
                for doc_id in document_ids:
                    if doc_id not in found_ids:
                        logger.warning(f"Publication not found or not ready: {doc_id}")
            else:
                # Get all documents ready for embedding
                publications = tracker.get_publications_for_embedding(limit=limit)

            if not publications:
                logger.info("No documents found for embedding generation")
                return results

            logger.info(f"Processing {len(publications)} documents for embeddings")

            for pub in publications:
                try:
                    # Update status to IN_PROGRESS
                    from backend.etl.models.tracking import EmbeddingStatus

                    tracker.update_embedding_status(
                        pub.publication_id,
                        EmbeddingStatus.IN_PROGRESS,
                    )

                    # Generate embeddings
                    result = await self.generate_embeddings_for_document(
                        document_id=pub.publication_id,
                        storage=storage,
                    )

                    results.append(result)

                    # Update status based on result
                    if result.status == EmbeddingGenerationStatus.SUCCESS:
                        tracker.update_embedding_status(
                            pub.publication_id,
                            EmbeddingStatus.EMBEDDED,
                        )
                        logger.info(f"Successfully embedded {pub.publication_id}")
                    else:
                        tracker.update_embedding_status(
                            pub.publication_id,
                            EmbeddingStatus.FAILED,
                            error=result.error_message,
                        )
                        logger.error(
                            f"Failed to embed {pub.publication_id}: "
                            f"{result.error_message}"
                        )

                except Exception as e:
                    logger.error(f"Error processing {pub.publication_id}: {e}")
                    tracker.update_embedding_status(
                        pub.publication_id,
                        EmbeddingStatus.FAILED,
                        error=str(e),
                    )

            return results

        except Exception as e:
            logger.error(f"Error in process_all_documents: {e}")
            raise


# Helper function for synchronous usage
def run_embeddings_generator(
    config_path: Path,
    storage,
    limit: int | None = None,
    document_ids: list[str] | None = None,
    tracker=None,
) -> list[EmbeddingResult]:
    """
    Synchronous wrapper for embedding generation.

    Args:
        config_path: Path to configuration file
        storage: Storage abstraction
        limit: Optional limit on number of documents
        document_ids: Optional list of specific document IDs
        tracker: Optional PublicationTracker instance (for testing)

    Returns:
        List of EmbeddingResult objects
    """
    generator = EmbeddingsGenerator(config_path=config_path)
    return asyncio.run(
        generator.process_all_documents(
            storage=storage,
            limit=limit,
            document_ids=document_ids,
            tracker=tracker,
        )
    )
