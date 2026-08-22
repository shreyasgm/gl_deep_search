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

from backend.etl.utils.atomic_io import atomic_writer


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
    # True when embeddings already existed and nothing was generated. Callers
    # must not treat a run of skipped documents as a failed run.
    skipped: bool = False

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

        A publication directory holding more than one text file gets one extra
        level for every file after the first, with the id suffixed to match:
        - processed/chunks/documents/growthlab/gl_url_123/report_v2/chunks.json
          → document_id ``gl_url_123__report_v2``

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
            # Local-inference memory controls. Default to None so the model's
            # own config decides; config.yaml sets these explicitly for the
            # 8B model, where fp32 weights alone (~32 GB) leave too little
            # headroom on an 80 GB A100 once activations are allocated.
            "dtype": None,
            "max_seq_length": None,
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
        self.dtype = merged["dtype"]
        self.max_seq_length = merged["max_seq_length"]

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
            # Load in reduced precision when configured. Without an explicit
            # dtype an 8B model lands in fp32 (~32 GB) and long-sequence
            # batches then OOM on an 80 GB A100.
            model_kwargs = {"dtype": self.dtype} if self.dtype else {}
            self.st_model = SentenceTransformer(
                self.model_name,
                trust_remote_code=True,
                model_kwargs=model_kwargs,
            )
            # Cap sequence length. Chunks target ~500 tokens, but outliers up
            # to max_chunk_size would otherwise drive quadratic attention cost.
            if self.max_seq_length:
                self.st_model.max_seq_length = self.max_seq_length
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
            # Skip early if embeddings already exist
            emb_dir = self._resolve_output_relative(document_id, storage)
            if emb_dir and storage:
                emb_rel = f"{emb_dir}/embeddings.parquet"
                meta_rel = f"{emb_dir}/metadata.json"
                if storage.exists(emb_rel) and storage.exists(meta_rel):
                    logger.info(f"Embeddings already exist for {document_id}, skipping")
                    return EmbeddingResult(
                        document_id=document_id,
                        source_path=Path(emb_rel),
                        embeddings=[],
                        total_embeddings=0,
                        processing_time=time.time() - start_time,
                        api_calls=0,
                        total_tokens=0,
                        status=EmbeddingGenerationStatus.SUCCESS,
                        skipped=True,
                    )

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
            # Keep halving until the batch reaches 1 before giving up. A fixed
            # retry count meant a batch_size of 32 bottomed out at 4, so the
            # documents that OOM the hardest were never actually retried small.
            max_oom_retries = max(1, batch_size.bit_length())

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
                    if batch_size <= 1 or oom_attempt >= max_oom_retries:
                        logger.error(
                            f"OOM after {oom_attempt} retries "
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

            # A short vector must never be written: the parquet would hold
            # fewer dimensions than metadata.json (and Qdrant) claim, which
            # only surfaces much later, at ingestion.
            model_dimensions = int(vectors.shape[1])
            if model_dimensions < self.dimensions:
                raise ValueError(
                    f"Model returned {model_dimensions}-dimensional embeddings "
                    f"but {self.dimensions} dimensions are configured"
                )

            # Truncate to configured dimensions (MRL)
            if model_dimensions > self.dimensions:
                vectors = vectors[:, : self.dimensions]
                # Re-normalize after truncation
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = vectors / norms
            result = [v.tolist() for v in vectors], 0, 0
            # Release per document, not just on OOM. Over a few hundred
            # documents the cached allocator fragments badly enough that a
            # later long document fails even though total usage is fine.
            del vectors
            release_gpu_memory()
            return result

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

        # Same guard as the local path: short vectors must fail the document
        # loudly rather than reach parquet under a wrong dimension count.
        if embeddings and len(embeddings[0]) < self.dimensions:
            raise ValueError(
                f"Model returned {len(embeddings[0])}-dimensional embeddings "
                f"but {self.dimensions} dimensions are configured"
            )

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

            # Save embeddings to Parquet.  Written atomically, and before
            # metadata.json, because the resume check above requires both files:
            # metadata.json is the commit marker for the pair.
            table = pa.Table.from_pandas(embeddings_df)
            with atomic_writer(embeddings_file, "wb") as f:
                pq.write_table(table, f, compression="snappy")

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

            # Written last and atomically: metadata.json appearing is what makes
            # the resume check treat this document as done.
            with atomic_writer(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved metadata to {metadata_file}")

            # Upload to remote storage (no-op for local)
            if output_rel and storage and hasattr(storage, "upload"):
                storage.upload(output_rel)

        except Exception as e:
            logger.error(f"Failed to save embeddings for {document_id}: {e}")
            raise

    @staticmethod
    def _chunks_search_patterns(document_id: str) -> list[str]:
        """Glob patterns, relative to ``processed/chunks``, for a document.

        A document that was first in its directory sits directly under a
        directory named after it. A document the chunker had to disambiguate
        (because its directory held another text file) sits one level deeper,
        under ``<pub_dir>/<stem>/``, and carries the id ``<pub_dir>__<stem>``.

        Args:
            document_id: Document identifier.

        Returns:
            Patterns to try, in order; the legacy layout comes first.
        """
        from backend.etl.utils.text_chunker import DOCUMENT_ID_SEPARATOR

        patterns = [f"**/{document_id}/chunks.json"]
        head, _, stem = document_id.rpartition(DOCUMENT_ID_SEPARATOR)
        if head and stem:
            patterns.append(f"**/{head}/{stem}/chunks.json")
        return patterns

    def _resolve_output_relative(self, document_id: str, storage) -> str | None:
        """Resolve the storage-relative output directory for embeddings."""
        if not storage or not hasattr(storage, "glob"):
            return None
        try:
            for pattern in self._chunks_search_patterns(document_id):
                # Sorted: glob order is not guaranteed, and an output path that
                # varies between runs breaks resume.
                matches = sorted(storage.glob(f"processed/chunks/{pattern}"))
                if not matches:
                    continue
                chunks_rel = matches[0]
                # Replace "chunks" segment with "embeddings" and drop "chunks.json"
                parts = Path(chunks_rel).parts
                try:
                    idx = list(parts).index("chunks")
                except ValueError:
                    continue
                new_parts = (
                    list(parts[:idx]) + ["embeddings"] + list(parts[idx + 1 : -1])
                )
                return str(Path(*new_parts))
        except Exception:
            pass
        return f"processed/embeddings/{document_id}"

    def _chunked_document_ids(self, storage) -> set[str]:
        """Return IDs of every document that has chunks on disk.

        Args:
            storage: Storage abstraction

        Returns:
            Set of document IDs with a ``chunks.json`` present.
        """
        from backend.etl.utils.text_chunker import document_ids_for_artifacts

        relatives = sorted(storage.glob("processed/chunks/**/chunks.json"))
        return set(document_ids_for_artifacts(relatives).values())

    def _discover_documents_from_chunks(
        self,
        storage,
        limit: int | None = None,
    ) -> list[str]:
        """Discover document IDs by scanning chunk files on disk.

        Returns document IDs that have chunks but no existing embeddings.
        """
        from backend.etl.utils.text_chunker import document_ids_for_artifacts

        # Sorted: glob order is not guaranteed, so an unsorted work list makes
        # a limited run process a different subset of documents each time.
        chunk_relatives = sorted(storage.glob("processed/chunks/**/chunks.json"))
        # Ids come from the chunker's own path scheme rather than the parent
        # directory name, so documents that had to be disambiguated
        # (<pub_dir>/<stem>/chunks.json → <pub_dir>__<stem>) resolve back to
        # the id their chunk_ids were built from.
        chunk_ids = document_ids_for_artifacts(chunk_relatives)

        # Build set of already-embedded document IDs from parquet paths
        embedded_relatives = sorted(
            storage.glob("processed/embeddings/**/embeddings.parquet")
        )
        embedded_ids = set(document_ids_for_artifacts(embedded_relatives).values())

        seen: set[str] = set()
        doc_ids: list[str] = []
        for rel in chunk_relatives:
            doc_id = chunk_ids[rel]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            if doc_id not in embedded_ids:
                doc_ids.append(doc_id)

        if limit:
            doc_ids = doc_ids[:limit]
        return doc_ids

    def _resolve_chunks_path(self, document_id: str, storage) -> Path | None:
        """Resolve path to chunks.json for a document.

        Uses ``storage.glob()`` to find the file in both local and cloud
        storage, then ``storage.download()`` to ensure it's available
        locally.
        """
        if storage and hasattr(storage, "glob") and callable(storage.glob):
            try:
                for pattern in self._chunks_search_patterns(document_id):
                    # Sorted so the same file is picked on every run.
                    matches = sorted(storage.glob(f"processed/chunks/{pattern}"))
                    if not matches:
                        continue
                    if len(matches) > 1:
                        # Two documents sharing an id is a bug upstream, not a
                        # situation to resolve silently.
                        logger.error(
                            f"Multiple chunks found for {document_id}: {matches}, "
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
            for pattern in self._chunks_search_patterns(document_id):
                matches = sorted(chunks_base.glob(pattern))
                if not matches:
                    continue
                if len(matches) > 1:
                    logger.error(
                        f"Multiple chunks found for {document_id}: {matches}, "
                        f"using: {matches[0]}"
                    )
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

        The work list is disk-driven: documents discovered by scanning chunk
        files are always merged with the publications the tracker considers
        eligible. Sources that never register in the tracker (e.g. OpenAlex)
        are therefore embedded too, and documents whose tracker row is stale
        (FAILED, or EMBEDDED without embeddings on disk) are picked back up.
        Idempotency is preserved because ``_discover_documents_from_chunks``
        excludes documents that already have embeddings on disk and
        ``generate_embeddings_for_document`` skips them as well.

        Args:
            storage: Storage abstraction
            limit: Optional limit on number of documents to process, applied
                once to the merged work list
            document_ids: Optional list of specific document IDs to process
            tracker: Optional PublicationTracker instance (for testing)

        Returns:
            List of EmbeddingResult objects
        """
        from backend.etl.models.tracking import EmbeddingStatus
        from backend.etl.utils.publication_tracker import PublicationTracker

        if tracker is None:
            tracker = PublicationTracker()
        results: list[EmbeddingResult] = []
        # Document ids with no tracker row, reported once at the end.
        untracked: list[str] = []

        def update_status(
            doc_id: str,
            status: EmbeddingStatus,
            error: str | None = None,
        ) -> None:
            """Update tracker status, tolerating documents with no tracker row.

            Args:
                doc_id: Document identifier
                status: New embedding status
                error: Optional error message
            """
            try:
                updated = tracker.update_embedding_status(doc_id, status, error=error)
            except Exception as e:
                # A locked or corrupt tracker DB must not look like success.
                logger.warning(f"Tracker update FAILED for {doc_id}: {e}")
                return
            if updated is False:
                # Expected for sources that never register in the tracker
                # (OpenAlex, lectures). Counted and reported once by the
                # caller rather than logged per document.
                untracked.append(doc_id)

        try:
            # Disk scan always runs, so untracked documents are never dropped.
            # No limit here: the limit applies to the merged list below.
            discovered = self._discover_documents_from_chunks(storage)

            if document_ids:
                # Restrict both sources to the explicitly requested documents
                requested = set(document_ids)
                tracked = [
                    pub.publication_id
                    for pub in tracker.get_publications_for_embedding()
                    if pub.publication_id in requested
                ]
                discovered = [doc_id for doc_id in discovered if doc_id in requested]
            else:
                tracked = [
                    pub.publication_id
                    for pub in tracker.get_publications_for_embedding(limit=limit)
                ]

            # Merge, tracker-derived IDs first, dropping duplicates
            seen: set[str] = set()
            work: list[str] = []
            for doc_id in [*tracked, *discovered]:
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                work.append(doc_id)

            if document_ids:
                for doc_id in document_ids:
                    if doc_id not in seen:
                        logger.warning(f"Publication not found or not ready: {doc_id}")

            # Only documents with chunks on disk can be embedded. The tracker
            # can disagree with disk (e.g. after processed/ is cleared), and
            # attempting those would mark healthy rows FAILED for no reason.
            chunked = self._chunked_document_ids(storage)
            missing = [doc_id for doc_id in work if doc_id not in chunked]
            if missing:
                work = [doc_id for doc_id in work if doc_id in chunked]
                preview = ", ".join(sorted(missing)[:10])
                suffix = ", ..." if len(missing) > 10 else ""
                logger.warning(
                    f"Skipping {len(missing)} tracker-listed documents with no "
                    f"chunks on disk (nothing to embed yet): {preview}{suffix}"
                )

            # Apply the limit once, to the merged list
            if limit:
                work = work[:limit]

            if not work:
                logger.info("No documents found for embedding generation")
                return results

            tracked_ids = set(tracked)
            from_tracker = sum(1 for doc_id in work if doc_id in tracked_ids)
            discovered_only = len(work) - from_tracker
            logger.info(
                f"Embedding {len(work)} documents ({from_tracker} from tracker, "
                f"{discovered_only} discovered on disk)"
            )

            for doc_id in work:
                try:
                    # Tracker updates are best-effort: an untracked document
                    # must still be embedded.
                    update_status(doc_id, EmbeddingStatus.IN_PROGRESS)

                    # Generate embeddings
                    result = await self.generate_embeddings_for_document(
                        document_id=doc_id,
                        storage=storage,
                    )

                    results.append(result)

                    # Update status based on result
                    if result.status == EmbeddingGenerationStatus.SUCCESS:
                        update_status(doc_id, EmbeddingStatus.EMBEDDED)
                        logger.info(f"Successfully embedded {doc_id}")
                    else:
                        update_status(
                            doc_id,
                            EmbeddingStatus.FAILED,
                            error=result.error_message,
                        )
                        logger.error(
                            f"Failed to embed {doc_id}: {result.error_message}"
                        )

                except Exception as e:
                    logger.error(f"Error processing {doc_id}: {e}")
                    update_status(doc_id, EmbeddingStatus.FAILED, error=str(e))

            if untracked:
                # Expected for OpenAlex and lecture transcripts, which are not
                # registered in the tracker. Reported once so genuine
                # tracker/disk divergence is still noticeable.
                logger.info(
                    f"{len(untracked)} embedded documents have no tracker row "
                    f"(normal for OpenAlex and lectures)"
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
