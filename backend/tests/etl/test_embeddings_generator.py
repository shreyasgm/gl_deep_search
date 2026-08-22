"""
Tests for the embeddings generation system.

This test suite focuses on reliability and critical workflows:
- Retry mechanism with exponential backoff
- Output format validation (Parquet + JSON)
- Resume capability (idempotency)
- PublicationTracker integration
- SentenceTransformer provider (mocked unit tests + real integration)
- Integration tests with real OpenRouter API (small scale)
"""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from backend.etl.models.publications import GrowthLabPublication
from backend.etl.models.tracking import EmbeddingStatus, ProcessingStatus
from backend.etl.utils.embeddings_generator import (
    ChunkEmbedding,
    EmbeddingGenerationStatus,
    EmbeddingResult,
    EmbeddingsGenerator,
)
from backend.etl.utils.publication_tracker import PublicationTracker
from backend.storage.local import LocalStorage


@pytest.fixture
def test_storage():
    """Create temporary directory for test storage."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create directory structure
    (temp_dir / "processed" / "chunks").mkdir(parents=True, exist_ok=True)
    (temp_dir / "processed" / "embeddings").mkdir(parents=True, exist_ok=True)

    # Create storage instance
    storage = LocalStorage(base_path=temp_dir)

    yield temp_dir, storage

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_config_dir(test_storage):
    """Create temporary directory with test configuration."""
    temp_dir = Path(tempfile.mkdtemp())
    storage_dir, _ = test_storage
    config_path = temp_dir / "config.yaml"

    # Create test configuration pointing to test storage
    config_content = f"""
file_processing:
  embedding:
    model: "openrouter"
    model_name: "qwen/qwen3-embedding-8b"
    dimensions: 1024
    batch_size: 32
    max_retries: 3
    retry_delays: [1, 2, 4]
    timeout: 30
    rate_limit_delay: 0.1

runtime:
  local_storage_path: "{storage_dir}/"
"""
    config_path.write_text(config_content)
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_chunks():
    """Create sample chunks data for testing."""
    return [
        {
            "chunk_id": "test_doc_chunk_0001",
            "source_document_id": "test_doc",
            "source_file_path": "test.txt",
            "chunk_index": 0,
            "text_content": (
                "This is a test chunk about economic growth and development."
            ),
            "character_start": 0,
            "character_end": 60,
            "page_numbers": [1],
            "section_title": "Introduction",
            "metadata": {"strategy": "hybrid"},
            "created_at": datetime.now().isoformat(),
            "chunk_size": 60,
        },
        {
            "chunk_id": "test_doc_chunk_0002",
            "source_document_id": "test_doc",
            "source_file_path": "test.txt",
            "chunk_index": 1,
            "text_content": (
                "Economic complexity theory provides insights into development."
            ),
            "character_start": 60,
            "character_end": 122,
            "page_numbers": [1],
            "section_title": "Introduction",
            "metadata": {"strategy": "hybrid"},
            "created_at": datetime.now().isoformat(),
            "chunk_size": 62,
        },
    ]


class TestEmbeddingsGeneratorUnit:
    """Unit tests for embeddings generator with mocked API."""

    @pytest.mark.asyncio
    async def test_retry_mechanism_with_eventual_success(
        self, temp_config_dir, sample_chunks
    ):
        """Test retry mechanism on API failure with eventual success."""
        config_path = temp_config_dir / "config.yaml"
        generator = EmbeddingsGenerator(config_path=config_path)

        # Mock OpenRouter to fail twice then succeed
        from openai import RateLimitError

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1024)]
        mock_response.usage.total_tokens = 10

        call_count = 0

        async def mock_create_with_retry(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                # Create a proper RateLimitError with required kwargs
                mock_resp = Mock()
                mock_resp.status_code = 429
                raise RateLimitError(
                    "Rate limit exceeded",
                    response=mock_resp,
                    body={"error": "rate_limit_exceeded"},
                )
            return mock_response

        with patch.object(
            generator.client.embeddings, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = mock_create_with_retry

            # Generate embeddings (should succeed after retries)
            texts = [sample_chunks[0]["text_content"]]
            (
                embeddings,
                api_calls,
                total_tokens,
            ) = await generator._generate_embeddings_batch(texts)

            # Verify retries occurred
            assert call_count == 3  # Total attempts (2 failures + 1 success)
            assert api_calls == 1  # Only successful API calls counted
            assert len(embeddings) == 1

    @pytest.mark.asyncio
    async def test_save_embeddings_format(
        self, temp_config_dir, test_storage, sample_chunks
    ):
        """Test that embeddings are saved in correct format (Parquet + JSON)."""
        config_path = temp_config_dir / "config.yaml"
        temp_dir, storage = test_storage

        doc_id = "test_doc"

        # Create chunks file first so path resolution works
        chunks_dir = (
            temp_dir / "processed" / "chunks" / "documents" / "growthlab" / doc_id
        )
        chunks_dir.mkdir(parents=True, exist_ok=True)
        chunks_file = chunks_dir / "chunks.json"
        with open(chunks_file, "w") as f:
            json.dump(sample_chunks, f)

        generator = EmbeddingsGenerator(config_path=config_path)

        # Create sample embeddings
        chunk_embeddings = [
            ChunkEmbedding(
                chunk_id=chunk["chunk_id"],
                embedding_vector=[0.1] * 1024,
                model="qwen/qwen3-embedding-8b",
                dimensions=1024,
                created_at=datetime.now(),
            )
            for chunk in sample_chunks
        ]

        # Save embeddings
        generator._save_embeddings(
            document_id=doc_id,
            chunks_data=sample_chunks,
            chunk_embeddings=chunk_embeddings,
            storage=storage,
        )

        # Find where the embeddings were actually saved
        embeddings_base = temp_dir / "processed" / "embeddings"
        embeddings_files = list(embeddings_base.rglob("embeddings.parquet"))
        assert len(embeddings_files) == 1, "Should have exactly one embeddings file"

        embeddings_file = embeddings_files[0]
        embeddings_dir = embeddings_file.parent
        metadata_file = embeddings_dir / "metadata.json"

        assert embeddings_file.exists()
        assert metadata_file.exists()

        # Verify Parquet content
        df = pd.read_parquet(embeddings_file)
        assert len(df) == 2
        assert "chunk_id" in df.columns
        assert "embedding" in df.columns
        assert len(df.iloc[0]["embedding"]) == 1024

        # Verify JSON metadata
        with open(metadata_file) as f:
            metadata = json.load(f)

        assert metadata["document_id"] == doc_id
        assert metadata["total_chunks"] == 2
        assert metadata["embedding_model"] == "qwen/qwen3-embedding-8b"
        assert metadata["embedding_dimensions"] == 1024
        assert len(metadata["chunks"]) == 2

    @pytest.mark.asyncio
    async def test_resume_capability(
        self, temp_config_dir, test_storage, sample_chunks
    ):
        """Test that existing embeddings are not overwritten (resume capability)."""
        config_path = temp_config_dir / "config.yaml"
        temp_dir, storage = test_storage

        doc_id = "test_doc"

        # Create chunks file first so path resolution works
        chunks_dir = (
            temp_dir / "processed" / "chunks" / "documents" / "growthlab" / doc_id
        )
        chunks_dir.mkdir(parents=True, exist_ok=True)
        chunks_file = chunks_dir / "chunks.json"
        with open(chunks_file, "w") as f:
            json.dump(sample_chunks, f)

        generator = EmbeddingsGenerator(config_path=config_path)

        chunk_embeddings = [
            ChunkEmbedding(
                chunk_id=chunk["chunk_id"],
                embedding_vector=[0.1] * 1024,
                model="qwen/qwen3-embedding-8b",
                dimensions=1024,
                created_at=datetime.now(),
            )
            for chunk in sample_chunks
        ]

        # Save embeddings first time
        generator._save_embeddings(
            document_id=doc_id,
            chunks_data=sample_chunks,
            chunk_embeddings=chunk_embeddings,
            storage=storage,
        )

        # Try to save again (should skip)
        generator._save_embeddings(
            document_id=doc_id,
            chunks_data=sample_chunks,
            chunk_embeddings=chunk_embeddings,
            storage=storage,
        )

        # Verify files still exist
        embeddings_base = temp_dir / "processed" / "embeddings"
        embeddings_files = list(embeddings_base.rglob("embeddings.parquet"))
        assert len(embeddings_files) == 1, "Should have exactly one embeddings file"
        assert embeddings_files[0].exists()


class TestSentenceTransformerProvider:
    """Unit tests for the sentence_transformer embedding provider."""

    @pytest.fixture
    def st_config_dir(self, test_storage):
        """Create config for sentence_transformer provider."""
        temp_dir = Path(tempfile.mkdtemp())
        storage_dir, _ = test_storage
        config_path = temp_dir / "config.yaml"

        config_content = f"""
file_processing:
  embedding:
    model: "sentence_transformer"
    model_name: "Qwen/Qwen3-Embedding-8B"
    dimensions: 1024
    batch_size: 32
    dtype: "bfloat16"
    max_seq_length: 2048
    max_retries: 3
    retry_delays: [1, 2, 4]
    timeout: 30
    rate_limit_delay: 0.1

runtime:
  local_storage_path: "{storage_dir}/"
"""
        config_path.write_text(config_content)
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_oom_retry_halves_batch_all_the_way_to_one(
        self, st_config_dir, sample_chunks
    ):
        """OOM retries must reach batch_size=1 before giving up.

        Regression test for the 2026-02-25 production run, where a fixed
        retry count of 3 meant a batch_size of 32 bottomed out at 4 and
        38 documents (13%) were abandoned to CUDA OOM.
        """
        config_path = st_config_dir / "config.yaml"

        attempted_batch_sizes = []

        def encode(texts, batch_size=32, **kwargs):
            attempted_batch_sizes.append(batch_size)
            if batch_size > 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate 11.72 GiB")
            vectors = np.random.randn(len(texts), 4096).astype(np.float32)
            return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        mock_model = Mock()
        mock_model.encode.side_effect = encode

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ):
            generator = EmbeddingsGenerator(config_path=config_path)

        texts = [chunk["text_content"] for chunk in sample_chunks]
        embeddings, _, _ = await generator._generate_embeddings_batch(texts)

        assert attempted_batch_sizes == [32, 16, 8, 4, 2, 1]
        assert len(embeddings) == len(texts)

    @pytest.mark.asyncio
    async def test_non_oom_runtime_error_is_not_retried(
        self, st_config_dir, sample_chunks
    ):
        """A RuntimeError that isn't an OOM must propagate immediately."""
        config_path = st_config_dir / "config.yaml"

        mock_model = Mock()
        mock_model.encode.side_effect = RuntimeError("kernel launch failed")

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ):
            generator = EmbeddingsGenerator(config_path=config_path)

        texts = [chunk["text_content"] for chunk in sample_chunks]
        with pytest.raises(RuntimeError, match="kernel launch failed"):
            await generator._generate_embeddings_batch(texts)

        assert mock_model.encode.call_count == 1

    def test_memory_controls_applied_to_model(self, st_config_dir):
        """bf16 dtype and the sequence cap must reach SentenceTransformer."""
        config_path = st_config_dir / "config.yaml"

        mock_model = Mock()
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ) as mock_cls:
            EmbeddingsGenerator(config_path=config_path)

        kwargs = mock_cls.call_args.kwargs
        assert kwargs["model_kwargs"] == {"dtype": "bfloat16"}
        assert mock_model.max_seq_length == 2048

    @pytest.mark.asyncio
    async def test_sentence_transformer_truncation_and_renormalization(
        self, st_config_dir, sample_chunks
    ):
        """Verify MRL truncation to 1024 dims with renormalization."""
        config_path = st_config_dir / "config.yaml"

        # Mock SentenceTransformer to avoid downloading the real model
        mock_model = Mock()
        # Simulate model returning 4096-dim normalized vectors
        raw_vectors = np.random.randn(2, 4096).astype(np.float32)
        norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
        raw_vectors = raw_vectors / norms
        mock_model.encode.return_value = raw_vectors

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ):
            generator = EmbeddingsGenerator(config_path=config_path)

        texts = [chunk["text_content"] for chunk in sample_chunks]
        (
            embeddings,
            api_calls,
            total_tokens,
        ) = await generator._generate_embeddings_batch(texts)

        # Should return 1024-dim vectors (truncated from 4096)
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1024
        assert len(embeddings[1]) == 1024

        # No API calls for local model
        assert api_calls == 0
        assert total_tokens == 0

        # Vectors should be normalized (L2 norm ~= 1.0)
        for emb in embeddings:
            norm = sum(x * x for x in emb) ** 0.5
            assert abs(norm - 1.0) < 1e-5, f"Expected norm ~1.0, got {norm}"

        # Verify encode was called with correct args
        mock_model.encode.assert_called_once_with(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

    @pytest.mark.asyncio
    async def test_sentence_transformer_no_truncation_when_dims_match(
        self, test_storage
    ):
        """When model output dims <= configured dims, no truncation happens."""
        temp_dir = Path(tempfile.mkdtemp())
        storage_dir, _ = test_storage
        config_path = temp_dir / "config.yaml"

        # Set dims to 4096 (same as model native output)
        config_content = f"""
file_processing:
  embedding:
    model: "sentence_transformer"
    model_name: "Qwen/Qwen3-Embedding-8B"
    dimensions: 4096
    batch_size: 32

runtime:
  local_storage_path: "{storage_dir}/"
"""
        config_path.write_text(config_content)

        mock_model = Mock()
        raw_vectors = np.random.randn(1, 4096).astype(np.float32)
        norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
        raw_vectors = raw_vectors / norms
        mock_model.encode.return_value = raw_vectors

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ):
            generator = EmbeddingsGenerator(config_path=config_path)

        embeddings, _, _ = await generator._generate_embeddings_batch(["test text"])

        # Should keep full 4096 dims
        assert len(embeddings[0]) == 4096

        shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_too_few_dimensions_fails_the_document(
        self, st_config_dir, sample_chunks
    ):
        """A model returning fewer dims than configured must fail loudly.

        Writing the short vectors would leave the parquet disagreeing with
        metadata.json, which only surfaces later, at Qdrant ingestion.
        """
        config_path = st_config_dir / "config.yaml"

        mock_model = Mock()
        raw_vectors = np.random.randn(2, 512).astype(np.float32)
        mock_model.encode.return_value = raw_vectors / np.linalg.norm(
            raw_vectors, axis=1, keepdims=True
        )

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ):
            generator = EmbeddingsGenerator(config_path=config_path)

        texts = [chunk["text_content"] for chunk in sample_chunks]
        with pytest.raises(ValueError, match="512.*1024"):
            await generator._generate_embeddings_batch(texts)

    @pytest.mark.asyncio
    async def test_real_sentence_transformer_inference(self, test_storage):
        """Integration test: load real model and generate embeddings.

        Uses all-MiniLM-L6-v2 (~80MB) — fast enough to run in every
        test suite without GPU.
        """
        temp_dir = Path(tempfile.mkdtemp())
        storage_dir, _ = test_storage
        config_path = temp_dir / "config.yaml"

        config_content = f"""
file_processing:
  embedding:
    model: "sentence_transformer"
    model_name: "all-MiniLM-L6-v2"
    dimensions: 384
    batch_size: 2

runtime:
  local_storage_path: "{storage_dir}/"
"""
        config_path.write_text(config_content)

        generator = EmbeddingsGenerator(config_path=config_path)
        texts = [
            "Economic growth requires productive capabilities.",
            "Development pathways depend on economic complexity.",
        ]
        (
            embeddings,
            api_calls,
            total_tokens,
        ) = await generator._generate_embeddings_batch(texts)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384
        assert api_calls == 0

        # Vectors should be normalized
        for emb in embeddings:
            norm = sum(x * x for x in emb) ** 0.5
            assert abs(norm - 1.0) < 1e-4

        # Different texts should produce different embeddings
        assert embeddings[0] != embeddings[1]

        shutil.rmtree(temp_dir)


class TestEmbeddingsGeneratorIntegration:
    """Integration tests with real OpenRouter API (small scale)."""

    @pytest.fixture
    def tracker_with_test_data(self, test_storage):
        """Create PublicationTracker with test data."""
        temp_dir, storage = test_storage

        # Use the real tracker (will use default database)
        tracker = PublicationTracker()

        # Add test publication with unique ID to avoid conflicts
        pub_id = f"test_integration_{datetime.now().timestamp()}"
        publication = GrowthLabPublication(
            paper_id=pub_id,
            title="Test Document",
            pub_url="https://example.com/test",
            file_urls=["https://example.com/test.pdf"],
        )
        tracker.add_publication(publication)

        # Mark as processed (ready for embedding)
        tracker.update_processing_status(pub_id, ProcessingStatus.PROCESSED)

        # Create sample chunks (structure: chunks/documents/growthlab/{doc_id})
        chunks_dir = (
            temp_dir / "processed" / "chunks" / "documents" / "growthlab" / pub_id
        )
        chunks_dir.mkdir(parents=True, exist_ok=True)

        chunks_data = [
            {
                "chunk_id": f"{pub_id}_chunk_0001",
                "source_document_id": pub_id,
                "source_file_path": "test.txt",
                "chunk_index": 0,
                "text_content": "Economic growth requires productive capabilities.",
                "character_start": 0,
                "character_end": 50,
                "page_numbers": [1],
                "section_title": "Introduction",
                "metadata": {"strategy": "hybrid"},
                "created_at": datetime.now().isoformat(),
                "chunk_size": 50,
            },
            {
                "chunk_id": f"{pub_id}_chunk_0002",
                "source_document_id": pub_id,
                "source_file_path": "test.txt",
                "chunk_index": 1,
                "text_content": "Development pathways depend on economic complexity.",
                "character_start": 50,
                "character_end": 102,
                "page_numbers": [1],
                "section_title": "Introduction",
                "metadata": {"strategy": "hybrid"},
                "created_at": datetime.now().isoformat(),
                "chunk_size": 52,
            },
        ]

        chunks_file = chunks_dir / "chunks.json"
        with open(chunks_file, "w") as f:
            json.dump(chunks_data, f, indent=2)

        yield tracker, temp_dir, storage, pub_id

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_embedding_generation(
        self, temp_config_dir, tracker_with_test_data, require_api_keys
    ):
        """Test complete embedding generation workflow with real OpenRouter API."""
        config_path = temp_config_dir / "config.yaml"
        tracker, temp_dir, storage, pub_id = tracker_with_test_data

        generator = EmbeddingsGenerator(config_path=config_path)

        # Generate embeddings
        result = await generator.generate_embeddings_for_document(
            document_id=pub_id,
            storage=storage,
        )

        # Verify result
        assert result.status == EmbeddingGenerationStatus.SUCCESS
        assert result.total_embeddings == 2
        assert result.api_calls >= 1
        assert result.processing_time > 0

        # Verify embeddings file (structure: embeddings/documents/growthlab/{doc_id})
        embeddings_dir = (
            temp_dir / "processed" / "embeddings" / "documents" / "growthlab" / pub_id
        )
        embeddings_file = embeddings_dir / "embeddings.parquet"
        metadata_file = embeddings_dir / "metadata.json"

        assert embeddings_file.exists()
        assert metadata_file.exists()

        # Verify Parquet content
        df = pd.read_parquet(embeddings_file)
        assert len(df) == 2
        assert all(len(emb) == 1024 for emb in df["embedding"])

        # Verify embeddings are different (not all zeros)
        assert (df.iloc[0]["embedding"] != df.iloc[1]["embedding"]).any()
        assert sum(df.iloc[0]["embedding"]) != 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_publication_tracker_integration(
        self, temp_config_dir, tracker_with_test_data, require_api_keys
    ):
        """Test integration with PublicationTracker status updates."""
        config_path = temp_config_dir / "config.yaml"
        tracker, temp_dir, storage, pub_id = tracker_with_test_data

        generator = EmbeddingsGenerator(config_path=config_path)

        # Get publication before processing
        pub_before = tracker.get_publication_status(pub_id)
        assert pub_before["embedding_status"] == EmbeddingStatus.PENDING.value

        # Process using process_all_documents (which handles tracker updates)
        results = await generator.process_all_documents(
            storage=storage, document_ids=[pub_id], tracker=tracker
        )

        # Verify results
        assert len(results) == 1
        assert results[0].status == EmbeddingGenerationStatus.SUCCESS

        # Get publication after processing
        pub_after = tracker.get_publication_status(pub_id)
        assert pub_after["embedding_status"] == EmbeddingStatus.EMBEDDED.value

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_processing_multiple_documents(
        self, temp_config_dir, test_storage, require_api_keys
    ):
        """Test batch processing of multiple documents."""
        config_path = temp_config_dir / "config.yaml"
        temp_dir, storage = test_storage

        # Use the real tracker with unique IDs
        tracker = PublicationTracker()
        base_id = f"test_batch_{datetime.now().timestamp()}"
        doc_ids = []

        for i in range(3):
            pub_id = f"{base_id}_{i}"
            doc_ids.append(pub_id)

            # Add to tracker
            publication = GrowthLabPublication(
                paper_id=pub_id,
                title=f"Test Document {i}",
                pub_url=f"https://example.com/test{i}",
                file_urls=[f"https://example.com/test{i}.pdf"],
            )
            tracker.add_publication(publication)
            tracker.update_processing_status(pub_id, ProcessingStatus.PROCESSED)

            # Create chunks (structure: chunks/documents/growthlab/{doc_id})
            chunks_dir = (
                temp_dir / "processed" / "chunks" / "documents" / "growthlab" / pub_id
            )
            chunks_dir.mkdir(parents=True, exist_ok=True)

            chunks_data = [
                {
                    "chunk_id": f"{pub_id}_chunk_0001",
                    "source_document_id": pub_id,
                    "source_file_path": f"test{i}.txt",
                    "chunk_index": 0,
                    "text_content": f"Test content for document {i}.",
                    "character_start": 0,
                    "character_end": 30,
                    "page_numbers": [1],
                    "section_title": "Introduction",
                    "metadata": {"strategy": "hybrid"},
                    "created_at": datetime.now().isoformat(),
                    "chunk_size": 30,
                }
            ]

            with open(chunks_dir / "chunks.json", "w") as f:
                json.dump(chunks_data, f)

        # Process using process_all_documents (which handles tracker updates)
        generator = EmbeddingsGenerator(config_path=config_path)
        results = await generator.process_all_documents(
            storage=storage, document_ids=doc_ids, tracker=tracker
        )

        # Verify results
        assert len(results) == 3
        assert all(r.status == EmbeddingGenerationStatus.SUCCESS for r in results)
        assert sum(r.total_embeddings for r in results) == 3

        # Verify all publications updated
        for i in range(3):
            pub = tracker.get_publication_status(f"{base_id}_{i}")
            assert pub["embedding_status"] == EmbeddingStatus.EMBEDDED.value


class TestEmbeddingsGeneratorErrorPaths:
    """Test error paths in generate_embeddings_for_document()."""

    @pytest.fixture
    def openrouter_generator(self, test_storage):
        """Create an EmbeddingsGenerator with openrouter config."""
        temp_dir = Path(tempfile.mkdtemp())
        storage_dir, _ = test_storage
        config_path = temp_dir / "config.yaml"

        config_content = f"""
file_processing:
  embedding:
    model: "openrouter"
    model_name: "qwen/qwen3-embedding-8b"
    dimensions: 1024
    batch_size: 32
    max_retries: 1
    retry_delays: [0]
    timeout: 5
    rate_limit_delay: 0

runtime:
  local_storage_path: "{storage_dir}/"
"""
        config_path.write_text(config_content)
        generator = EmbeddingsGenerator(config_path=config_path)
        yield generator, storage_dir
        shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_chunks_file_not_found(self, openrouter_generator, test_storage):
        """When chunks file does not exist, should return FAILED status."""
        generator, storage_dir = openrouter_generator
        _, storage = test_storage

        # Use a document_id with no corresponding chunks file
        result = await generator.generate_embeddings_for_document(
            document_id="nonexistent_doc_xyz",
            storage=storage,
        )

        assert result.status == EmbeddingGenerationStatus.FAILED
        assert "not found" in result.error_message.lower()
        assert result.total_embeddings == 0
        assert result.embeddings == []

    @pytest.mark.asyncio
    async def test_empty_chunks_list(self, openrouter_generator, test_storage):
        """When chunks file contains empty list, should return FAILED status."""
        generator, storage_dir = openrouter_generator
        _, storage = test_storage

        # Create a chunks file with empty list
        doc_id = "empty_chunks_doc"
        chunks_dir = (
            Path(storage_dir)
            / "processed"
            / "chunks"
            / "documents"
            / "growthlab"
            / doc_id
        )
        chunks_dir.mkdir(parents=True, exist_ok=True)
        with open(chunks_dir / "chunks.json", "w") as f:
            json.dump([], f)

        result = await generator.generate_embeddings_for_document(
            document_id=doc_id,
            storage=storage,
        )

        assert result.status == EmbeddingGenerationStatus.FAILED
        assert "no chunks" in result.error_message.lower()
        assert result.total_embeddings == 0

    @pytest.mark.asyncio
    async def test_api_returns_no_embeddings(self, openrouter_generator, test_storage):
        """When batch generation returns empty list, should return FAILED."""
        generator, storage_dir = openrouter_generator
        _, storage = test_storage

        doc_id = "no_embeddings_doc"
        chunks_dir = (
            Path(storage_dir)
            / "processed"
            / "chunks"
            / "documents"
            / "growthlab"
            / doc_id
        )
        chunks_dir.mkdir(parents=True, exist_ok=True)
        chunks_data = [
            {
                "chunk_id": f"{doc_id}_chunk_0001",
                "source_document_id": doc_id,
                "source_file_path": "test.txt",
                "chunk_index": 0,
                "text_content": "Some text content.",
                "character_start": 0,
                "character_end": 18,
                "page_numbers": [1],
                "section_title": None,
                "metadata": {},
                "created_at": datetime.now().isoformat(),
                "chunk_size": 18,
            }
        ]
        with open(chunks_dir / "chunks.json", "w") as f:
            json.dump(chunks_data, f)

        # Mock batch generation to return empty embeddings
        with patch.object(
            generator,
            "_generate_embeddings_batch",
            new_callable=AsyncMock,
            return_value=([], 1, 0),
        ):
            result = await generator.generate_embeddings_for_document(
                document_id=doc_id,
                storage=storage,
            )

        assert result.status == EmbeddingGenerationStatus.FAILED
        assert "failed to generate" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_unexpected_exception_during_batch(
        self, openrouter_generator, test_storage
    ):
        """When batch generation raises unexpected exception, should return FAILED."""
        generator, storage_dir = openrouter_generator
        _, storage = test_storage

        doc_id = "exception_doc"
        chunks_dir = (
            Path(storage_dir)
            / "processed"
            / "chunks"
            / "documents"
            / "growthlab"
            / doc_id
        )
        chunks_dir.mkdir(parents=True, exist_ok=True)
        chunks_data = [
            {
                "chunk_id": f"{doc_id}_chunk_0001",
                "source_document_id": doc_id,
                "source_file_path": "test.txt",
                "chunk_index": 0,
                "text_content": "Some text content.",
                "character_start": 0,
                "character_end": 18,
                "page_numbers": [1],
                "section_title": None,
                "metadata": {},
                "created_at": datetime.now().isoformat(),
                "chunk_size": 18,
            }
        ]
        with open(chunks_dir / "chunks.json", "w") as f:
            json.dump(chunks_data, f)

        # Mock batch generation to raise unexpected error
        with patch.object(
            generator,
            "_generate_embeddings_batch",
            new_callable=AsyncMock,
            side_effect=RuntimeError("GPU out of memory"),
        ):
            result = await generator.generate_embeddings_for_document(
                document_id=doc_id,
                storage=storage,
            )

        assert result.status == EmbeddingGenerationStatus.FAILED
        assert "GPU out of memory" in result.error_message


class TestEmbeddingsBatching:
    """Test that batching splits texts correctly and makes the right
    number of API calls."""

    @pytest.mark.asyncio
    async def test_65_texts_at_batch_size_32(self, test_storage):
        """65 texts with batch_size=32 should produce 3 API calls (32, 32, 1)."""
        temp_dir = Path(tempfile.mkdtemp())
        storage_dir, _ = test_storage
        config_path = temp_dir / "config.yaml"

        config_content = f"""
file_processing:
  embedding:
    model: "openrouter"
    model_name: "qwen/qwen3-embedding-8b"
    dimensions: 1024
    batch_size: 32
    max_retries: 1
    retry_delays: [0]
    timeout: 5
    rate_limit_delay: 0

runtime:
  local_storage_path: "{storage_dir}/"
"""
        config_path.write_text(config_content)
        generator = EmbeddingsGenerator(config_path=config_path)

        # Track batch sizes seen by the mock
        batch_sizes_seen = []

        async def mock_create(*args, **kwargs):
            batch_input = kwargs.get("input", args[0] if args else [])
            batch_sizes_seen.append(len(batch_input))

            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 1024) for _ in batch_input]
            mock_response.usage = Mock(total_tokens=len(batch_input) * 10)
            return mock_response

        with patch.object(
            generator.client.embeddings, "create", new_callable=AsyncMock
        ) as mock_embeddings_create:
            mock_embeddings_create.side_effect = mock_create

            texts = [f"Text number {i}" for i in range(65)]
            (
                embeddings,
                api_calls,
                total_tokens,
            ) = await generator._generate_embeddings_batch(texts)

        # Should have made exactly 3 API calls
        assert api_calls == 3
        assert len(batch_sizes_seen) == 3
        assert batch_sizes_seen == [32, 32, 1]

        # Should have 65 embeddings total
        assert len(embeddings) == 65

        shutil.rmtree(temp_dir)


class TestProcessAllDocumentsWorkList:
    """Tests for how process_all_documents() builds its work list."""

    @pytest.fixture
    def generator(self, test_storage):
        """Create an EmbeddingsGenerator that loads no model and needs no key."""
        temp_dir = Path(tempfile.mkdtemp())
        storage_dir, _ = test_storage
        config_path = temp_dir / "config.yaml"

        config_content = f"""
file_processing:
  embedding:
    model: "openrouter"
    model_name: "qwen/qwen3-embedding-8b"
    api_key: "unused-in-these-tests"
    dimensions: 1024
    batch_size: 32
    max_retries: 1
    retry_delays: [0]
    timeout: 5
    rate_limit_delay: 0

runtime:
  local_storage_path: "{storage_dir}/"
"""
        config_path.write_text(config_content)
        yield EmbeddingsGenerator(config_path=config_path)
        shutil.rmtree(temp_dir)

    @staticmethod
    def _mock_tracker(publication_ids: list[str], strict: bool = False) -> Mock:
        """Build a mock tracker returning the given eligible publication IDs.

        Args:
            publication_ids: IDs returned by get_publications_for_embedding()
            strict: If True, status updates raise for any other ID, mimicking
                a tracker that holds no row for that document.

        Returns:
            Mock tracker instance
        """
        tracker = Mock()
        tracker.get_publications_for_embedding.return_value = [
            Mock(publication_id=pub_id) for pub_id in publication_ids
        ]
        if strict:
            known = set(publication_ids)

            def update(doc_id, status, error=None):
                if doc_id not in known:
                    raise ValueError(f"No tracker row for {doc_id}")
                return True

            tracker.update_embedding_status.side_effect = update
        return tracker

    @staticmethod
    def _success_result(document_id: str):
        """Build a successful EmbeddingResult for a document."""
        return EmbeddingResult(
            document_id=document_id,
            source_path=Path("chunks.json"),
            embeddings=[],
            total_embeddings=1,
            processing_time=0.0,
            api_calls=0,
            total_tokens=0,
            status=EmbeddingGenerationStatus.SUCCESS,
        )

    async def _run(self, generator, tracker, discovered, chunked=None, **kwargs):
        """Run process_all_documents with disk scan and embedding mocked.

        Args:
            generator: EmbeddingsGenerator under test
            tracker: Mock tracker
            discovered: IDs the disk scan should return
            chunked: IDs that have chunks on disk. Defaults to every ID in
                play, so the chunk-presence filter is a no-op.
            **kwargs: Extra arguments for process_all_documents()

        Returns:
            Tuple of (embedded document IDs, results, disk scan mock)
        """
        embedded: list[str] = []

        async def fake_embed(document_id, storage):
            embedded.append(document_id)
            return self._success_result(document_id)

        tracked_ids = [
            pub.publication_id
            for pub in tracker.get_publications_for_embedding.return_value
        ]
        if chunked is None:
            chunked = {*tracked_ids, *discovered}
        storage = Mock()
        storage.glob.return_value = [
            f"processed/chunks/documents/src/{doc_id}/chunks.json"
            for doc_id in sorted(chunked)
        ]

        with (
            patch.object(
                generator,
                "_discover_documents_from_chunks",
                return_value=list(discovered),
            ) as mock_discover,
            patch.object(
                generator,
                "generate_embeddings_for_document",
                new=AsyncMock(side_effect=fake_embed),
            ),
        ):
            results = await generator.process_all_documents(
                storage=storage, tracker=tracker, **kwargs
            )

        return embedded, results, mock_discover

    @pytest.mark.asyncio
    async def test_untracked_disk_document_embedded_alongside_tracked(self, generator):
        """Regression: an OpenAlex-style document that exists only on disk is
        still embedded when the tracker also returns eligible publications.

        The old code treated the tracker list and the disk scan as an
        either/or, so a single tracker hit suppressed disk discovery entirely
        and every untracked (OpenAlex) document was silently skipped.
        """
        tracker = self._mock_tracker(["gl_doc_1", "gl_doc_2"], strict=True)

        embedded, results, _ = await self._run(
            generator, tracker, discovered=["oa_W12345"]
        )

        assert embedded == ["gl_doc_1", "gl_doc_2", "oa_W12345"]
        assert len(results) == 3
        assert all(r.status == EmbeddingGenerationStatus.SUCCESS for r in results)
        # Tracked documents still get their status updated...
        tracker.update_embedding_status.assert_any_call(
            "gl_doc_1", EmbeddingStatus.EMBEDDED, error=None
        )
        # ...and the untracked one is embedded even though every tracker
        # update for it raises.
        assert "oa_W12345" in embedded

    @pytest.mark.asyncio
    async def test_disk_documents_embedded_when_tracker_empty(self, generator):
        """Old fallback behaviour must not regress: with nothing eligible in
        the tracker, documents found on disk are still embedded."""
        tracker = self._mock_tracker([], strict=True)

        embedded, results, _ = await self._run(
            generator, tracker, discovered=["oa_A", "oa_B"]
        )

        assert embedded == ["oa_A", "oa_B"]
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_document_in_both_sources_embedded_once(self, generator):
        """A document present in both the tracker list and the disk scan is
        embedded exactly once, with tracker-derived order preserved."""
        tracker = self._mock_tracker(["dup_doc", "gl_doc_2"])

        embedded, results, _ = await self._run(
            generator, tracker, discovered=["dup_doc", "oa_W999"]
        )

        assert embedded == ["dup_doc", "gl_doc_2", "oa_W999"]
        assert embedded.count("dup_doc") == 1
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_limit_applies_to_merged_work_list(self, generator):
        """limit caps the merged list, not each source separately."""
        tracker = self._mock_tracker(["gl_doc_1", "gl_doc_2"])

        embedded, results, mock_discover = await self._run(
            generator,
            tracker,
            discovered=["oa_A", "oa_B"],
            limit=3,
        )

        assert embedded == ["gl_doc_1", "gl_doc_2", "oa_A"]
        assert len(results) == 3
        # The disk scan itself is never limited; the merged list is.
        assert mock_discover.call_count == 1
        assert "limit" not in mock_discover.call_args.kwargs
        assert len(mock_discover.call_args.args) == 1

    async def test_tracker_rows_without_chunks_are_skipped_not_failed(self, generator):
        """Tracker rows whose chunks no longer exist must be skipped silently.

        Regression test for the production cluster state: the tracking DB
        listed 118 documents as PROCESSED/PENDING after data/processed had
        been cleared. Attempting them would mark healthy rows FAILED and
        pollute the tracker, when the correct behaviour is to leave them
        alone until their chunks are regenerated.
        """
        tracker = self._mock_tracker(["stale_1", "stale_2", "has_chunks"])

        embedded, results, _ = await self._run(
            generator,
            tracker,
            discovered=["oa_W999"],
            chunked={"has_chunks", "oa_W999"},
        )

        assert embedded == ["has_chunks", "oa_W999"]
        assert "stale_1" not in embedded
        assert "stale_2" not in embedded
        # The stale rows must not be touched at all — no FAILED writes.
        updated = {
            call.args[0] for call in tracker.update_embedding_status.call_args_list
        }
        assert "stale_1" not in updated
        assert "stale_2" not in updated
        assert len(results) == 2


class TestDisambiguatedDocumentPaths:
    """Documents the chunker had to disambiguate must flow through unchanged.

    A publication directory holding two text files produces
    ``<pub_id>/chunks.json`` for the first file and
    ``<pub_id>/<stem>/chunks.json`` (document id ``<pub_id>__<stem>``) for the
    second. Discovery, chunk lookup and output placement all have to agree on
    that, or the recovered document is embedded under the wrong id or not at
    all.
    """

    @pytest.fixture
    def generator(self, test_storage):
        """Create an EmbeddingsGenerator that loads no model and needs no key."""
        temp_dir = Path(tempfile.mkdtemp())
        storage_dir, _ = test_storage
        config_path = temp_dir / "config.yaml"
        config_path.write_text(
            "file_processing:\n"
            "  embedding:\n"
            '    model: "openrouter"\n'
            '    api_key: "unused-in-these-tests"\n'
            "    dimensions: 1024\n"
            "\n"
            "runtime:\n"
            f'  local_storage_path: "{storage_dir}/"\n'
        )
        yield EmbeddingsGenerator(config_path=config_path)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def _write_chunks(root: Path, relative: str, document_id: str) -> None:
        """Write a minimal chunks.json at *relative* under *root*."""
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                [
                    {
                        "chunk_id": f"{document_id}_chunk_0000",
                        "source_document_id": document_id,
                        "text_content": "Growth diagnostics.",
                    }
                ]
            )
        )

    def test_discovery_recovers_the_suffixed_document_id(self, generator, test_storage):
        """Both documents of a shared directory must be discovered separately."""
        root, storage = test_storage
        base = "processed/chunks/documents/growthlab/gl_url_abc"
        self._write_chunks(root, f"{base}/chunks.json", "gl_url_abc")
        self._write_chunks(
            root, f"{base}/report_v2/chunks.json", "gl_url_abc__report_v2"
        )
        self._write_chunks(
            root,
            "processed/chunks/documents/lecture_transcripts/0_intro/chunks.json",
            "0_intro",
        )

        discovered = generator._discover_documents_from_chunks(storage)

        assert sorted(discovered) == [
            "0_intro",
            "gl_url_abc",
            "gl_url_abc__report_v2",
        ]
        assert generator._chunked_document_ids(storage) == set(discovered)

    def test_chunks_and_output_paths_resolve_for_both_layouts(
        self, generator, test_storage
    ):
        """The suffixed id must find its own chunks.json and its own output dir."""
        root, storage = test_storage
        base = "processed/chunks/documents/growthlab/gl_url_abc"
        self._write_chunks(root, f"{base}/chunks.json", "gl_url_abc")
        self._write_chunks(
            root, f"{base}/report_v2/chunks.json", "gl_url_abc__report_v2"
        )

        legacy = generator._resolve_chunks_path("gl_url_abc", storage)
        nested = generator._resolve_chunks_path("gl_url_abc__report_v2", storage)
        assert legacy == root / f"{base}/chunks.json"
        assert nested == root / f"{base}/report_v2/chunks.json"

        assert generator._resolve_output_relative("gl_url_abc", storage) == str(
            Path("processed/embeddings/documents/growthlab/gl_url_abc")
        )
        assert generator._resolve_output_relative(
            "gl_url_abc__report_v2", storage
        ) == str(Path("processed/embeddings/documents/growthlab/gl_url_abc/report_v2"))
        assert generator._resolve_output_dir("gl_url_abc__report_v2", storage) == (
            root / "processed/embeddings/documents/growthlab/gl_url_abc/report_v2"
        )

    def test_existing_nested_embeddings_are_not_regenerated(
        self, generator, test_storage
    ):
        """Resume must recognise a nested embeddings.parquet as already done."""
        root, storage = test_storage
        base = "processed/chunks/documents/growthlab/gl_url_abc"
        self._write_chunks(root, f"{base}/chunks.json", "gl_url_abc")
        self._write_chunks(
            root, f"{base}/report_v2/chunks.json", "gl_url_abc__report_v2"
        )

        embedded = root / "processed/embeddings/documents/growthlab/gl_url_abc"
        (embedded / "report_v2").mkdir(parents=True, exist_ok=True)
        (embedded / "embeddings.parquet").write_bytes(b"")
        (embedded / "report_v2" / "embeddings.parquet").write_bytes(b"")

        assert generator._discover_documents_from_chunks(storage) == []
