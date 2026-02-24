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
from dotenv import load_dotenv

from backend.etl.models.publications import GrowthLabPublication
from backend.etl.models.tracking import EmbeddingStatus, ProcessingStatus
from backend.etl.utils.embeddings_generator import (
    ChunkEmbedding,
    EmbeddingGenerationStatus,
    EmbeddingsGenerator,
)
from backend.etl.utils.publication_tracker import PublicationTracker
from backend.storage.local import LocalStorage

# Load environment variables from .env file for integration tests
load_dotenv(dotenv_path=Path(__file__).parents[2] / "etl" / ".env")


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
    async def test_sentence_transformer_save_format(
        self, st_config_dir, test_storage, sample_chunks
    ):
        """Verify embeddings are saved as 1024-dim in Parquet + metadata JSON."""
        config_path = st_config_dir / "config.yaml"
        temp_dir, storage = test_storage
        doc_id = "test_st_doc"

        # Create chunks file
        chunks_dir = (
            temp_dir / "processed" / "chunks" / "documents" / "growthlab" / doc_id
        )
        chunks_dir.mkdir(parents=True, exist_ok=True)
        with open(chunks_dir / "chunks.json", "w") as f:
            json.dump(sample_chunks, f)

        mock_model = Mock()
        raw_vectors = np.random.randn(2, 4096).astype(np.float32)
        norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
        mock_model.encode.return_value = raw_vectors / norms

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ):
            generator = EmbeddingsGenerator(config_path=config_path)

        # Create 1024-dim chunk embeddings
        chunk_embeddings = [
            ChunkEmbedding(
                chunk_id=chunk["chunk_id"],
                embedding_vector=[0.1] * 1024,
                model="Qwen/Qwen3-Embedding-8B",
                dimensions=1024,
                created_at=datetime.now(),
            )
            for chunk in sample_chunks
        ]

        generator._save_embeddings(
            document_id=doc_id,
            chunks_data=sample_chunks,
            chunk_embeddings=chunk_embeddings,
            storage=storage,
        )

        # Verify output
        embeddings_base = temp_dir / "processed" / "embeddings"
        embeddings_files = list(embeddings_base.rglob("embeddings.parquet"))
        assert len(embeddings_files) == 1

        df = pd.read_parquet(embeddings_files[0])
        assert len(df) == 2
        assert len(df.iloc[0]["embedding"]) == 1024

        metadata_file = embeddings_files[0].parent / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)
        assert metadata["embedding_model"] == "Qwen/Qwen3-Embedding-8B"
        assert metadata["embedding_dimensions"] == 1024

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
        self, temp_config_dir, tracker_with_test_data
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
        self, temp_config_dir, tracker_with_test_data
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
        self, temp_config_dir, test_storage
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
