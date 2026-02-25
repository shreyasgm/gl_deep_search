"""Tests for the embedding service (query-time dense + sparse embedding)."""

import math
from unittest.mock import AsyncMock, Mock, patch

import pytest
from qdrant_client import models as qdrant_models

from backend.service.config import ServiceSettings
from backend.service.embedding_service import EmbeddingService


@pytest.fixture
def settings():
    """Service settings with OpenRouter defaults."""
    return ServiceSettings(
        embedding_model="qwen/qwen3-embedding-8b",
        embedding_dimensions=1024,
        embedding_api_base_url="https://openrouter.ai/api/v1",
        embedding_api_key="test-key",
        openai_api_key="test-openai-key",
    )


class TestEmbedQueryTruncation:
    """Test MRL dimension truncation in embed_query."""

    @pytest.mark.asyncio
    async def test_truncation_when_api_returns_full_dims(self, settings):
        """When API returns 4096 dims, truncate to 1024 and renormalize."""
        service = EmbeddingService(settings)

        # Mock AsyncOpenAI client
        mock_client = AsyncMock()
        full_vector = [0.1] * 4096
        mock_response = Mock()
        mock_response.data = [Mock(embedding=full_vector)]
        mock_client.embeddings.create.return_value = mock_response
        service._client = mock_client

        result = await service.embed_query("test query")

        assert len(result) == 1024
        # Should be renormalized
        norm = math.sqrt(sum(x * x for x in result))
        assert abs(norm - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_no_truncation_when_dims_match(self, settings):
        """When API returns exactly 1024 dims, no truncation needed."""
        service = EmbeddingService(settings)

        mock_client = AsyncMock()
        # Return exactly 1024 dims
        vector_1024 = [0.03] * 1024
        mock_response = Mock()
        mock_response.data = [Mock(embedding=vector_1024)]
        mock_client.embeddings.create.return_value = mock_response
        service._client = mock_client

        result = await service.embed_query("test query")

        # Should be unchanged (no truncation)
        assert len(result) == 1024
        assert result == vector_1024

    @pytest.mark.asyncio
    async def test_embed_query_calls_correct_model(self, settings):
        """Verify the correct model ID is passed to the API."""
        service = EmbeddingService(settings)

        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1024)]
        mock_client.embeddings.create.return_value = mock_response
        service._client = mock_client

        await service.embed_query("test")

        mock_client.embeddings.create.assert_called_once_with(
            model="qwen/qwen3-embedding-8b",
            input=["test"],
        )


class TestEmbeddingServiceInit:
    """Test initialization with OpenRouter config."""

    def test_initialize_creates_client_with_openrouter_base_url(self, settings):
        """Verify AsyncOpenAI is created with OpenRouter base_url."""
        with (
            patch("backend.service.embedding_service.AsyncOpenAI") as mock_openai,
            patch("backend.service.embedding_service.SparseTextEmbedding"),
        ):
            service = EmbeddingService(settings)
            service.initialize()

            mock_openai.assert_called_once_with(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
            )


class TestSparseEmbedding:
    """Tests for sparse (BM25) embedding methods."""

    @pytest.fixture
    def initialized_service(self, settings):
        """EmbeddingService with mocked sparse model."""
        with (
            patch("backend.service.embedding_service.AsyncOpenAI"),
            patch("backend.service.embedding_service.SparseTextEmbedding") as mock_cls,
        ):
            mock_sparse = Mock()
            mock_cls.return_value = mock_sparse
            service = EmbeddingService(settings)
            service.initialize()
        return service, mock_sparse

    def test_sparse_embed_query_returns_sparse_vector(self, initialized_service):
        """sparse_embed_query returns a SparseVector with correct indices/values."""
        service, mock_sparse = initialized_service
        import numpy as np

        result_obj = Mock()
        result_obj.indices = np.array([3, 17, 42])
        result_obj.values = np.array([0.9, 0.4, 0.1])
        mock_sparse.query_embed.return_value = iter([result_obj])

        sv = service.sparse_embed_query("some text")

        assert isinstance(sv, qdrant_models.SparseVector)
        assert sv.indices == [3, 17, 42]
        assert sv.values == [0.9, 0.4, 0.1]

    def test_sparse_embed_query_calls_query_embed_not_embed(self, initialized_service):
        """sparse_embed_query must use query_embed (BM25 query mode), not embed."""
        service, mock_sparse = initialized_service
        import numpy as np

        result_obj = Mock()
        result_obj.indices = np.array([0])
        result_obj.values = np.array([1.0])
        mock_sparse.query_embed.return_value = iter([result_obj])

        service.sparse_embed_query("some text")

        mock_sparse.query_embed.assert_called_once_with("some text")
        mock_sparse.embed.assert_not_called()

    def test_sparse_embed_documents_returns_one_vector_per_text(
        self, initialized_service
    ):
        """sparse_embed_documents returns a list with one SparseVector per input."""
        service, mock_sparse = initialized_service
        import numpy as np

        r1 = Mock()
        r1.indices = np.array([1, 5])
        r1.values = np.array([0.8, 0.3])
        r2 = Mock()
        r2.indices = np.array([2, 9])
        r2.values = np.array([0.7, 0.2])
        mock_sparse.embed.return_value = iter([r1, r2])

        vectors = service.sparse_embed_documents(["text one", "text two"])

        assert len(vectors) == 2
        assert vectors[0].indices == [1, 5]
        assert vectors[0].values == [0.8, 0.3]
        assert vectors[1].indices == [2, 9]
        assert vectors[1].values == [0.7, 0.2]

    def test_sparse_model_not_initialized_raises(self, settings):
        """Accessing sparse methods before initialize() raises RuntimeError."""
        service = EmbeddingService(settings)

        with pytest.raises(RuntimeError, match="not initialized"):
            service.sparse_embed_query("hello")

    def test_sparse_embed_query_empty_result(self, initialized_service):
        """Empty indices/values from BM25 should produce a valid SparseVector."""
        service, mock_sparse = initialized_service
        import numpy as np

        result_obj = Mock()
        result_obj.indices = np.array([], dtype=np.int64)
        result_obj.values = np.array([], dtype=np.float64)
        mock_sparse.query_embed.return_value = iter([result_obj])

        sv = service.sparse_embed_query("xyzzy")

        assert sv.indices == []
        assert sv.values == []
