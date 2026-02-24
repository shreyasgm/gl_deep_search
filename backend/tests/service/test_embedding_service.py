"""Tests for the embedding service (query-time dense + sparse embedding)."""

import math
from unittest.mock import AsyncMock, Mock, patch

import pytest

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
