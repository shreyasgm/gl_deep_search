"""Tests for the FastAPI application endpoints and helper functions."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client import models as qdrant_models
from starlette.testclient import TestClient

from backend.service.main import (
    _build_qdrant_filter,
    _scored_point_to_chunk_result,
    app,
    get_agent,
    get_embedding,
    get_qdrant,
    get_service_settings,
)
from backend.service.models import ChunkResult, SearchFilters

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_settings():
    """Minimal ServiceSettings mock."""
    settings = MagicMock()
    settings.qdrant_collection = "test_collection"
    settings.max_top_k = 50
    return settings


@pytest.fixture
def mock_qdrant_service():
    """Mocked QdrantService for dependency injection."""
    qdrant = AsyncMock()
    qdrant.health_check = AsyncMock(return_value=True)
    info_mock = MagicMock()
    info_mock.points_count = 42
    qdrant.collection_info = AsyncMock(return_value=info_mock)
    qdrant.search = AsyncMock(return_value=[])
    return qdrant


@pytest.fixture
def mock_embedding_service():
    """Mocked EmbeddingService for dependency injection."""
    embedding = AsyncMock()
    embedding.embed_query = AsyncMock(return_value=[0.1] * 1024)
    return embedding


@pytest.fixture
def mock_agent():
    """Mocked SearchAgent for dependency injection."""
    agent = AsyncMock()
    agent.run = AsyncMock(
        return_value={
            "answer": "Test answer",
            "citations": [
                {
                    "source_number": 1,
                    "document_title": "Test Doc",
                    "document_url": "http://example.com",
                    "document_year": 2023,
                    "document_authors": ["Author A"],
                    "relevant_quote": "Some quote",
                }
            ],
            "search_queries": ["test query"],
            "chunks": [{"text_content": "chunk"}],
        }
    )
    return agent


@pytest.fixture
def client(mock_settings, mock_qdrant_service, mock_embedding_service, mock_agent):
    """TestClient with overridden dependencies."""
    app.dependency_overrides[get_service_settings] = lambda: mock_settings
    app.dependency_overrides[get_qdrant] = lambda: mock_qdrant_service
    app.dependency_overrides[get_embedding] = lambda: mock_embedding_service
    app.dependency_overrides[get_agent] = lambda: mock_agent
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# _build_qdrant_filter() tests
# ---------------------------------------------------------------------------


class TestBuildQdrantFilter:
    """Tests for _build_qdrant_filter() helper in main.py."""

    def test_none_filters_returns_none(self):
        result = _build_qdrant_filter(None)
        assert result is None

    def test_empty_filters_returns_none(self):
        filters = SearchFilters(year=None, document_id=None)
        result = _build_qdrant_filter(filters)
        assert result is None

    def test_year_only(self):
        filters = SearchFilters(year=2023)
        result = _build_qdrant_filter(filters)
        assert result is not None
        assert len(result.must) == 1
        assert result.must[0].key == "document_year"
        assert result.must[0].match.value == 2023

    def test_document_id_only(self):
        filters = SearchFilters(document_id="doc-abc")
        result = _build_qdrant_filter(filters)
        assert result is not None
        assert len(result.must) == 1
        assert result.must[0].key == "document_id"
        assert result.must[0].match.value == "doc-abc"

    def test_year_and_document_id(self):
        filters = SearchFilters(year=2020, document_id="doc-xyz")
        result = _build_qdrant_filter(filters)
        assert result is not None
        assert len(result.must) == 2


# ---------------------------------------------------------------------------
# _scored_point_to_chunk_result() tests
# ---------------------------------------------------------------------------


class TestScoredPointToChunkResult:
    """Tests for _scored_point_to_chunk_result() helper."""

    def test_full_payload(self):
        point = qdrant_models.ScoredPoint(
            id="uuid-123",
            version=1,
            score=0.92,
            payload={
                "document_id": "doc-1",
                "text_content": "Some text content",
                "page_numbers": [1, 2],
                "section_title": "Introduction",
                "chunk_index": 3,
                "token_count": 150,
                "document_title": "My Document",
                "document_authors": ["Alice", "Bob"],
                "document_year": 2022,
                "document_abstract": "An abstract",
                "document_url": "http://example.com/doc",
            },
        )
        result = _scored_point_to_chunk_result(point)

        assert isinstance(result, ChunkResult)
        assert result.chunk_id == "uuid-123"
        assert result.document_id == "doc-1"
        assert result.text_content == "Some text content"
        assert result.score == 0.92
        assert result.page_numbers == [1, 2]
        assert result.section_title == "Introduction"
        assert result.chunk_index == 3
        assert result.token_count == 150
        assert result.document_title == "My Document"
        assert result.document_authors == ["Alice", "Bob"]
        assert result.document_year == 2022
        assert result.document_abstract == "An abstract"
        assert result.document_url == "http://example.com/doc"

    def test_missing_optional_fields_uses_defaults(self):
        point = qdrant_models.ScoredPoint(
            id="uuid-456",
            version=1,
            score=0.5,
            payload={},  # all fields missing
        )
        result = _scored_point_to_chunk_result(point)

        assert result.chunk_id == "uuid-456"
        assert result.document_id == ""
        assert result.text_content == ""
        assert result.score == 0.5
        assert result.page_numbers == []
        assert result.section_title is None
        assert result.chunk_index == 0
        assert result.token_count == 0
        assert result.document_title is None
        assert result.document_authors == []
        assert result.document_year is None
        assert result.document_abstract is None
        assert result.document_url is None

    def test_none_payload_uses_defaults(self):
        point = qdrant_models.ScoredPoint(
            id="uuid-789",
            version=1,
            score=0.3,
            payload=None,
        )
        result = _scored_point_to_chunk_result(point)
        assert result.chunk_id == "uuid-789"
        assert result.document_id == ""


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["qdrant_connected"] is True
        assert data["collection"] == "test_collection"
        assert data["points_count"] == 42


class TestSearchChunksEndpoint:
    """Tests for POST /search/chunks."""

    def test_valid_request_returns_search_response(
        self, client, mock_qdrant_service, mock_embedding_service
    ):
        # Set up mock to return a scored point
        fake_point = qdrant_models.ScoredPoint(
            id="chunk-1",
            version=1,
            score=0.88,
            payload={
                "document_id": "doc-1",
                "text_content": "Growth diagnostics...",
                "page_numbers": [5],
                "section_title": "Methods",
                "chunk_index": 2,
                "token_count": 100,
                "document_title": "Growth Diagnostics",
                "document_authors": ["Hausmann"],
                "document_year": 2005,
                "document_abstract": None,
                "document_url": "http://example.com",
            },
        )
        mock_qdrant_service.search = AsyncMock(return_value=[fake_point])

        response = client.post(
            "/search/chunks",
            json={"query": "growth diagnostics", "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "growth diagnostics"
        assert data["total_results"] == 1
        assert len(data["results"]) == 1
        result = data["results"][0]
        assert result["document_title"] == "Growth Diagnostics"
        assert result["score"] == 0.88

    def test_top_k_clamped_to_max(
        self, client, mock_settings, mock_qdrant_service, mock_embedding_service
    ):
        """top_k in the request is clamped to settings.max_top_k."""
        mock_settings.max_top_k = 5

        response = client.post(
            "/search/chunks",
            json={"query": "test query", "top_k": 50},
        )

        assert response.status_code == 200
        # Verify qdrant.search received the clamped value (5, not 50)
        mock_qdrant_service.search.assert_called_once()
        call_kwargs = mock_qdrant_service.search.call_args
        assert call_kwargs.kwargs["top_k"] == 5


class TestAgentSearchEndpoint:
    """Tests for POST /search."""

    def test_valid_request_returns_agent_response(self, client):
        response = client.post(
            "/search",
            json={"query": "What is economic complexity?", "top_k": 10},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What is economic complexity?"
        assert data["answer"] == "Test answer"
        assert len(data["citations"]) == 1
        assert data["citations"][0]["source_number"] == 1
        assert data["search_queries_used"] == ["test query"]
        assert data["chunks_retrieved"] == 1


# ---------------------------------------------------------------------------
# Error-handling endpoint tests
# ---------------------------------------------------------------------------


class TestEndpointErrorHandling:
    """Tests for error paths in API endpoints."""

    def test_search_chunks_500_on_service_error(
        self, client, mock_qdrant_service, mock_embedding_service
    ):
        """Qdrant search raises → 500 response."""
        mock_qdrant_service.search = AsyncMock(
            side_effect=RuntimeError("Qdrant connection lost")
        )

        response = client.post(
            "/search/chunks",
            json={"query": "test query", "top_k": 5},
        )

        assert response.status_code == 500
        assert "search failed" in response.json()["detail"].lower()

    def test_agent_search_500_on_agent_error(self, client, mock_agent):
        """Agent.run raises → 500 response."""
        mock_agent.run = AsyncMock(side_effect=RuntimeError("Agent crashed"))

        response = client.post(
            "/search",
            json={"query": "What is complexity?", "top_k": 10},
        )

        assert response.status_code == 500
        assert "failed" in response.json()["detail"].lower()

    def test_health_degraded_on_collection_error(self, client, mock_qdrant_service):
        """collection_info raises → health returns unhealthy status."""
        mock_qdrant_service.collection_info = AsyncMock(
            side_effect=RuntimeError("Collection gone")
        )
        # health_check still passes
        mock_qdrant_service.health_check = AsyncMock(return_value=True)

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        # When collection_info fails, connected is set to False
        assert data["status"] == "unhealthy"
        assert data["qdrant_connected"] is False
