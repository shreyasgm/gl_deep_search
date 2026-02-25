"""Tests for the LangGraph search agent."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from qdrant_client import models as qdrant_models

from backend.service.agent import (
    AgentState,
    CitationRef,
    SearchAgent,
    SynthesisResult,
)
from backend.service.config import ServiceSettings

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings():
    """Minimal ServiceSettings for agent tests."""
    return ServiceSettings(
        qdrant_url="http://localhost:6333",
        anthropic_api_key="test-anthropic-key",
        embedding_api_key="test-embedding-key",
        qdrant_collection="test_collection",
        agent_model="claude-sonnet-4-20250514",
    )


@pytest.fixture
def mock_qdrant():
    """Mocked QdrantService."""
    qdrant = AsyncMock()
    qdrant.hybrid_search = AsyncMock(return_value=[])
    return qdrant


@pytest.fixture
def mock_embedding():
    """Mocked EmbeddingService."""
    embedding = AsyncMock()
    embedding.embed_query = AsyncMock(return_value=[0.1] * 1024)
    embedding.sparse_embed_query = Mock(
        return_value=qdrant_models.SparseVector(indices=[0, 1], values=[0.5, 0.3])
    )
    return embedding


# ---------------------------------------------------------------------------
# _build_filter() tests — static pure function
# ---------------------------------------------------------------------------


class TestBuildFilter:
    """Tests for SearchAgent._build_filter() static method."""

    def test_empty_dict_returns_none(self):
        result = SearchAgent._build_filter({})
        assert result is None

    def test_year_only(self):
        result = SearchAgent._build_filter({"year": 2023})
        assert result is not None
        assert len(result.must) == 1
        condition = result.must[0]
        assert condition.key == "document_year"
        assert condition.match.value == 2023

    def test_document_id_only(self):
        result = SearchAgent._build_filter({"document_id": "doc-123"})
        assert result is not None
        assert len(result.must) == 1
        condition = result.must[0]
        assert condition.key == "document_id"
        assert condition.match.value == "doc-123"

    def test_year_and_document_id_combined(self):
        result = SearchAgent._build_filter({"year": 2020, "document_id": "abc"})
        assert result is not None
        assert len(result.must) == 2
        keys = {c.key for c in result.must}
        assert keys == {"document_year", "document_id"}

    def test_falsy_year_zero_is_skipped(self):
        """year=0 is falsy in Python, so walrus `if year := ...` skips it."""
        result = SearchAgent._build_filter({"year": 0})
        assert result is None


# ---------------------------------------------------------------------------
# _should_retry() tests — pure conditional
# ---------------------------------------------------------------------------


class TestShouldRetry:
    """Tests for SearchAgent._should_retry() conditional edge."""

    @patch("backend.service.agent.init_chat_model")
    def _make_agent(self, settings, mock_qdrant, mock_embedding, mock_init):
        """Helper to create a SearchAgent with mocked LLM."""
        mock_init.return_value = MagicMock()
        return SearchAgent(mock_qdrant, mock_embedding, settings)

    def test_no_chunks_retry_zero_returns_retrieve(
        self, settings, mock_qdrant, mock_embedding
    ):
        agent = self._make_agent(settings, mock_qdrant, mock_embedding)
        state: AgentState = {
            "query": "test",
            "user_filters": {},
            "top_k": 10,
            "search_queries": [],
            "extracted_filters": {},
            "chunks": [],
            "retry_count": 0,
            "answer": "",
            "citations": [],
        }
        assert agent._should_retry(state) == "retrieve"

    def test_no_chunks_retry_two_returns_synthesize(
        self, settings, mock_qdrant, mock_embedding
    ):
        agent = self._make_agent(settings, mock_qdrant, mock_embedding)
        state: AgentState = {
            "query": "test",
            "user_filters": {},
            "top_k": 10,
            "search_queries": [],
            "extracted_filters": {},
            "chunks": [],
            "retry_count": 2,
            "answer": "",
            "citations": [],
        }
        assert agent._should_retry(state) == "synthesize"

    def test_has_chunks_returns_synthesize(self, settings, mock_qdrant, mock_embedding):
        agent = self._make_agent(settings, mock_qdrant, mock_embedding)
        state: AgentState = {
            "query": "test",
            "user_filters": {},
            "top_k": 10,
            "search_queries": [],
            "extracted_filters": {},
            "chunks": [{"text_content": "some chunk", "score": 0.9}],
            "retry_count": 0,
            "answer": "",
            "citations": [],
        }
        assert agent._should_retry(state) == "synthesize"


# ---------------------------------------------------------------------------
# _synthesize() — citation enrichment tests
# ---------------------------------------------------------------------------


class TestSynthesizeCitationEnrichment:
    """Tests for citation enrichment in _synthesize()."""

    @patch("backend.service.agent.init_chat_model")
    def _make_agent(self, settings, mock_qdrant, mock_embedding, mock_init):
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm
        agent = SearchAgent(mock_qdrant, mock_embedding, settings)
        return agent, mock_llm

    async def test_source_number_1_maps_to_chunks_0(
        self, settings, mock_qdrant, mock_embedding
    ):
        """source_number=1 should enrich from chunks[0]."""
        agent, mock_llm = self._make_agent(settings, mock_qdrant, mock_embedding)

        # LLM returns a citation with source_number=1 but empty metadata
        synthesis_result = SynthesisResult(
            answer="Answer text [1]",
            citations=[
                CitationRef(
                    source_number=1,
                    document_title=None,
                    document_url=None,
                    document_year=None,
                    document_authors=[],
                    relevant_quote="a quote",
                )
            ],
        )
        mock_structured = AsyncMock(return_value=synthesis_result)
        mock_llm.with_structured_output.return_value.ainvoke = mock_structured

        chunks = [
            {
                "text_content": "chunk zero text",
                "document_title": "Title From Chunk",
                "document_url": "http://example.com",
                "document_year": 2022,
                "document_authors": ["Author A"],
                "score": 0.95,
            },
        ]

        state: AgentState = {
            "query": "test query",
            "user_filters": {},
            "top_k": 10,
            "search_queries": ["test query"],
            "extracted_filters": {},
            "chunks": chunks,
            "retry_count": 1,
            "answer": "",
            "citations": [],
        }

        result = await agent._synthesize(state)

        assert len(result["citations"]) == 1
        citation = result["citations"][0]
        # source_number=1 should enrich from chunks[0]
        assert citation["document_title"] == "Title From Chunk"
        assert citation["document_url"] == "http://example.com"
        assert citation["document_year"] == 2022
        assert citation["document_authors"] == ["Author A"]

    async def test_out_of_bounds_source_number_no_error(
        self, settings, mock_qdrant, mock_embedding
    ):
        """Out-of-bounds source_number should not raise IndexError."""
        agent, mock_llm = self._make_agent(settings, mock_qdrant, mock_embedding)

        synthesis_result = SynthesisResult(
            answer="Answer text [99]",
            citations=[
                CitationRef(
                    source_number=99,
                    document_title=None,
                    document_url=None,
                    document_year=None,
                    document_authors=[],
                    relevant_quote="a quote",
                )
            ],
        )
        mock_structured = AsyncMock(return_value=synthesis_result)
        mock_llm.with_structured_output.return_value.ainvoke = mock_structured

        chunks = [{"text_content": "only chunk", "score": 0.9}]

        state: AgentState = {
            "query": "test query",
            "user_filters": {},
            "top_k": 10,
            "search_queries": ["test query"],
            "extracted_filters": {},
            "chunks": chunks,
            "retry_count": 1,
            "answer": "",
            "citations": [],
        }

        # Should not raise
        result = await agent._synthesize(state)
        assert len(result["citations"]) == 1
        # Metadata should remain None/empty since index is out of bounds
        citation = result["citations"][0]
        assert citation["document_title"] is None

    async def test_citation_fields_not_overwritten_when_present(
        self, settings, mock_qdrant, mock_embedding
    ):
        """Citation fields from LLM should NOT be overwritten if already populated."""
        agent, mock_llm = self._make_agent(settings, mock_qdrant, mock_embedding)

        synthesis_result = SynthesisResult(
            answer="Answer [1]",
            citations=[
                CitationRef(
                    source_number=1,
                    document_title="LLM-Provided Title",
                    document_url="http://llm-url.com",
                    document_year=2024,
                    document_authors=["LLM Author"],
                    relevant_quote="a quote",
                )
            ],
        )
        mock_structured = AsyncMock(return_value=synthesis_result)
        mock_llm.with_structured_output.return_value.ainvoke = mock_structured

        chunks = [
            {
                "text_content": "chunk text",
                "document_title": "Chunk Title",
                "document_url": "http://chunk-url.com",
                "document_year": 2020,
                "document_authors": ["Chunk Author"],
                "score": 0.9,
            }
        ]

        state: AgentState = {
            "query": "test",
            "user_filters": {},
            "top_k": 10,
            "search_queries": ["test"],
            "extracted_filters": {},
            "chunks": chunks,
            "retry_count": 1,
            "answer": "",
            "citations": [],
        }

        result = await agent._synthesize(state)
        citation = result["citations"][0]
        # LLM-provided values should be preserved (not overwritten by chunk metadata)
        assert citation["document_title"] == "LLM-Provided Title"
        assert citation["document_url"] == "http://llm-url.com"
        assert citation["document_year"] == 2024
        assert citation["document_authors"] == ["LLM Author"]


# ---------------------------------------------------------------------------
# _retrieve() dedup tests
# ---------------------------------------------------------------------------


class TestRetrieveDedup:
    """Tests for deduplication logic in _retrieve()."""

    @patch("backend.service.agent.init_chat_model")
    def _make_agent(self, settings, mock_qdrant, mock_embedding, mock_init):
        mock_init.return_value = MagicMock()
        return SearchAgent(mock_qdrant, mock_embedding, settings)

    @pytest.mark.asyncio
    async def test_retrieve_deduplicates_by_chunk_id_keeps_highest_score(
        self, settings, mock_qdrant, mock_embedding
    ):
        """Duplicate chunk_ids across queries keep only the highest score."""
        agent = self._make_agent(settings, mock_qdrant, mock_embedding)

        # Two queries will each return overlapping results
        point_a1 = Mock()
        point_a1.id = "pt-1"
        point_a1.score = 0.7
        point_a1.payload = {
            "chunk_id": "chunk-A",
            "text_content": "text A",
        }

        point_b = Mock()
        point_b.id = "pt-2"
        point_b.score = 0.6
        point_b.payload = {
            "chunk_id": "chunk-B",
            "text_content": "text B",
        }

        # Same chunk_id "chunk-A" but higher score in second query
        point_a2 = Mock()
        point_a2.id = "pt-3"
        point_a2.score = 0.95
        point_a2.payload = {
            "chunk_id": "chunk-A",
            "text_content": "text A",
        }

        point_c = Mock()
        point_c.id = "pt-4"
        point_c.score = 0.5
        point_c.payload = {
            "chunk_id": "chunk-C",
            "text_content": "text C",
        }

        # First query returns chunk-A (0.7) + chunk-B (0.6)
        # Second query returns chunk-A (0.95) + chunk-C (0.5)
        mock_qdrant.hybrid_search = AsyncMock(
            side_effect=[[point_a1, point_b], [point_a2, point_c]]
        )

        state: AgentState = {
            "query": "test",
            "user_filters": {},
            "top_k": 10,
            "search_queries": ["query one", "query two"],
            "extracted_filters": {},
            "chunks": [],
            "retry_count": 0,
            "answer": "",
            "citations": [],
        }

        result = await agent._retrieve(state)
        chunks = result["chunks"]

        # Should have 3 unique chunks (A, B, C)
        chunk_ids = [c["chunk_id"] for c in chunks]
        assert len(chunk_ids) == 3
        assert len(set(chunk_ids)) == 3

        # chunk-A should have the higher score (0.95, not 0.7)
        chunk_a = next(c for c in chunks if c["chunk_id"] == "chunk-A")
        assert chunk_a["score"] == 0.95

        # Results should be sorted by score descending
        scores = [c["score"] for c in chunks]
        assert scores == sorted(scores, reverse=True)

        # Total count should not exceed top_k
        assert len(chunks) <= state["top_k"]


# ---------------------------------------------------------------------------
# Integration test — real LangGraph execution
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSearchAgentIntegration:
    """Integration test: real LangGraph graph, mocked Qdrant/Embedding/LLM."""

    async def test_run_end_to_end_with_mocked_services(
        self, settings, mock_qdrant, mock_embedding
    ):
        """Test that the full graph executes in correct node order."""
        # Build fake scored points returned by hybrid_search
        fake_point = Mock()
        fake_point.id = "point-1"
        fake_point.score = 0.95
        fake_point.payload = {
            "chunk_id": "chunk-1",
            "text_content": "Economic complexity is...",
            "document_title": "Atlas of Complexity",
            "document_year": 2021,
            "document_authors": ["Hausmann"],
            "document_url": "http://example.com/atlas",
        }
        mock_qdrant.hybrid_search = AsyncMock(return_value=[fake_point])

        with patch("backend.service.agent.init_chat_model") as mock_init:
            mock_llm = MagicMock()
            mock_init.return_value = mock_llm

            # Query analysis node
            from backend.service.agent import QueryAnalysis

            mock_llm.with_structured_output.return_value.ainvoke = AsyncMock()

            # We need to return different things for different calls.
            # Call order: analyze_query, grade_documents, synthesize
            from backend.service.agent import GradingResult

            call_count = 0

            async def side_effect_ainvoke(messages):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # analyze_query
                    return QueryAnalysis(
                        search_queries=["economic complexity"], year_filter=None
                    )
                elif call_count == 2:
                    # grade_documents
                    return GradingResult(relevant_indices=[0])
                else:
                    # synthesize
                    return SynthesisResult(
                        answer="Economic complexity describes...",
                        citations=[
                            CitationRef(
                                source_number=1,
                                document_title=None,
                                document_url=None,
                                document_year=None,
                                document_authors=[],
                                relevant_quote="Economic complexity is...",
                            )
                        ],
                    )

            mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
                side_effect=side_effect_ainvoke
            )

            agent = SearchAgent(mock_qdrant, mock_embedding, settings)
            result = await agent.run("What is economic complexity?")

        # Verify the graph completed and produced expected output
        assert result["answer"] != ""
        answer_text = result["answer"].lower()
        assert "economic complexity" in answer_text or len(answer_text) > 0
        assert len(result["citations"]) >= 1
        assert result["search_queries"] == ["economic complexity"]
        # Citation should be enriched from chunk metadata
        assert result["citations"][0]["document_title"] == "Atlas of Complexity"

    async def test_retry_path_no_chunks_then_broader_search(
        self, settings, mock_qdrant, mock_embedding
    ):
        """First search graded empty → retry broadens query → succeeds."""
        fake_point = Mock()
        fake_point.id = "point-retry"
        fake_point.score = 0.85
        fake_point.payload = {
            "chunk_id": "retry-chunk",
            "text_content": "Retry found this content",
            "document_title": "Retry Doc",
            "document_year": 2023,
            "document_authors": ["Smith"],
            "document_url": "http://example.com/retry",
        }

        # Both searches return points — grading determines what survives
        mock_qdrant.hybrid_search = AsyncMock(return_value=[fake_point])

        with patch("backend.service.agent.init_chat_model") as mock_init:
            mock_llm = MagicMock()
            mock_init.return_value = mock_llm

            from backend.service.agent import GradingResult, QueryAnalysis

            call_count = 0

            async def side_effect_ainvoke(messages):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # analyze_query
                    return QueryAnalysis(
                        search_queries=["specific query"], year_filter=2020
                    )
                elif call_count == 2:
                    # grade_documents (first attempt) — filter everything out
                    return GradingResult(relevant_indices=[])
                elif call_count == 3:
                    # grade_documents (retry) — keep the chunk
                    return GradingResult(relevant_indices=[0])
                else:
                    # synthesize
                    return SynthesisResult(
                        answer="Found via retry",
                        citations=[
                            CitationRef(
                                source_number=1,
                                document_title=None,
                                document_url=None,
                                document_year=None,
                                document_authors=[],
                                relevant_quote="Retry found this content",
                            )
                        ],
                    )

            mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
                side_effect=side_effect_ainvoke
            )

            agent = SearchAgent(mock_qdrant, mock_embedding, settings)
            result = await agent.run("specific query", filters={"year": 2020})

        # Should have gone through retry (at least 2 retrieve calls)
        assert result["retry_count"] >= 2
        assert len(result["answer"]) > 0

    async def test_max_retries_exhausted_synthesizes_empty(
        self, settings, mock_qdrant, mock_embedding
    ):
        """All searches return nothing after grading → synthesize with empty chunks."""
        mock_qdrant.hybrid_search = AsyncMock(return_value=[])

        with patch("backend.service.agent.init_chat_model") as mock_init:
            mock_llm = MagicMock()
            mock_init.return_value = mock_llm

            from backend.service.agent import GradingResult, QueryAnalysis

            call_count = 0

            async def side_effect_ainvoke(messages):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return QueryAnalysis(
                        search_queries=["nothing here"], year_filter=None
                    )
                else:
                    # grading with no chunks — shouldn't be called
                    return GradingResult(relevant_indices=[])

            mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
                side_effect=side_effect_ainvoke
            )

            agent = SearchAgent(mock_qdrant, mock_embedding, settings)
            result = await agent.run("nonexistent topic xyz")

        # Synthesize should produce the "no relevant info" message
        assert "couldn't find" in result["answer"].lower()
        assert result["citations"] == []

    async def test_query_analysis_failure_uses_raw_query(
        self, settings, mock_qdrant, mock_embedding
    ):
        """Query analysis LLM call fails → falls back to raw query."""
        fake_point = Mock()
        fake_point.id = "point-fallback"
        fake_point.score = 0.9
        fake_point.payload = {
            "chunk_id": "fallback-chunk",
            "text_content": "Fallback content",
            "document_title": "Fallback Doc",
            "document_year": 2021,
            "document_authors": ["Jones"],
            "document_url": "http://example.com/fallback",
        }
        mock_qdrant.hybrid_search = AsyncMock(return_value=[fake_point])

        with patch("backend.service.agent.init_chat_model") as mock_init:
            mock_llm = MagicMock()
            mock_init.return_value = mock_llm

            from backend.service.agent import GradingResult

            call_count = 0

            async def side_effect_ainvoke(messages):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # analyze_query FAILS
                    raise RuntimeError("LLM service unavailable")
                elif call_count == 2:
                    # grade_documents
                    return GradingResult(relevant_indices=[0])
                else:
                    # synthesize
                    return SynthesisResult(
                        answer="Answer from fallback",
                        citations=[
                            CitationRef(
                                source_number=1,
                                document_title=None,
                                document_url=None,
                                document_year=None,
                                document_authors=[],
                                relevant_quote="Fallback content",
                            )
                        ],
                    )

            mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
                side_effect=side_effect_ainvoke
            )

            agent = SearchAgent(mock_qdrant, mock_embedding, settings)
            raw_query = "my raw query about economics"
            result = await agent.run(raw_query)

        # Should have used the raw query as fallback
        assert result["search_queries"] == [raw_query]
        assert len(result["answer"]) > 0
