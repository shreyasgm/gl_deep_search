"""LangGraph search agent with hybrid BM25 retrieval and LLM synthesis."""

import asyncio
from typing import Any, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import BaseModel, Field
from qdrant_client import models as qdrant_models

from backend.service.config import ServiceSettings
from backend.service.embedding_service import EmbeddingService
from backend.service.qdrant_service import QdrantService

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

_QUERY_ANALYSIS_PROMPT = (
    "You are a search query optimizer for a research "
    "document database. Given a user query, generate "
    "1-3 search queries optimized for embedding-based "
    "semantic search over academic/policy research "
    "documents from the Growth Lab at Harvard University. "
    "Also extract a year filter if the user mentions a "
    "specific year. Keep queries concise and focused on "
    "key concepts."
)

_GRADING_PROMPT = (
    "You are a relevance grader for a research document "
    "search system. Given a user query and numbered "
    "document chunks, return the indices of chunks that "
    "are relevant to answering the query. Be inclusive — "
    "keep chunks that provide useful context, even if "
    "they don't directly answer the query. Only exclude "
    "clearly irrelevant chunks."
)

_SYNTHESIS_PROMPT = (
    "You are a research assistant for the Growth Lab at "
    "Harvard University. Answer the user's question using "
    "ONLY the provided document context. Cite sources "
    "using [n] notation matching the source numbers. "
    "If the context doesn't fully answer the question, "
    "say so honestly. Do not make up information beyond "
    "what the sources provide. Be concise but thorough."
)


# ---------------------------------------------------------------------------
# Pydantic models for LLM structured output
# ---------------------------------------------------------------------------


class QueryAnalysis(BaseModel):
    """LLM output: optimized search queries + extracted filters."""

    search_queries: list[str] = Field(
        description="1-3 search queries optimized for embedding-based retrieval"
    )
    year_filter: int | None = Field(
        default=None,
        description="Year extracted from the query, if any",
    )


class GradingResult(BaseModel):
    """LLM output: indices of relevant chunks."""

    relevant_indices: list[int] = Field(
        description="0-based indices of chunks that are relevant to the query"
    )


class CitationRef(BaseModel):
    """A single citation in the synthesized answer."""

    source_number: int = Field(
        description="The [n] reference number used in the answer"
    )
    document_title: str | None = None
    document_url: str | None = None
    document_year: int | None = None
    document_authors: list[str] = Field(default_factory=list)
    relevant_quote: str = Field(
        description="A short verbatim quote from the source that supports the citation"
    )


class SynthesisResult(BaseModel):
    """LLM output: synthesized answer with citations."""

    answer: str = Field(description="The answer with [n] citation references")
    citations: list[CitationRef] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    query: str
    user_filters: dict
    top_k: int

    search_queries: list[str]
    extracted_filters: dict

    chunks: list[dict]
    retry_count: int

    answer: str
    citations: list[dict]


# ---------------------------------------------------------------------------
# SearchAgent
# ---------------------------------------------------------------------------


class SearchAgent:
    """LangGraph agent for query understanding, hybrid retrieval, and synthesis."""

    def __init__(
        self,
        qdrant: QdrantService,
        embedding: EmbeddingService,
        settings: ServiceSettings,
    ) -> None:
        self.qdrant = qdrant
        self.embedding = embedding
        self.settings = settings
        self.llm = init_chat_model(
            settings.agent_model,
            model_provider="anthropic",
            temperature=0,
            api_key=settings.anthropic_api_key,
        )
        self.graph = self._build_graph()

    async def run(
        self,
        query: str,
        filters: dict | None = None,
        top_k: int = 10,
    ) -> dict:
        """Execute the full agent pipeline and return the final state."""
        initial_state: AgentState = {
            "query": query,
            "user_filters": filters or {},
            "top_k": top_k,
            "search_queries": [],
            "extracted_filters": {},
            "chunks": [],
            "retry_count": 0,
            "answer": "",
            "citations": [],
        }
        return await self.graph.ainvoke(initial_state)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        graph = StateGraph(AgentState)

        graph.add_node("analyze_query", self._analyze_query)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("grade_documents", self._grade_documents)
        graph.add_node("synthesize", self._synthesize)

        graph.add_edge(START, "analyze_query")
        graph.add_edge("analyze_query", "retrieve")
        graph.add_edge("retrieve", "grade_documents")
        graph.add_conditional_edges(
            "grade_documents",
            self._should_retry,
            {"retrieve": "retrieve", "synthesize": "synthesize"},
        )
        graph.add_edge("synthesize", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Node: analyze_query
    # ------------------------------------------------------------------

    async def _analyze_query(self, state: AgentState) -> dict:
        query = state["query"]
        logger.info(f"Analyzing query: {query!r}")

        try:
            llm_structured = self.llm.with_structured_output(QueryAnalysis)
            result = await llm_structured.ainvoke(
                [
                    SystemMessage(content=_QUERY_ANALYSIS_PROMPT),
                    HumanMessage(content=query),
                ]
            )
            assert isinstance(result, QueryAnalysis)
            search_queries = result.search_queries or [query]
            extracted_filters = {}
            if result.year_filter is not None:
                extracted_filters["year"] = result.year_filter
            logger.info(
                f"Query analysis: queries={search_queries}, filters={extracted_filters}"
            )
        except Exception:
            logger.warning("Query analysis failed, using raw query as fallback")
            search_queries = [query]
            extracted_filters = {}

        return {
            "search_queries": search_queries,
            "extracted_filters": extracted_filters,
        }

    # ------------------------------------------------------------------
    # Node: retrieve
    # ------------------------------------------------------------------

    async def _retrieve(self, state: AgentState) -> dict:
        queries = state["search_queries"]
        top_k = state["top_k"]
        retry_count = state["retry_count"]

        # Merge filters: user-provided take precedence over LLM-extracted
        merged_filters = {**state["extracted_filters"], **state["user_filters"]}

        # On retry, drop all filters and use just the raw query
        if retry_count > 0:
            logger.info("Retry: broadening search — dropping filters, using raw query")
            queries = [state["query"]]
            merged_filters = {}

        # Build Qdrant filter
        qdrant_filter = self._build_filter(merged_filters)

        logger.info(
            f"Retrieving (attempt {retry_count + 1}): "
            f"{len(queries)} queries, top_k={top_k}, filters={merged_filters}"
        )

        # Run hybrid search for each query in parallel
        all_points: dict[str, dict] = {}  # chunk_id -> best result

        async def _search_one(q: str):
            dense_vec = await self.embedding.embed_query(q)
            sparse_vec = self.embedding.sparse_embed_query(q)
            return await self.qdrant.hybrid_search(
                collection=self.settings.qdrant_collection,
                dense_vector=dense_vec,
                sparse_vector=sparse_vec,
                top_k=top_k,
                filters=qdrant_filter,
            )

        search_results = await asyncio.gather(*[_search_one(q) for q in queries])

        for points in search_results:
            for pt in points:
                payload = pt.payload or {}
                cid = payload.get("chunk_id", str(pt.id))
                existing = all_points.get(cid)
                if existing is None or pt.score > existing["score"]:
                    all_points[cid] = {
                        "chunk_id": cid,
                        "score": pt.score,
                        **payload,
                    }

        # Sort by score descending, take top_k
        chunks = sorted(all_points.values(), key=lambda c: c["score"], reverse=True)[
            :top_k
        ]
        logger.info(f"Retrieved {len(chunks)} unique chunks")

        return {"chunks": chunks, "retry_count": retry_count + 1}

    # ------------------------------------------------------------------
    # Node: grade_documents
    # ------------------------------------------------------------------

    async def _grade_documents(self, state: AgentState) -> dict:
        chunks = state["chunks"]
        query = state["query"]

        if not chunks:
            logger.info("No chunks to grade")
            return {"chunks": []}

        # Build numbered context for the LLM
        context_lines = []
        for i, chunk in enumerate(chunks):
            text = chunk.get("text_content", "")
            title = chunk.get("document_title", "Unknown")
            year = chunk.get("document_year", "")
            context_lines.append(f"[{i}] ({title}, {year}) {text[:1500]}")

        context_block = "\n\n".join(context_lines)

        try:
            llm_structured = self.llm.with_structured_output(GradingResult)
            result = await llm_structured.ainvoke(
                [
                    SystemMessage(content=_GRADING_PROMPT),
                    HumanMessage(
                        content=(
                            f"Query: {query}\n\n"
                            f"Document chunks:\n{context_block}\n\n"
                            "Return the indices of relevant chunks."
                        )
                    ),
                ]
            )
            assert isinstance(result, GradingResult)
            valid_indices = [i for i in result.relevant_indices if 0 <= i < len(chunks)]
            graded = [chunks[i] for i in valid_indices]
            logger.info(f"Grading: kept {len(graded)}/{len(chunks)} chunks")
        except Exception:
            logger.warning("Grading failed, keeping all chunks")
            graded = chunks

        return {"chunks": graded}

    # ------------------------------------------------------------------
    # Conditional edge: should_retry
    # ------------------------------------------------------------------

    def _should_retry(self, state: AgentState) -> str:
        if not state["chunks"] and state["retry_count"] < 2:
            logger.info(
                "No relevant chunks after grading — retrying with broader search"
            )
            return "retrieve"
        return "synthesize"

    # ------------------------------------------------------------------
    # Node: synthesize
    # ------------------------------------------------------------------

    async def _synthesize(self, state: AgentState) -> dict:
        chunks = state["chunks"]
        query = state["query"]

        if not chunks:
            return {
                "answer": (
                    "I couldn't find relevant information in the Growth Lab "
                    "documents to answer this query."
                ),
                "citations": [],
            }

        # Build numbered context
        context_lines = []
        for i, chunk in enumerate(chunks, start=1):
            text = chunk.get("text_content", "")
            title = chunk.get("document_title", "Unknown")
            year = chunk.get("document_year", "")
            authors = ", ".join(chunk.get("document_authors", []))
            context_lines.append(
                f'[{i}] "{text[:2000]}"\n    (Source: {title}, {authors}, {year})'
            )

        context_block = "\n\n".join(context_lines)

        try:
            llm_structured = self.llm.with_structured_output(SynthesisResult)
            result = await llm_structured.ainvoke(
                [
                    SystemMessage(content=_SYNTHESIS_PROMPT),
                    HumanMessage(
                        content=(
                            f"Question: {query}\n\n"
                            f"Sources:\n{context_block}\n\n"
                            "Answer the question with citations."
                        )
                    ),
                ]
            )
            assert isinstance(result, SynthesisResult)

            citations = [
                {
                    "source_number": c.source_number,
                    "document_title": c.document_title,
                    "document_url": c.document_url,
                    "document_year": c.document_year,
                    "document_authors": c.document_authors,
                    "relevant_quote": c.relevant_quote,
                }
                for c in result.citations
            ]

            # Enrich citations with metadata from chunks
            for citation in citations:
                idx = int(citation["source_number"]) - 1
                if 0 <= idx < len(chunks):
                    chunk = chunks[idx]
                    if not citation["document_title"]:
                        citation["document_title"] = chunk.get("document_title")
                    if not citation["document_url"]:
                        citation["document_url"] = chunk.get("document_url")
                    if not citation["document_year"]:
                        citation["document_year"] = chunk.get("document_year")
                    if not citation["document_authors"]:
                        citation["document_authors"] = chunk.get("document_authors", [])

            logger.info(
                f"Synthesis complete: {len(result.answer)} chars, "
                f"{len(citations)} citations"
            )
            return {"answer": result.answer, "citations": citations}

        except Exception:
            logger.exception("Synthesis failed")
            return {
                "answer": "An error occurred while generating the answer.",
                "citations": [],
            }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_filter(
        filters: dict,
    ) -> qdrant_models.Filter | None:
        conditions: list[qdrant_models.FieldCondition] = []

        if year := filters.get("year"):
            conditions.append(
                qdrant_models.FieldCondition(
                    key="document_year",
                    match=qdrant_models.MatchValue(value=year),
                )
            )
        if doc_id := filters.get("document_id"):
            conditions.append(
                qdrant_models.FieldCondition(
                    key="document_id",
                    match=qdrant_models.MatchValue(value=doc_id),
                )
            )

        return qdrant_models.Filter(must=conditions) if conditions else None
