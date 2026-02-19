"""HTTP client for the Growth Lab Deep Search backend API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class APIError:
    """Represents an API call failure."""

    message: str
    status_code: int | None = None


@dataclass
class HealthStatus:
    """Parsed health check response."""

    healthy: bool
    qdrant_connected: bool
    collection: str
    points_count: int


@dataclass
class ChunkResult:
    """A single chunk from vector similarity search."""

    chunk_id: str
    document_id: str
    text_content: str
    score: float
    page_numbers: list[int]
    section_title: str | None
    chunk_index: int
    token_count: int
    document_title: str | None
    document_authors: list[str]
    document_year: int | None
    document_abstract: str | None
    document_url: str | None


@dataclass
class ChunkSearchResponse:
    """Parsed response from /search/chunks."""

    query: str
    results: list[ChunkResult]
    total_results: int


@dataclass
class Citation:
    """A single citation from agent search."""

    source_number: int
    document_title: str | None
    document_url: str | None
    document_year: int | None
    document_authors: list[str]
    relevant_quote: str


@dataclass
class AgentSearchResponse:
    """Parsed response from /search."""

    query: str
    answer: str
    citations: list[Citation]
    search_queries_used: list[str]
    chunks_retrieved: int


class SearchClient:
    """Synchronous client for the Growth Lab Deep Search API."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def health(self) -> HealthStatus | APIError:
        """Check backend health status."""
        try:
            resp = self._session.get(f"{self.base_url}/health", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return HealthStatus(
                healthy=data["status"] == "healthy",
                qdrant_connected=data["qdrant_connected"],
                collection=data["collection"],
                points_count=data["points_count"],
            )
        except requests.ConnectionError:
            return APIError("Cannot connect to backend. Is the server running?")
        except requests.Timeout:
            return APIError("Health check timed out.", status_code=None)
        except requests.HTTPError as e:
            return APIError(
                _extract_detail(e.response), status_code=e.response.status_code
            )

    def agent_search(
        self,
        query: str,
        top_k: int = 10,
        year: int | None = None,
    ) -> AgentSearchResponse | APIError:
        """Run agent search with LLM synthesis and citations."""
        body = self._build_search_body(query, top_k, year)
        try:
            resp = self._session.post(f"{self.base_url}/search", json=body, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return AgentSearchResponse(
                query=data["query"],
                answer=data["answer"],
                citations=[
                    Citation(
                        source_number=c["source_number"],
                        document_title=c.get("document_title"),
                        document_url=c.get("document_url"),
                        document_year=c.get("document_year"),
                        document_authors=c.get("document_authors", []),
                        relevant_quote=c["relevant_quote"],
                    )
                    for c in data.get("citations", [])
                ],
                search_queries_used=data.get("search_queries_used", []),
                chunks_retrieved=data.get("chunks_retrieved", 0),
            )
        except requests.ConnectionError:
            return APIError("Cannot connect to backend. Is the server running?")
        except requests.Timeout:
            return APIError(
                "Search timed out. Try a simpler query or reduce the number of results."
            )
        except requests.HTTPError as e:
            return APIError(
                _extract_detail(e.response), status_code=e.response.status_code
            )

    def chunk_search(
        self,
        query: str,
        top_k: int = 10,
        year: int | None = None,
    ) -> ChunkSearchResponse | APIError:
        """Run raw vector similarity search."""
        body = self._build_search_body(query, top_k, year)
        try:
            resp = self._session.post(
                f"{self.base_url}/search/chunks", json=body, timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
            return ChunkSearchResponse(
                query=data["query"],
                results=[
                    ChunkResult(
                        chunk_id=r["chunk_id"],
                        document_id=r["document_id"],
                        text_content=r["text_content"],
                        score=r["score"],
                        page_numbers=r.get("page_numbers", []),
                        section_title=r.get("section_title"),
                        chunk_index=r.get("chunk_index", 0),
                        token_count=r.get("token_count", 0),
                        document_title=r.get("document_title"),
                        document_authors=r.get("document_authors", []),
                        document_year=r.get("document_year"),
                        document_abstract=r.get("document_abstract"),
                        document_url=r.get("document_url"),
                    )
                    for r in data.get("results", [])
                ],
                total_results=data.get("total_results", 0),
            )
        except requests.ConnectionError:
            return APIError("Cannot connect to backend. Is the server running?")
        except requests.Timeout:
            return APIError(
                "Search timed out. Try a simpler query or reduce the number of results."
            )
        except requests.HTTPError as e:
            return APIError(
                _extract_detail(e.response), status_code=e.response.status_code
            )

    @staticmethod
    def _build_search_body(query: str, top_k: int, year: int | None) -> dict[str, Any]:
        body: dict[str, Any] = {"query": query, "top_k": top_k}
        if year is not None:
            body["filters"] = {"year": year}
        return body


def _extract_detail(response: requests.Response | None) -> str:
    """Extract error detail from a FastAPI error response."""
    if response is None:
        return "Unknown error"
    try:
        data = response.json()
        return data.get("detail", f"HTTP {response.status_code}")
    except (ValueError, KeyError):
        return f"HTTP {response.status_code}: {response.text[:200]}"
