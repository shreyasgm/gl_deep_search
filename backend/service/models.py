"""Pydantic v2 request/response models for the search API."""

from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    """Optional filters to narrow search results."""

    year: int | None = None
    document_id: str | None = None


class SearchRequest(BaseModel):
    """Incoming search request."""

    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=50)
    filters: SearchFilters | None = None


class ChunkResult(BaseModel):
    """A single chunk returned from vector search."""

    chunk_id: str
    document_id: str
    text_content: str
    score: float
    page_numbers: list[int]
    section_title: str | None = None
    chunk_index: int
    token_count: int

    # Document-level metadata
    document_title: str | None = None
    document_authors: list[str] = Field(default_factory=list)
    document_year: int | None = None
    document_abstract: str | None = None
    document_url: str | None = None


class SearchResponse(BaseModel):
    """Response wrapper for search results."""

    query: str
    results: list[ChunkResult]
    total_results: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    qdrant_connected: bool
    collection: str
    points_count: int
