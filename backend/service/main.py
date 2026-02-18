"""FastAPI application for the Growth Lab Deep Search API."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from loguru import logger
from qdrant_client import models as qdrant_models

from backend.service.config import ServiceSettings
from backend.service.embedding_service import EmbeddingService
from backend.service.models import (
    ChunkResult,
    HealthResponse,
    SearchFilters,
    SearchRequest,
    SearchResponse,
)
from backend.service.qdrant_service import QdrantService

# ---------------------------------------------------------------------------
# Module-level singletons (populated during lifespan)
# ---------------------------------------------------------------------------
_settings: ServiceSettings | None = None
_qdrant: QdrantService | None = None
_embedding: EmbeddingService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup / shutdown lifecycle."""
    global _settings, _qdrant, _embedding  # noqa: PLW0603

    _settings = ServiceSettings()
    logger.info("ServiceSettings loaded")

    _qdrant = QdrantService(_settings)
    await _qdrant.connect()

    # Ensure payload indexes exist for filterable fields (idempotent)
    collection = _settings.qdrant_collection
    for field, schema in [
        ("document_year", qdrant_models.PayloadSchemaType.INTEGER),
        ("document_id", qdrant_models.PayloadSchemaType.KEYWORD),
    ]:
        await _qdrant.client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=schema,
        )
    logger.info(f"Payload indexes ensured for '{collection}'")

    _embedding = EmbeddingService(_settings)
    _embedding.initialize()
    logger.info("EmbeddingService initialized")

    yield

    await _embedding.close()
    await _qdrant.close()
    logger.info("Services shut down")


app = FastAPI(
    title="Growth Lab Deep Search",
    description="Vector search API over Growth Lab research documents.",
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------


def get_service_settings() -> ServiceSettings:
    assert _settings is not None  # noqa: S101
    return _settings


def get_qdrant() -> QdrantService:
    assert _qdrant is not None  # noqa: S101
    return _qdrant


def get_embedding() -> EmbeddingService:
    assert _embedding is not None  # noqa: S101
    return _embedding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_qdrant_filter(
    filters: SearchFilters | None,
) -> qdrant_models.Filter | None:
    """Convert API-level filters to a Qdrant Filter object."""
    if filters is None:
        return None

    conditions: list[qdrant_models.FieldCondition] = []

    if filters.year is not None:
        conditions.append(
            qdrant_models.FieldCondition(
                key="document_year",
                match=qdrant_models.MatchValue(value=filters.year),
            )
        )
    if filters.document_id is not None:
        conditions.append(
            qdrant_models.FieldCondition(
                key="document_id",
                match=qdrant_models.MatchValue(value=filters.document_id),
            )
        )

    return qdrant_models.Filter(must=conditions) if conditions else None


def _scored_point_to_chunk_result(
    point: qdrant_models.ScoredPoint,
) -> ChunkResult:
    """Map a Qdrant ScoredPoint to our API ChunkResult model."""
    payload = point.payload or {}
    return ChunkResult(
        chunk_id=str(point.id),
        document_id=payload.get("document_id", ""),
        text_content=payload.get("text_content", ""),
        score=point.score,
        page_numbers=payload.get("page_numbers", []),
        section_title=payload.get("section_title"),
        chunk_index=payload.get("chunk_index", 0),
        token_count=payload.get("token_count", 0),
        document_title=payload.get("document_title"),
        document_authors=payload.get("document_authors"),
        document_year=payload.get("document_year"),
        document_abstract=payload.get("document_abstract"),
        document_url=payload.get("document_url"),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/search/chunks", response_model=SearchResponse)
async def search_chunks(
    request: SearchRequest,
    qdrant: Annotated[QdrantService, Depends(get_qdrant)],
    embedding: Annotated[EmbeddingService, Depends(get_embedding)],
    settings: Annotated[ServiceSettings, Depends(get_service_settings)],
) -> SearchResponse:
    """Perform vector similarity search over document chunks."""
    try:
        top_k = min(request.top_k, settings.max_top_k)

        query_vector = await embedding.embed_query(request.query)

        qdrant_filter = _build_qdrant_filter(request.filters)

        points = await qdrant.search(
            collection=settings.qdrant_collection,
            query_vector=query_vector,
            top_k=top_k,
            filters=qdrant_filter,
        )

        results = [_scored_point_to_chunk_result(p) for p in points]

        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
        )
    except Exception as exc:
        logger.error(f"Search failed: {exc}")
        raise HTTPException(status_code=500, detail="Search failed") from exc


@app.get("/health", response_model=HealthResponse)
async def health(
    qdrant: Annotated[QdrantService, Depends(get_qdrant)],
    settings: Annotated[ServiceSettings, Depends(get_service_settings)],
) -> HealthResponse:
    """Check service health and Qdrant connectivity."""
    connected = await qdrant.health_check()
    points_count = 0

    if connected:
        try:
            info = await qdrant.collection_info(settings.qdrant_collection)
            points_count = info.points_count or 0
        except Exception:
            connected = False

    return HealthResponse(
        status="healthy" if connected else "unhealthy",
        qdrant_connected=connected,
        collection=settings.qdrant_collection,
        points_count=points_count,
    )
