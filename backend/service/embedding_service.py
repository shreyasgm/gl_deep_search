"""Async wrapper for query-time embedding via OpenAI + BM25 sparse via fastembed."""

from fastembed import SparseTextEmbedding
from loguru import logger
from openai import AsyncOpenAI
from qdrant_client import models as qdrant_models

from backend.service.config import ServiceSettings


class EmbeddingService:
    """Dense (OpenAI) + sparse (BM25) embedding client."""

    def __init__(self, settings: ServiceSettings) -> None:
        self._settings = settings
        self._client: AsyncOpenAI | None = None
        self._sparse_model: SparseTextEmbedding | None = None

    def initialize(self) -> None:
        """Create the AsyncOpenAI client and load the BM25 sparse model."""
        self._client = AsyncOpenAI(api_key=self._settings.openai_api_key)
        self._sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        logger.info("BM25 sparse embedding model loaded")

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client:
            await self._client.close()
            self._client = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            raise RuntimeError(
                "EmbeddingService not initialized — call initialize() first"
            )
        return self._client

    @property
    def sparse_model(self) -> SparseTextEmbedding:
        if self._sparse_model is None:
            raise RuntimeError(
                "EmbeddingService not initialized — call initialize() first"
            )
        return self._sparse_model

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string and return the dense vector."""
        response = await self.client.embeddings.create(
            model=self._settings.embedding_model,
            input=[text],
        )
        return response.data[0].embedding

    def sparse_embed_query(self, text: str) -> qdrant_models.SparseVector:
        """BM25 sparse embedding for a single query (sync, CPU-only, fast)."""
        results = list(self.sparse_model.query_embed(text))
        return qdrant_models.SparseVector(
            indices=results[0].indices.tolist(),
            values=results[0].values.tolist(),
        )

    def sparse_embed_documents(
        self, texts: list[str]
    ) -> list[qdrant_models.SparseVector]:
        """BM25 sparse embeddings for a batch of documents."""
        results = list(self.sparse_model.embed(texts))
        return [
            qdrant_models.SparseVector(
                indices=r.indices.tolist(),
                values=r.values.tolist(),
            )
            for r in results
        ]
