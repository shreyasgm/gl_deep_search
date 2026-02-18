"""Async wrapper for query-time embedding via OpenAI."""

from openai import AsyncOpenAI

from backend.service.config import ServiceSettings


class EmbeddingService:
    """Thin async client for generating query embeddings."""

    def __init__(self, settings: ServiceSettings) -> None:
        self._settings = settings
        self._client: AsyncOpenAI | None = None

    def initialize(self) -> None:
        """Create the AsyncOpenAI client."""
        self._client = AsyncOpenAI(api_key=self._settings.openai_api_key)

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

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string and return the vector."""
        response = await self.client.embeddings.create(
            model=self._settings.embedding_model,
            input=[text],
        )
        return response.data[0].embedding
