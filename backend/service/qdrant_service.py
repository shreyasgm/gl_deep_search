"""Async Qdrant client wrapper for the search service."""

from loguru import logger
from qdrant_client import AsyncQdrantClient, models

from backend.service.config import ServiceSettings


class QdrantService:
    """Thin wrapper around AsyncQdrantClient."""

    def __init__(self, settings: ServiceSettings) -> None:
        self._settings = settings
        self._client: AsyncQdrantClient | None = None

    async def connect(self) -> None:
        """Create the async client connection."""
        self._client = AsyncQdrantClient(
            url=self._settings.qdrant_url,
            api_key=self._settings.qdrant_api_key,
        )
        logger.info(f"Connected to Qdrant at {self._settings.qdrant_url}")

    async def close(self) -> None:
        """Close the client connection."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Qdrant connection closed")

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            raise RuntimeError("QdrantService not connected — call connect() first")
        return self._client

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    async def ensure_collection(
        self,
        name: str,
        vector_size: int,
        distance: models.Distance = models.Distance.COSINE,
    ) -> None:
        """Create the collection if it doesn't already exist."""
        exists = await self.client.collection_exists(name)
        if exists:
            logger.info(f"Collection '{name}' already exists")
            return
        await self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=distance,
            ),
        )
        logger.info(
            f"Created collection '{name}' (size={vector_size}, distance={distance})"
        )

    async def collection_info(self, name: str | None = None) -> models.CollectionInfo:
        """Return collection info."""
        name = name or self._settings.qdrant_collection
        return await self.client.get_collection(name)

    # ------------------------------------------------------------------
    # Point operations
    # ------------------------------------------------------------------

    async def upsert_points(
        self,
        collection: str,
        points: list[models.PointStruct],
        batch_size: int = 100,
    ) -> None:
        """Upsert points in batches."""
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await self.client.upsert(
                collection_name=collection,
                points=batch,
            )
        logger.info(f"Upserted {len(points)} points into '{collection}'")

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        filters: models.Filter | None = None,
    ) -> list[models.ScoredPoint]:
        """Run a vector similarity search."""
        response = await self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
            query_filter=filters,
            with_payload=True,
        )
        return response.points

    async def get_by_document_id(
        self,
        collection: str,
        document_id: str,
    ) -> list[models.Record]:
        """Retrieve all points belonging to a given document."""
        result = await self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            ),
            with_payload=True,
            with_vectors=False,
            limit=1000,
        )
        return result[0]  # (records, next_offset)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Return True if Qdrant is reachable."""
        try:
            await self.client.get_collections()
            return True
        except Exception:
            return False
