"""Service configuration using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceSettings(BaseSettings):
    """Configuration for the search service."""

    model_config = SettingsConfigDict(
        env_file="backend/etl/.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None

    # LLM providers
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # Embedding settings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Qdrant collection
    qdrant_collection: str = "gl_chunks"

    # Search defaults
    default_top_k: int = 10
    max_top_k: int = 50

    # Agent
    agent_model: str = "claude-sonnet-4-20250514"

    # Logging
    log_level: str = "INFO"


def get_settings() -> ServiceSettings:
    """Return a ServiceSettings instance."""
    return ServiceSettings()
