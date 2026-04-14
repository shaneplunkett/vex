"""Vex configuration loaded from environment variables."""

from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Vex server configuration."""

    model_config = SettingsConfigDict(
        env_prefix="VEX_BRAIN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server — default to loopback; deployments override via env to 0.0.0.0
    host: str = Field(default="127.0.0.1", description="Host to bind the MCP server to")
    port: int = Field(default=8000, description="Port to bind the MCP server to")

    # Database
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/vex_brain",
        description="Postgres connection string",
    )
    db_pool_min: int = Field(default=2, ge=1, description="Minimum asyncpg pool size")
    db_pool_max: int = Field(default=10, ge=1, description="Maximum asyncpg pool size")

    # Embedding
    openai_api_key: SecretStr | None = Field(default=None, description="OpenAI API key for embeddings")
    embedding_model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")
    embedding_dimensions: int = Field(default=1536, description="Embedding vector dimensions")

    # Extraction
    anthropic_api_key: SecretStr | None = Field(default=None, description="Anthropic API key for extraction")
    extraction_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Claude model for entity extraction",
    )
    maintenance_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Claude model for maintenance operations (reconsolidate, type validation)",
    )

    # Speaker names (used by coreference resolution and singleton enforcement)
    human_speaker: str = Field(default="User", description="Name for the human speaker in conversations")
    assistant_speaker: str = Field(default="Assistant", description="Name for the assistant speaker in conversations")

    # Pipeline
    pipeline_mode: Literal["api", "agent"] = Field(
        default="api",
        description="Pipeline mode: 'api' runs extraction, 'agent' stops at embedded",
    )
    pipeline_concurrency: int = Field(default=3, ge=1, description="Max concurrent pipeline tasks")
    extraction_concurrency: int = Field(default=5, ge=1, description="Max concurrent extraction API calls")

    # Validator thresholds — set to 1.0 for calibration (everything to review queue)
    validator_existing_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Min confidence to auto-apply existing entity match"
    )
    validator_new_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0, description="Min confidence to auto-create new entity"
    )
    validator_fuzzy_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0, description="pg_trgm similarity threshold for fuzzy name matching"
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )

    @model_validator(mode="after")
    def validate_pool_bounds(self) -> "Settings":
        """Ensure db_pool_max >= db_pool_min."""
        if self.db_pool_max < self.db_pool_min:
            msg = f"db_pool_max ({self.db_pool_max}) must be >= db_pool_min ({self.db_pool_min})"
            raise ValueError(msg)
        return self


_settings: Settings | None = None


def get_settings() -> Settings:
    """Lazy singleton — avoids import-time env reads, testable via reset."""
    global _settings  # noqa: PLW0603
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings singleton — for testing only."""
    global _settings  # noqa: PLW0603
    _settings = None
