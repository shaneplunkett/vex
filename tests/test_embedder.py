"""Tests for embedding generation."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from app.pipeline.embedder import (
    _DIMENSIONS,
    _MODEL,
    _is_retryable,
    embed_texts,
)


@dataclass
class FakeEmbeddingItem:
    """Mock OpenAI embedding response item."""

    index: int
    embedding: list[float]


@dataclass
class FakeEmbeddingResponse:
    """Mock OpenAI embedding response."""

    data: list[FakeEmbeddingItem]


def _make_response(texts: list[str]) -> FakeEmbeddingResponse:
    """Create a mock embedding response with deterministic fake vectors."""
    return FakeEmbeddingResponse(
        data=[FakeEmbeddingItem(index=i, embedding=[float(i)] * _DIMENSIONS) for i in range(len(texts))]
    )


@pytest.fixture
def mock_openai_client() -> AsyncMock:
    """Mock OpenAI async client."""
    client = AsyncMock()
    client.embeddings.create = AsyncMock(side_effect=lambda **kwargs: _make_response(kwargs["input"]))
    return client


@pytest.fixture(autouse=True)
def _patch_client(mock_openai_client: AsyncMock) -> None:
    """Patch _get_client to return our mock."""
    with patch("app.pipeline.embedder._get_client", return_value=mock_openai_client):
        yield


async def test_embed_single_text(mock_openai_client: AsyncMock) -> None:
    """Single text returns a vector."""
    results = await embed_texts(["hello world"])
    assert len(results) == 1
    assert results[0] is not None
    assert len(results[0]) == _DIMENSIONS


async def test_embed_multiple_texts(mock_openai_client: AsyncMock) -> None:
    """Multiple texts return vectors in order."""
    texts = ["hello", "world", "test"]
    results = await embed_texts(texts)
    assert len(results) == 3
    assert all(v is not None for v in results)
    # Verify ordering — each vector starts with its index
    assert results[0][0] == 0.0
    assert results[1][0] == 1.0
    assert results[2][0] == 2.0


async def test_embed_batching(mock_openai_client: AsyncMock) -> None:
    """Texts exceeding batch size are split into multiple API calls."""
    texts = [f"text {i}" for i in range(150)]
    results = await embed_texts(texts)
    assert len(results) == 150
    # Should have made 2 API calls (100 + 50)
    assert mock_openai_client.embeddings.create.call_count == 2


async def test_embed_empty_list() -> None:
    """Empty input returns empty output."""
    results = await embed_texts([])
    assert results == []


async def test_retry_on_rate_limit(mock_openai_client: AsyncMock) -> None:
    """Retries on 429 rate limit errors."""
    import openai

    rate_limit_error = openai.RateLimitError(
        message="rate limited",
        response=AsyncMock(status_code=429, headers={}),
        body=None,
    )
    mock_openai_client.embeddings.create = AsyncMock(side_effect=[rate_limit_error, _make_response(["test"])])

    with patch("app.pipeline.embedder._BASE_DELAY", 0.01):
        results = await embed_texts(["test"])

    assert results[0] is not None
    assert mock_openai_client.embeddings.create.call_count == 2


async def test_retry_exhausted_returns_none(mock_openai_client: AsyncMock) -> None:
    """All retries exhausted returns None for all texts."""
    import openai

    error = openai.RateLimitError(
        message="rate limited",
        response=AsyncMock(status_code=429, headers={}),
        body=None,
    )
    mock_openai_client.embeddings.create = AsyncMock(side_effect=error)

    with patch("app.pipeline.embedder._BASE_DELAY", 0.01):
        results = await embed_texts(["test"])

    assert results == [None]


async def test_non_retryable_error_fails_immediately(mock_openai_client: AsyncMock) -> None:
    """Non-retryable errors don't retry."""
    import openai

    error = openai.AuthenticationError(
        message="bad key",
        response=AsyncMock(status_code=401, headers={}),
        body=None,
    )
    mock_openai_client.embeddings.create = AsyncMock(side_effect=error)

    results = await embed_texts(["test"])

    assert results == [None]
    assert mock_openai_client.embeddings.create.call_count == 1


def test_is_retryable_rate_limit() -> None:
    """Rate limit errors are retryable."""
    import openai

    error = openai.RateLimitError(
        message="rate limited",
        response=AsyncMock(status_code=429, headers={}),
        body=None,
    )
    assert _is_retryable(error) is True


def test_is_retryable_server_error() -> None:
    """500+ errors are retryable."""
    import openai

    error = openai.APIStatusError(
        message="server error",
        response=AsyncMock(status_code=500, headers={}),
        body=None,
    )
    assert _is_retryable(error) is True


def test_is_retryable_timeout() -> None:
    """Timeout errors are retryable."""
    import openai

    error = openai.APITimeoutError(request=AsyncMock())
    assert _is_retryable(error) is True


def test_is_retryable_auth_error() -> None:
    """Auth errors are NOT retryable."""
    import openai

    error = openai.AuthenticationError(
        message="bad key",
        response=AsyncMock(status_code=401, headers={}),
        body=None,
    )
    assert _is_retryable(error) is False


async def test_model_identifier() -> None:
    """Verify the model identifier matches expected value."""
    assert _MODEL == "text-embedding-3-small"
    assert _DIMENSIONS == 1536
