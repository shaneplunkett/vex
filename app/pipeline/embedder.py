"""Embedding generation — generates vector embeddings for chunks via OpenAI API.

Uses text-embedding-3-small (1536 dimensions) with batched requests,
exponential backoff on failures, and per-chunk error isolation.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    import openai

logger = structlog.get_logger()

_MODEL = "text-embedding-3-small"
_DIMENSIONS = 1536
_BATCH_SIZE = 100
_MAX_RETRIES = 3
_BASE_DELAY = 1.0  # seconds


def _get_client() -> openai.AsyncOpenAI:
    """Create an async OpenAI client using configured API key."""
    import openai as openai_mod

    from app.config import get_settings

    settings = get_settings()
    api_key = settings.openai_api_key
    if api_key is None:
        msg = "VEX_BRAIN_OPENAI_API_KEY is required for embedding generation"
        raise RuntimeError(msg)

    return openai_mod.AsyncOpenAI(api_key=api_key.get_secret_value(), timeout=30.0)


async def embed_texts(texts: list[str]) -> list[list[float] | None]:
    """Embed a list of texts, returning vectors in the same order.

    Returns None for texts that failed to embed. Batches into groups
    of _BATCH_SIZE and retries on transient failures.
    """
    client = _get_client()
    results: list[list[float] | None] = [None] * len(texts)

    for batch_start in range(0, len(texts), _BATCH_SIZE):
        batch_end = min(batch_start + _BATCH_SIZE, len(texts))
        batch_texts = texts[batch_start:batch_end]

        vectors = await _embed_batch_with_retry(client, batch_texts)

        for i, vec in enumerate(vectors):
            results[batch_start + i] = vec

    return results


async def _embed_batch_with_retry(
    client: openai.AsyncOpenAI,
    texts: list[str],
) -> list[list[float] | None]:
    """Embed a single batch with exponential backoff retry.

    Returns a list of vectors (or None for failures) matching input order.
    """
    last_error: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            start = time.monotonic()
            from app.config import get_settings

            settings = get_settings()
            response = await client.embeddings.create(
                model=settings.embedding_model,
                input=texts,
                dimensions=settings.embedding_dimensions,
            )
            elapsed = time.monotonic() - start

            logger.info(
                "embedder.batch_complete",
                batch_size=len(texts),
                latency_ms=round(elapsed * 1000),
            )

            # Response data is ordered by index
            vectors: list[list[float] | None] = [None] * len(texts)
            for item in response.data:
                vectors[item.index] = item.embedding

            return vectors

        except Exception as e:
            last_error = e
            # Retry on transient errors (429, 500, timeout)
            if _is_retryable(e) and attempt < _MAX_RETRIES - 1:
                delay = _BASE_DELAY * (2**attempt)
                logger.warning(
                    "embedder.retry",
                    attempt=attempt + 1,
                    delay_s=delay,
                    error=str(e),
                )
                await asyncio.sleep(delay)
            else:
                # Non-retryable or final attempt — log and exit
                break

    logger.error(
        "embedder.batch_failed",
        batch_size=len(texts),
        retries=_MAX_RETRIES,
        error=str(last_error),
    )
    return [None] * len(texts)


def _is_retryable(error: Exception) -> bool:
    """Check if an error is transient and worth retrying."""
    import openai as openai_mod

    if isinstance(error, openai_mod.RateLimitError):
        return True
    if isinstance(error, openai_mod.APIStatusError) and error.status_code >= 500:
        return True
    return isinstance(error, (openai_mod.APITimeoutError, openai_mod.APIConnectionError))
