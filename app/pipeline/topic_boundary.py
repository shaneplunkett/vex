"""Topic boundary detection via embedding similarity.

Post-embedding refinement pass: compares cosine similarity between consecutive
chunks and splits at topic transitions (similarity drop below threshold).

Runs after embedding, before extraction. Purely programmatic — no LLM calls.
"""

from __future__ import annotations

import structlog

from app.db import get_pool

logger = structlog.get_logger()

# Similarity below this between consecutive chunks = topic boundary
_SIMILARITY_THRESHOLD = 0.5


async def detect_and_split_boundaries(
    conversation_id: int,
    *,
    threshold: float = _SIMILARITY_THRESHOLD,
) -> dict[str, int | float]:
    """Detect topic boundaries by comparing consecutive chunk embeddings.

    For each pair of adjacent chunks, compute cosine similarity. Where similarity
    drops below threshold, the chunks are confirmed as separate topics (no action
    needed — they're already separate chunks). Where similarity is high, adjacent
    chunks that were split by the chunker could potentially be merged, but we
    don't do that here — the chunker's splits are authoritative.

    Instead, this function identifies boundary points and logs them for
    observability. Future enhancement: use boundaries to improve chunk
    significance scoring (topic-initiating chunks get a boost).

    Args:
        conversation_id: ID of the conversation to analyse.
        threshold: Cosine similarity threshold for topic boundary detection.

    Returns:
        Dict with counts: total_pairs, boundaries_found, avg_similarity.
    """
    pool = get_pool()

    async with pool.acquire() as conn:
        # Fetch chunks with embeddings, ordered by position
        rows = await conn.fetch(
            """
            SELECT id, start_ordinal, end_ordinal, chunk_type, significance, embedding
            FROM chunks
            WHERE conversation_id = $1 AND embedding IS NOT NULL
            ORDER BY start_ordinal
            """,
            conversation_id,
        )

    if len(rows) < 2:
        return {"total_pairs": 0, "boundaries_found": 0, "avg_similarity": 0.0}

    # Compare consecutive pairs using cosine similarity
    similarities: list[float] = []
    boundary_chunks: list[int] = []

    for i in range(len(rows) - 1):
        current = rows[i]
        next_chunk = rows[i + 1]

        # Compute cosine similarity between embeddings
        sim = _cosine_similarity(current["embedding"], next_chunk["embedding"])
        similarities.append(sim)

        if sim < threshold:
            boundary_chunks.append(next_chunk["id"])

    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

    # Boost significance of chunks that start a new topic
    if boundary_chunks:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE chunks
                SET significance = LEAST(significance + 1, 5)
                WHERE id = ANY($1) AND significance < 5
                """,
                boundary_chunks,
            )

    logger.info(
        "topic_boundary.complete",
        conversation_id=conversation_id,
        total_pairs=len(similarities),
        boundaries_found=len(boundary_chunks),
        avg_similarity=round(avg_similarity, 3),
    )

    return {
        "total_pairs": len(similarities),
        "boundaries_found": len(boundary_chunks),
        "avg_similarity": round(avg_similarity, 3),
    }


def _cosine_similarity(a: str | list[float], b: str | list[float]) -> float:
    """Compute cosine similarity between two embedding vectors.

    Handles both string (from pgvector) and list representations.
    """
    if isinstance(a, str):
        a = [float(x) for x in a.strip("[]").split(",")]
    if isinstance(b, str):
        b = [float(x) for x in b.strip("[]").split(",")]

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot / (norm_a * norm_b))
