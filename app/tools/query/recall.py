"""recall() — hybrid semantic + keyword search over chunks with RRF scoring."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog

from app.db import get_pool
from app.pipeline.embedder import embed_texts

logger = structlog.get_logger()

_RRF_K = 60  # Reciprocal rank fusion constant
_RECENCY_HALFLIFE_DAYS = 180  # Days for recency to reach ~0.5


async def recall(
    query: str,
    *,
    depth: int = 1,
    limit: int = 10,
    chunk_type: str | None = None,
) -> list[dict[str, Any]]:
    """Hybrid semantic + keyword search over chunks.

    Args:
        query: Natural language search query
        depth: Response detail level (1=entities, 2=+chunks, 3=+conversation segments)
        limit: Maximum results to return
        chunk_type: Optional filter by chunk type (e.g., "decision", "emotional")

    Returns:
        List of result dicts with chunk data, scores, and depth-dependent context.
    """
    pool = get_pool()

    # Validate inputs
    depth = max(1, min(3, depth))
    limit = max(1, min(50, limit))

    if not query or not query.strip():
        return []

    # Step 1: Embed query — graceful degradation to keyword-only on failure
    query_vector: list[float] | None = None
    try:
        vectors = await embed_texts([query])
        query_vector = vectors[0]
    except Exception:
        logger.exception("recall.embedding_error", query_length=len(query))

    if query_vector is None:
        logger.warning("recall.keyword_only_mode", query_length=len(query))

    async with pool.acquire() as conn:
        # Step 2: pgvector cosine similarity search (top 20)
        vector_results = await _vector_search(conn, query_vector, chunk_type)

        # Step 3: tsvector BM25 keyword search (top 20)
        keyword_results = await _keyword_search(conn, query, chunk_type)

        # Step 4: Reciprocal rank fusion
        fused = _reciprocal_rank_fusion(vector_results, keyword_results)

        # Step 5: Apply recency + significance weighting
        scored = _apply_scoring(fused)

        # Sort and limit
        scored.sort(key=lambda x: x["final_score"], reverse=True)
        top_results = scored[:limit]

        # Expand based on depth
        results = await _expand_results(conn, top_results, depth)

        # Update access tracking — best-effort, don't crash if it fails
        try:
            await _update_access_tracking(conn, top_results)
        except Exception:
            logger.warning("recall.access_tracking_failed", exc_info=True)

    logger.info(
        "recall.complete",
        query_length=len(query),
        depth=depth,
        vector_candidates=len(vector_results),
        keyword_candidates=len(keyword_results),
        fused_candidates=len(fused),
        results_returned=len(results),
    )
    return results


async def _vector_search(
    conn: Any,
    query_vector: list[float] | None,
    chunk_type: str | None,
) -> list[dict[str, Any]]:
    """Search chunks by embedding cosine similarity."""
    if query_vector is None:
        return []

    type_filter = "AND c.chunk_type = $2" if chunk_type else ""
    params: list[Any] = [str(query_vector)]
    if chunk_type:
        params.append(chunk_type)

    rows = await conn.fetch(
        f"""
        SELECT c.id, c.conversation_id, c.content, c.chunk_type, c.significance,
               c.start_ordinal, c.end_ordinal, c.start_message_id, c.end_message_id,
               c.created_at,
               1 - (c.embedding <=> $1::vector) as similarity
        FROM chunks c
        WHERE c.embedding IS NOT NULL {type_filter}
        ORDER BY c.embedding <=> $1::vector
        LIMIT 20
        """,
        *params,
    )

    return [{**dict(row), "vector_rank": i + 1} for i, row in enumerate(rows)]


async def _keyword_search(
    conn: Any,
    query: str,
    chunk_type: str | None,
) -> list[dict[str, Any]]:
    """Search chunks by tsvector full-text search."""
    type_filter = "AND c.chunk_type = $2" if chunk_type else ""
    params: list[Any] = [query]
    if chunk_type:
        params.append(chunk_type)

    rows = await conn.fetch(
        f"""
        SELECT c.id, c.conversation_id, c.content, c.chunk_type, c.significance,
               c.start_ordinal, c.end_ordinal, c.start_message_id, c.end_message_id,
               c.created_at,
               ts_rank(to_tsvector('english', c.content),
                       plainto_tsquery('english', $1)) as keyword_score
        FROM chunks c
        WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', $1)
              {type_filter}
        ORDER BY keyword_score DESC
        LIMIT 20
        """,
        *params,
    )

    return [{**dict(row), "keyword_rank": i + 1} for i, row in enumerate(rows)]


def _reciprocal_rank_fusion(
    vector_results: list[dict[str, Any]],
    keyword_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combine vector and keyword results using reciprocal rank fusion."""
    # Index by chunk id
    combined: dict[int, dict[str, Any]] = {}

    for result in vector_results:
        chunk_id = result["id"]
        combined[chunk_id] = {
            **result,
            "rrf_vector": 1.0 / (_RRF_K + result["vector_rank"]),
            "rrf_keyword": 0.0,
        }

    for result in keyword_results:
        chunk_id = result["id"]
        if chunk_id in combined:
            combined[chunk_id]["rrf_keyword"] = 1.0 / (_RRF_K + result["keyword_rank"])
            # Keep the higher keyword_score if present
            if "keyword_score" not in combined[chunk_id]:
                combined[chunk_id]["keyword_score"] = result.get("keyword_score", 0)
        else:
            combined[chunk_id] = {
                **result,
                "rrf_vector": 0.0,
                "rrf_keyword": 1.0 / (_RRF_K + result["keyword_rank"]),
            }

    for entry in combined.values():
        entry["hybrid_score"] = entry["rrf_vector"] + entry["rrf_keyword"]

    return list(combined.values())


def _apply_scoring(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply recency decay and significance weighting to hybrid scores.

    Normalises hybrid scores to [0, 1] before weighting so that relevance
    actually dominates the final score as intended.
    """
    if not results:
        return results

    now = datetime.now(tz=timezone.utc)

    # Normalise hybrid scores to [0, 1] — RRF scores are tiny (~0.03 max)
    # without normalisation they'd contribute ~5% of final score
    max_hybrid = max(r.get("hybrid_score", 0) for r in results)
    if max_hybrid == 0:
        max_hybrid = 1.0  # avoid division by zero

    for result in results:
        # Recency: asymptotic decay — never reaches zero
        created_at = result.get("created_at")
        if created_at:
            days_ago = max(0, (now - created_at).total_seconds() / 86400)
            recency = 1.0 / (1.0 + days_ago / _RECENCY_HALFLIFE_DAYS)
        else:
            recency = 0.5  # unknown age = neutral

        # Significance: normalised to 0-1
        significance = (result.get("significance") or 3) / 5.0

        # Normalise hybrid to [0, 1]
        hybrid = result.get("hybrid_score", 0) / max_hybrid

        result["recency_score"] = recency
        result["significance_score"] = significance
        result["final_score"] = (hybrid * 0.6) + (recency * 0.2) + (significance * 0.2)

    return results


async def _expand_results(
    conn: Any,
    results: list[dict[str, Any]],
    depth: int,
) -> list[dict[str, Any]]:
    """Expand results based on requested depth level.

    Uses batched queries to avoid N+1 performance issues.
    """
    if not results:
        return []

    chunk_ids = [r["id"] for r in results]
    conv_ids = list({r["conversation_id"] for r in results})

    # Batch: linked entities for all chunks
    entity_rows = await conn.fetch(
        """
        SELECT ec.chunk_id, e.id, e.name, e.entity_type, e.summary
        FROM entities e
        JOIN entity_chunks ec ON ec.entity_id = e.id
        WHERE ec.chunk_id = ANY($1)
        """,
        chunk_ids,
    )
    entities_by_chunk: dict[int, list[dict[str, Any]]] = {}
    for row in entity_rows:
        entities_by_chunk.setdefault(row["chunk_id"], []).append(
            {"id": row["id"], "name": row["name"], "entity_type": row["entity_type"], "summary": row["summary"]}
        )

    # Batch: conversation metadata (depth >= 2)
    convs_by_id: dict[int, dict[str, Any]] = {}
    if depth >= 2:
        conv_rows = await conn.fetch(
            "SELECT id, name, source, started_at, ended_at FROM conversations WHERE id = ANY($1)",
            conv_ids,
        )
        convs_by_id = {row["id"]: dict(row) for row in conv_rows}

    # Batch: messages (depth >= 3) — one query per chunk for ordinal ranges
    messages_by_chunk: dict[int, list[dict[str, Any]]] = {}
    if depth >= 3:
        for result in results:
            msg_rows = await conn.fetch(
                """
                SELECT role, content, timestamp, ordinal
                FROM messages
                WHERE conversation_id = $1 AND ordinal BETWEEN $2 AND $3
                ORDER BY ordinal
                LIMIT 100
                """,
                result["conversation_id"],
                result["start_ordinal"],
                result["end_ordinal"],
            )
            messages_by_chunk[result["id"]] = [dict(m) for m in msg_rows]

    # Build expanded results
    expanded = []
    for result in results:
        entry: dict[str, Any] = {
            "chunk_id": result["id"],
            "conversation_id": result["conversation_id"],
            "chunk_type": result["chunk_type"],
            "significance": result["significance"],
            "final_score": result["final_score"],
            "entities": entities_by_chunk.get(result["id"], []),
        }

        if depth >= 2:
            entry["content"] = result["content"]
            entry["start_ordinal"] = result["start_ordinal"]
            entry["end_ordinal"] = result["end_ordinal"]
            conv = convs_by_id.get(result["conversation_id"])
            if conv:
                entry["conversation"] = {k: v for k, v in conv.items() if k != "id"}

        if depth >= 3:
            entry["messages"] = messages_by_chunk.get(result["id"], [])

        expanded.append(entry)

    return expanded


async def _update_access_tracking(
    conn: Any,
    results: list[dict[str, Any]],
) -> None:
    """Update access_count and last_accessed_at on returned chunks and entities."""
    if not results:
        return

    chunk_ids = [r["id"] for r in results]

    await conn.execute(
        """
        UPDATE chunks
        SET access_count = access_count + 1, last_accessed_at = now()
        WHERE id = ANY($1)
        """,
        chunk_ids,
    )

    await conn.execute(
        """
        UPDATE entities
        SET access_count = access_count + 1, last_accessed_at = now()
        WHERE id IN (
            SELECT entity_id FROM entity_chunks WHERE chunk_id = ANY($1)
        )
        """,
        chunk_ids,
    )
