"""search() — keyword search over raw messages with date/source filtering."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog

from app.db import get_pool

logger = structlog.get_logger()


async def search(
    query: str,
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    source: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Keyword search across raw message content.

    Returns matching messages with conversation context (name, date,
    surrounding messages for context).

    Args:
        query: Search terms
        date_from: ISO date string filter (inclusive), e.g. "2026-01-01"
        date_to: ISO date string filter (inclusive), e.g. "2026-03-25"
        source: Filter by source ("cc" or "claude_ai")
        limit: Maximum results
    """
    if not query or not query.strip():
        return []

    limit = max(1, min(50, limit))

    # Parse and validate date inputs — asyncpg needs datetime objects
    parsed_from: datetime | None = None
    parsed_to: datetime | None = None
    if date_from:
        try:
            parsed_from = datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc)
        except ValueError:
            logger.warning("search.invalid_date_from", value=date_from)
            return []
    if date_to:
        try:
            dt = datetime.fromisoformat(date_to)
            # If date-only input ("2026-03-25"), bump to end of day
            if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
                parsed_to = dt.replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
            else:
                parsed_to = dt.replace(tzinfo=timezone.utc)
        except ValueError:
            logger.warning("search.invalid_date_to", value=date_to)
            return []

    pool = get_pool()

    # Build parameterised query with optional filters.
    # Date filters use message timestamp (not conversation start) for accuracy.
    conditions = ["to_tsvector('english', m.content) @@ plainto_tsquery('english', $1)"]
    params: list[Any] = [query]
    param_idx = 2

    if parsed_from:
        conditions.append(f"m.timestamp >= ${param_idx}")
        params.append(parsed_from)
        param_idx += 1

    if parsed_to:
        conditions.append(f"m.timestamp <= ${param_idx}")
        params.append(parsed_to)
        param_idx += 1

    if source:
        conditions.append(f"c.source = ${param_idx}")
        params.append(source)
        param_idx += 1

    where_clause = " AND ".join(conditions)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT m.id, m.conversation_id, m.role, m.content, m.timestamp, m.ordinal,
                   c.name as conversation_name, c.source, c.started_at,
                   ts_rank(to_tsvector('english', m.content),
                           plainto_tsquery('english', $1)) as rank
            FROM messages m
            JOIN conversations c ON c.id = m.conversation_id
            WHERE {where_clause}
            ORDER BY rank DESC
            LIMIT ${param_idx}
            """,
            *params,
            limit,
        )

        results = []
        for row in rows:
            # Fetch surrounding messages for context (2 before, 2 after)
            context_msgs = await conn.fetch(
                """
                SELECT role, content, ordinal
                FROM messages
                WHERE conversation_id = $1
                  AND ordinal BETWEEN $2 AND $3
                ORDER BY ordinal
                """,
                row["conversation_id"],
                max(0, row["ordinal"] - 2),
                row["ordinal"] + 2,
            )

            results.append(
                {
                    "message_id": row["id"],
                    "conversation_id": row["conversation_id"],
                    "conversation_name": row["conversation_name"],
                    "source": row["source"],
                    "started_at": row["started_at"],
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"],
                    "ordinal": row["ordinal"],
                    "rank": float(row["rank"]),
                    "context": [dict(m) for m in context_msgs],
                }
            )

    logger.info(
        "search.complete",
        query_length=len(query),
        results=len(results),
        filters={"date_from": date_from, "date_to": date_to, "source": source},
    )
    return results
