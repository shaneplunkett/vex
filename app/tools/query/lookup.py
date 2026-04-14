"""Lookup tools — get_conversation(), recent_conversations(), get_entity()."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import structlog

from app.db import get_pool
from app.tools.query._common import MELBOURNE_TZ

logger = structlog.get_logger()


async def get_conversation(conversation_id: int) -> dict[str, Any] | None:
    """Return the full transcript and metadata for a single conversation.

    Args:
        conversation_id: Primary key of the conversation.

    Returns:
        Dict with conversation metadata and ordered list of messages, or None
        if the conversation does not exist.
    """
    pool = get_pool()

    async with pool.acquire() as conn:
        conv_row = await conn.fetchrow(
            """
            SELECT id, name, source, started_at, ended_at, message_count,
                   pipeline_status, created_at
            FROM conversations
            WHERE id = $1
            """,
            conversation_id,
        )

        if conv_row is None:
            logger.info("get_conversation.not_found", conversation_id=conversation_id)
            return None

        message_rows = await conn.fetch(
            """
            SELECT id, role, content, timestamp, ordinal
            FROM messages
            WHERE conversation_id = $1
            ORDER BY ordinal
            """,
            conversation_id,
        )

    messages = [
        {
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "timestamp": row["timestamp"],
            "ordinal": row["ordinal"],
        }
        for row in message_rows
    ]

    result: dict[str, Any] = {
        "id": conv_row["id"],
        "name": conv_row["name"],
        "source": conv_row["source"],
        "started_at": conv_row["started_at"],
        "ended_at": conv_row["ended_at"],
        "message_count": conv_row["message_count"],
        "pipeline_status": conv_row["pipeline_status"],
        "created_at": conv_row["created_at"],
        "messages": messages,
    }

    logger.info(
        "get_conversation.complete",
        conversation_id=conversation_id,
        message_count=len(messages),
    )
    return result


async def recent_conversations(*, days: int = 7, limit: int = 10) -> list[dict[str, Any]]:
    """Return a list of recent conversations ordered by start time descending.

    Args:
        days: Include conversations started within this many days ago
              (Melbourne timezone for "today").
        limit: Maximum number of conversations to return.

    Returns:
        List of dicts with conversation metadata.
    """
    days = max(1, min(365, days))
    limit = max(1, min(100, limit))

    # Use Melbourne calendar-day boundary so results are stable within a local day
    today_melb = datetime.now(tz=MELBOURNE_TZ).date()
    cutoff_date = today_melb - timedelta(days=days)
    cutoff = datetime.combine(cutoff_date, datetime.min.time(), tzinfo=MELBOURNE_TZ)

    pool = get_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, name, source, started_at, ended_at,
                   message_count, pipeline_status
            FROM conversations
            WHERE started_at >= $1
            ORDER BY started_at DESC
            LIMIT $2
            """,
            cutoff,
            limit,
        )

    results = [
        {
            "id": row["id"],
            "name": row["name"],
            "source": row["source"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "message_count": row["message_count"],
            "pipeline_status": row["pipeline_status"],
        }
        for row in rows
    ]

    logger.info(
        "recent_conversations.complete",
        days=days,
        limit=limit,
        results=len(results),
    )
    return results


async def get_entity(name_or_id: str) -> dict[str, Any] | None:
    """Return an entity with its current relations and linked chunk summaries.

    Tries to match by numeric ID first, then falls back to case-insensitive
    name matching (ILIKE).

    Args:
        name_or_id: Entity ID (numeric string) or name (partial match supported).

    Returns:
        Dict with entity fields, relations, and chunk summaries, or None if not found.
    """
    name_or_id = name_or_id.strip()
    if not name_or_id:
        logger.warning("get_entity.blank_lookup")
        return None

    pool = get_pool()

    async with pool.acquire() as conn:
        # Try by ID if numeric
        entity_row = None
        if name_or_id.isdigit():
            entity_row = await conn.fetchrow(
                """
                SELECT id, name, entity_type, summary, aliases,
                       created_at, updated_at, access_count
                FROM entities
                WHERE id = $1
                """,
                int(name_or_id),
            )

        # Fall back to case-insensitive name match
        if entity_row is None:
            # Escape ILIKE pattern characters so literal %, _ in names don't act as wildcards
            escaped = name_or_id.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            entity_row = await conn.fetchrow(
                """
                SELECT id, name, entity_type, summary, aliases,
                       created_at, updated_at, access_count
                FROM entities
                WHERE name ILIKE $1
                ORDER BY
                    CASE WHEN lower(name) = lower($2) THEN 0 ELSE 1 END,
                    access_count DESC
                LIMIT 1
                """,
                f"%{escaped}%",
                name_or_id,
            )

        if entity_row is None:
            logger.info("get_entity.not_found", name_or_id=name_or_id)
            return None

        entity_id = entity_row["id"]

        # Current (non-superseded) relations — both directions
        relation_rows = await conn.fetch(
            """
            SELECT r.id, r.relation_type, r.description, r.valid_from,
                   r.source_id, r.target_id,
                   es.name AS source_name, et.name AS target_name
            FROM relations r
            JOIN entities es ON es.id = r.source_id
            JOIN entities et ON et.id = r.target_id
            WHERE (r.source_id = $1 OR r.target_id = $1)
              AND r.superseded_at IS NULL
            ORDER BY r.created_at DESC
            """,
            entity_id,
        )

        relations = [
            {
                "id": row["id"],
                "relation_type": row["relation_type"],
                "description": row["description"],
                "valid_from": row["valid_from"],
                "source_id": row["source_id"],
                "source_name": row["source_name"],
                "target_id": row["target_id"],
                "target_name": row["target_name"],
                "direction": "outbound" if row["source_id"] == entity_id else "inbound",
            }
            for row in relation_rows
        ]

        # Linked chunk summaries (chunk_type, significance, conversation name + date)
        chunk_rows = await conn.fetch(
            """
            SELECT ch.id, ch.chunk_type, ch.significance, ch.content,
                   c.name AS conversation_name, c.started_at AS conversation_date
            FROM entity_chunks ec
            JOIN chunks ch ON ch.id = ec.chunk_id
            JOIN conversations c ON c.id = ch.conversation_id
            WHERE ec.entity_id = $1
            ORDER BY ch.significance DESC, c.started_at DESC
            LIMIT 20
            """,
            entity_id,
        )

        chunks = [
            {
                "chunk_id": row["id"],
                "chunk_type": row["chunk_type"],
                "significance": row["significance"],
                "content": row["content"],
                "conversation_name": row["conversation_name"],
                "conversation_date": row["conversation_date"],
            }
            for row in chunk_rows
        ]

        # Update access tracking — best-effort, return fresh count
        access_count = entity_row["access_count"]
        try:
            access_count = await conn.fetchval(
                """
                UPDATE entities
                SET access_count = access_count + 1, last_accessed_at = now()
                WHERE id = $1
                RETURNING access_count
                """,
                entity_id,
            )
        except Exception:
            logger.warning("get_entity.access_tracking_failed", exc_info=True)

    result: dict[str, Any] = {
        "id": entity_row["id"],
        "name": entity_row["name"],
        "entity_type": entity_row["entity_type"],
        "summary": entity_row["summary"],
        "aliases": entity_row["aliases"],
        "created_at": entity_row["created_at"],
        "updated_at": entity_row["updated_at"],
        "access_count": access_count,
        "relations": relations,
        "chunks": chunks,
    }

    logger.info(
        "get_entity.complete",
        entity_id=entity_id,
        name=entity_row["name"],
        relations=len(relations),
        chunks=len(chunks),
    )
    return result
