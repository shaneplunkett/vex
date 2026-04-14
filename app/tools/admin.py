"""Admin tools — system health, maintenance, and reprocessing.

stats(): system-wide counts and distributions.
get_audit_report(): view audit findings from audit_log.
reprocess_conversation(): reset pipeline status and re-queue.
reembed_all(): nullify embeddings for re-embedding (model changes).
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from app.db import get_pool

logger = structlog.get_logger()


async def stats() -> dict[str, Any]:
    """Return system-wide statistics.

    Counts by layer (conversations, messages, chunks, entities, relations),
    entity type distribution, pipeline status breakdown, and queue depth.

    Returns:
        Dict with nested counts and distributions.
    """
    pool = get_pool()

    async with pool.acquire() as conn:
        # Layer counts
        conversations = await conn.fetchval("SELECT COUNT(*) FROM conversations")
        messages = await conn.fetchval("SELECT COUNT(*) FROM messages")
        chunks = await conn.fetchval("SELECT COUNT(*) FROM chunks")
        chunks_with_embedding = await conn.fetchval("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
        entities = await conn.fetchval("SELECT COUNT(*) FROM entities")
        relations = await conn.fetchval("SELECT COUNT(*) FROM relations WHERE superseded_at IS NULL")
        relations_superseded = await conn.fetchval("SELECT COUNT(*) FROM relations WHERE superseded_at IS NOT NULL")

        # Pipeline status breakdown
        pipeline_rows = await conn.fetch(
            "SELECT pipeline_status, COUNT(*) AS cnt FROM conversations GROUP BY pipeline_status ORDER BY cnt DESC"
        )
        pipeline_status = {row["pipeline_status"]: row["cnt"] for row in pipeline_rows}

        # Entity type distribution
        type_rows = await conn.fetch(
            "SELECT entity_type, COUNT(*) AS cnt FROM entities GROUP BY entity_type ORDER BY cnt DESC"
        )
        entity_types = {row["entity_type"]: row["cnt"] for row in type_rows}

        # Queue depths
        review_pending = await conn.fetchval("SELECT COUNT(*) FROM review_queue WHERE status = 'pending'")
        review_resolved = await conn.fetchval("SELECT COUNT(*) FROM review_queue WHERE status = 'resolved'")
        review_dismissed = await conn.fetchval("SELECT COUNT(*) FROM review_queue WHERE status = 'dismissed'")

        # Denylist count
        denylist_count = await conn.fetchval("SELECT COUNT(*) FROM extraction_denylist")

        # Entity-chunk links
        entity_chunk_links = await conn.fetchval("SELECT COUNT(*) FROM entity_chunks")
        relation_chunk_links = await conn.fetchval("SELECT COUNT(*) FROM relation_chunks")

    return {
        "layers": {
            "conversations": conversations,
            "messages": messages,
            "chunks": chunks,
            "chunks_with_embedding": chunks_with_embedding,
            "entities": entities,
            "relations_active": relations,
            "relations_superseded": relations_superseded,
        },
        "pipeline_status": pipeline_status,
        "entity_types": entity_types,
        "review_queue": {
            "pending": review_pending,
            "resolved": review_resolved,
            "dismissed": review_dismissed,
        },
        "denylist_count": denylist_count,
        "links": {
            "entity_chunks": entity_chunk_links,
            "relation_chunks": relation_chunk_links,
        },
    }


async def get_audit_report(latest: bool = True) -> list[dict[str, Any]]:
    """Return audit findings from audit_log.

    Args:
        latest: If True, return only the most recent report per audit_type.
                If False, return all reports.

    Returns:
        List of audit report dicts.
    """
    pool = get_pool()

    async with pool.acquire() as conn:
        if latest:
            rows = await conn.fetch(
                """
                SELECT DISTINCT ON (audit_type)
                    id, audit_type, findings, actions_taken, created_at
                FROM audit_log
                ORDER BY audit_type, created_at DESC
                """
            )
        else:
            rows = await conn.fetch(
                """
                SELECT id, audit_type, findings, actions_taken, created_at
                FROM audit_log
                ORDER BY created_at DESC
                LIMIT 50
                """
            )

    return [
        {
            "id": row["id"],
            "audit_type": row["audit_type"],
            "findings": row["findings"],
            "actions_taken": row["actions_taken"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


async def reprocess_conversation(conversation_id: int) -> dict[str, Any]:
    """Reset a conversation's pipeline status and re-queue for processing.

    Resets to 'pending' and spawns the pipeline as a background task.
    Existing chunks, embeddings, and entity links from previous runs
    are preserved — the pipeline is idempotent for chunking/embedding.

    Args:
        conversation_id: ID of the conversation to reprocess.

    Returns:
        Dict with conversation_id and action taken.
    """
    pool = get_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, pipeline_status FROM conversations WHERE id = $1",
            conversation_id,
        )
        if row is None:
            return {"error": f"Conversation {conversation_id} not found"}

        old_status = row["pipeline_status"]

        await conn.execute(
            """
            UPDATE conversations
            SET pipeline_status = 'pending', pipeline_error = NULL
            WHERE id = $1
            """,
            conversation_id,
        )

    # Spawn pipeline as background task
    from app.pipeline.orchestrator import process_conversation

    asyncio.create_task(
        process_conversation(conversation_id),
        name=f"pipeline:reprocess:{conversation_id}",
    )

    logger.info(
        "admin.reprocess",
        conversation_id=conversation_id,
        old_status=old_status,
    )
    return {
        "conversation_id": conversation_id,
        "old_status": old_status,
        "new_status": "pending",
        "action": "requeued",
    }


async def reembed_all() -> dict[str, Any]:
    """Nullify all chunk embeddings to trigger re-embedding.

    Use after changing embedding model or dimensions. Does NOT re-run
    the pipeline — call reprocess on individual conversations or wait
    for the startup sweep.

    Returns:
        Dict with count of chunks affected.
    """
    pool = get_pool()

    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE chunks SET embedding = NULL
            WHERE embedding IS NOT NULL
            """
        )

        # Reset conversations to 'chunked' so the pipeline re-embeds them
        conv_count = await conn.fetchval(
            """
            UPDATE conversations
            SET pipeline_status = 'chunked'
            WHERE pipeline_status IN ('embedded', 'extracted', 'complete')
            RETURNING COUNT(*)
            """
        )

    # Parse the "UPDATE N" result
    chunks_cleared = int(result.split()[-1]) if result else 0

    logger.info(
        "admin.reembed_all",
        chunks_cleared=chunks_cleared,
        conversations_reset=conv_count,
    )
    return {
        "chunks_cleared": chunks_cleared,
        "conversations_reset": conv_count or 0,
        "action": "embeddings_nullified",
        "note": "Run startup_sweep or reprocess individual conversations to re-embed",
    }
