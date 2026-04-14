"""Coreference storage — applies resolution to chunks and updates content column."""

from __future__ import annotations

import structlog

from app.db import get_pool
from app.pipeline.coreference import resolve_chunk

logger = structlog.get_logger()


async def resolve_conversation_chunks(conversation_id: int) -> int:
    """Resolve coreferences on all chunks for a conversation.

    Updates chunks.content with resolved text (raw_content preserved).
    Designed to run immediately after chunking, before embedding.
    Does not change pipeline_status — conversation stays 'chunked'.

    Per-chunk errors are logged and skipped — one bad chunk doesn't
    block the rest. All successful updates are wrapped in a transaction.

    Returns the number of chunks updated.
    """
    pool = get_pool()
    updated = 0

    async with pool.acquire() as conn:
        chunks = await conn.fetch(
            """
            SELECT id, raw_content FROM chunks
            WHERE conversation_id = $1
            ORDER BY start_ordinal
            """,
            conversation_id,
        )

        # Collect resolved updates, skipping failures
        updates: list[tuple[str, int]] = []
        for chunk in chunks:
            raw = chunk["raw_content"]
            if not raw:
                continue
            try:
                resolved = resolve_chunk(raw)
                if resolved != raw:
                    updates.append((resolved, chunk["id"]))
            except Exception:
                logger.exception(
                    "coref_store.chunk_failed",
                    conversation_id=conversation_id,
                    chunk_id=chunk["id"],
                )

        # Apply all successful updates atomically
        if updates:
            async with conn.transaction():
                for resolved_text, chunk_id in updates:
                    await conn.execute(
                        "UPDATE chunks SET content = $1 WHERE id = $2",
                        resolved_text,
                        chunk_id,
                    )
            updated = len(updates)

    logger.info(
        "coref_store.resolved",
        conversation_id=conversation_id,
        chunks_total=len(chunks),
        chunks_updated=updated,
    )
    return updated
