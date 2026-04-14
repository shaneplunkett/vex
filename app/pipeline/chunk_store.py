"""Chunk storage — reads messages from DB, runs chunker, stores results."""

from __future__ import annotations

import asyncpg
import structlog

from app.db import get_pool
from app.pipeline.chunker import Chunk, chunk_conversation

logger = structlog.get_logger()


async def chunk_pending_conversations() -> int:
    """Find conversations with pipeline_status='pending' and chunk them.

    Returns the number of conversations chunked.
    """
    pool = get_pool()
    chunked = 0

    async with pool.acquire() as conn:
        conversations = await conn.fetch(
            """
            SELECT id, session_id FROM conversations
            WHERE pipeline_status = 'pending'
            ORDER BY id
            """
        )

        for conv in conversations:
            conv_id = conv["id"]
            try:
                messages = await conn.fetch(
                    """
                    SELECT id, role, content, timestamp, ordinal
                    FROM messages
                    WHERE conversation_id = $1
                    ORDER BY ordinal
                    """,
                    conv_id,
                )

                msg_dicts = [dict(m) for m in messages]
                chunks = chunk_conversation(msg_dicts, conv_id)

                if not chunks:
                    logger.warning("chunk_store.no_chunks", conversation_id=conv_id)
                    continue

                async with conn.transaction():
                    await _store_chunks(conn, conv_id, chunks)
                    await conn.execute(
                        "UPDATE conversations SET pipeline_status = 'chunked' WHERE id = $1",
                        conv_id,
                    )

                chunked += 1
                logger.info(
                    "chunk_store.conversation_chunked",
                    conversation_id=conv_id,
                    chunk_count=len(chunks),
                )

            except Exception:
                logger.exception(
                    "chunk_store.conversation_failed",
                    conversation_id=conv_id,
                )
                await conn.execute(
                    """
                    UPDATE conversations
                    SET pipeline_status = 'failed', pipeline_error = 'chunking failed'
                    WHERE id = $1
                    """,
                    conv_id,
                )

    logger.info("chunk_store.batch_complete", chunked=chunked, total=len(conversations))
    return chunked


async def _store_chunks(
    conn: asyncpg.Connection,  # type: ignore[type-arg]
    conversation_id: int,
    chunks: list[Chunk],
) -> None:
    """Store chunks for a conversation using batched insert."""
    chunk_rows = [
        (
            conversation_id,
            chunk.content,
            chunk.raw_content,
            chunk.start_message_id,
            chunk.end_message_id,
            chunk.start_ordinal,
            chunk.end_ordinal,
            chunk.chunk_type,
            chunk.significance,
        )
        for chunk in chunks
    ]

    await conn.executemany(
        """
        INSERT INTO chunks
            (conversation_id, content, raw_content, start_message_id, end_message_id,
             start_ordinal, end_ordinal, chunk_type, significance)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """,
        chunk_rows,
    )
