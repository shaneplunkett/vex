"""Database storage for parsed conversations.

Handles inserting ParsedConversation objects into L1 tables (conversations + messages).
Uses session_id UNIQUE constraint for idempotent re-imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncpg.exceptions
import structlog

from app.db import get_pool

if TYPE_CHECKING:
    from app.importers.cc import ParsedConversation

logger = structlog.get_logger()


async def store_conversation(conversation: ParsedConversation, source: str) -> int | None:
    """Store a parsed conversation and its messages in L1 tables.

    Returns the conversation ID if inserted, None if it already existed (dedup).
    """
    pool = get_pool()

    async with pool.acquire() as conn:
        async with conn.transaction():
            # Insert conversation — session_id UNIQUE handles dedup
            try:
                conv_id: int = await conn.fetchval(
                    """
                    INSERT INTO conversations
                        (source, session_id, name, started_at, ended_at, message_count, pipeline_status)
                    VALUES ($1, $2, $3, $4, $5, $6, 'pending')
                    RETURNING id
                    """,
                    source,
                    conversation.session_id,
                    conversation.name,
                    conversation.started_at,
                    conversation.ended_at,
                    conversation.message_count,
                )
            except asyncpg.exceptions.UniqueViolationError:
                logger.debug(
                    "store.skip_duplicate",
                    session_id=conversation.session_id,
                    source=source,
                )
                return None

            # Batch insert messages
            message_rows = [
                (
                    conv_id,
                    msg.role,
                    msg.content,
                    msg.timestamp,
                    msg.ordinal,
                )
                for msg in conversation.messages
            ]

            await conn.executemany(
                """
                INSERT INTO messages (conversation_id, role, content, timestamp, ordinal)
                VALUES ($1, $2, $3, $4, $5)
                """,
                message_rows,
            )

        logger.info(
            "store.conversation_stored",
            conversation_id=conv_id,
            session_id=conversation.session_id,
            message_count=conversation.message_count,
        )
        return conv_id


async def store_conversations(conversations: list[ParsedConversation], source: str) -> tuple[int, int]:
    """Store multiple parsed conversations. Returns (imported, skipped) counts."""
    imported = 0
    skipped = 0

    for conversation in conversations:
        result = await store_conversation(conversation, source)
        if result is not None:
            imported += 1
        else:
            skipped += 1

    logger.info(
        "store.batch_complete",
        source=source,
        imported=imported,
        skipped=skipped,
        total=len(conversations),
    )
    return imported, skipped
