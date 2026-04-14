"""Pipeline orchestrator — coordinates the full ingestion pipeline as async background tasks.

Pipeline stages per conversation:
  pending → chunked (chunk_store)
           → embedded (embed_store)
  L3 extraction/complete deferred.
  Coreference resolution removed — zero search quality improvement measured.

Concurrency: capped at 3 simultaneous conversations via asyncio.Semaphore.
Error isolation: one conversation failing sets pipeline_status='failed' and
does not affect others.
"""

from __future__ import annotations

import asyncio
import traceback
from datetime import datetime

import structlog

from app.db import get_pool
from app.pipeline.chunk_store import _store_chunks
from app.pipeline.chunker import chunk_conversation
from app.pipeline.embed_store import embed_conversation_chunks
from app.pipeline.extractor import extract_conversation_chunks
from app.pipeline.linker import apply_conversation_extractions
from app.pipeline.topic_boundary import detect_and_split_boundaries

logger = structlog.get_logger()


def _ensure_datetime(value: datetime | str | None) -> datetime | None:
    """Coerce string timestamps to datetime objects for asyncpg."""
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return None


# Cap concurrent pipeline workers — lazy init from settings
_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """Get or create the pipeline semaphore from settings."""
    global _semaphore  # noqa: PLW0603
    if _semaphore is None:
        from app.config import get_settings

        _semaphore = asyncio.Semaphore(get_settings().pipeline_concurrency)
    return _semaphore


# Track in-flight tasks for graceful shutdown
_running_tasks: set[asyncio.Task] = set()


# ---------------------------------------------------------------------------
# Per-conversation chunking (single-conversation variant of chunk_pending_conversations)
# ---------------------------------------------------------------------------


async def _chunk_conversation(conversation_id: int) -> int:
    """Chunk a single conversation and update pipeline_status to 'chunked'.

    Returns the number of chunks stored.
    Raises on failure — caller is responsible for marking status='failed'.
    """
    pool = get_pool()

    async with pool.acquire() as conn:
        messages = await conn.fetch(
            """
            SELECT id, role, content, timestamp, ordinal
            FROM messages
            WHERE conversation_id = $1
            ORDER BY ordinal
            """,
            conversation_id,
        )

        msg_dicts = [dict(m) for m in messages]
        chunks = chunk_conversation(msg_dicts, conversation_id)

        if not chunks:
            logger.warning("orchestrator.no_chunks", conversation_id=conversation_id)
            # Mark failed — nothing to embed
            await conn.execute(
                """
                UPDATE conversations
                SET pipeline_status = 'failed', pipeline_error = 'chunker produced no chunks'
                WHERE id = $1
                """,
                conversation_id,
            )
            return 0

        async with conn.transaction():
            await _store_chunks(conn, conversation_id, chunks)
            await conn.execute(
                "UPDATE conversations SET pipeline_status = 'chunked' WHERE id = $1",
                conversation_id,
            )

        logger.info(
            "orchestrator.chunked",
            conversation_id=conversation_id,
            chunk_count=len(chunks),
        )
        return len(chunks)


# ---------------------------------------------------------------------------
# Core pipeline runner
# ---------------------------------------------------------------------------


async def process_conversation(conversation_id: int) -> None:
    """Run all pipeline stages for a single conversation.

    Stages:
      1. Chunking   → status: 'chunked'
      2. Embedding   → status: 'embedded'
      3. Extraction  → (extractor runs, returns results)
      4. Validation + Linking → status: 'extracted'

    Sets status='failed' with pipeline_error on any unhandled exception.
    Uses _semaphore to cap concurrency across all in-flight conversations.
    """
    async with _get_semaphore():
        log = logger.bind(conversation_id=conversation_id)

        try:
            # Check current status to resume from correct stage
            pool = get_pool()
            async with pool.acquire() as conn:
                status = await conn.fetchval(
                    "SELECT pipeline_status FROM conversations WHERE id = $1",
                    conversation_id,
                )

            log.info("orchestrator.pipeline_start", resume_from=status)

            # --- Stage 1: Chunking (only if pending) ---
            if status == "pending":
                chunk_count = await _chunk_conversation(conversation_id)
                if chunk_count == 0:
                    return
                status = "chunked"

            # --- Stage 2: Embedding (only if chunked) ---
            if status == "chunked":
                await embed_conversation_chunks(conversation_id)
                # Topic boundary detection — boost significance of topic-initiating chunks
                await detect_and_split_boundaries(conversation_id)
                status = "embedded"

            # --- Stage 3: Extraction + Validation + Linking (only if embedded) ---
            if status == "embedded":
                from app.config import get_settings

                if get_settings().pipeline_mode == "agent":
                    log.info("orchestrator.agent_mode_stopping_at_embedded")
                    return

                extraction_result = await extract_conversation_chunks(conversation_id)
                linker_result = None
                if extraction_result["results"]:
                    linker_result = await apply_conversation_extractions(conversation_id, extraction_result["results"])

                # Only mark extracted if extraction AND linking both fully succeeded
                extraction_clean = extraction_result["failed"] == 0
                linker_clean = linker_result is None or linker_result.get("chunks_failed", 0) == 0

                if extraction_clean and linker_clean:
                    async with pool.acquire() as conn:
                        await conn.execute(
                            "UPDATE conversations SET pipeline_status = 'extracted' WHERE id = $1",
                            conversation_id,
                        )
                else:
                    log.warning(
                        "orchestrator.partial_extraction",
                        extraction_failed=extraction_result["failed"],
                        linker_failed=linker_result.get("chunks_failed", 0) if linker_result else 0,
                    )

            log.info("orchestrator.pipeline_complete")

        except Exception:
            error_detail = traceback.format_exc()
            log.exception("orchestrator.pipeline_error")
            try:
                pool = get_pool()
                async with pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE conversations
                        SET pipeline_status = 'failed', pipeline_error = $1
                        WHERE id = $2
                        """,
                        error_detail[:2000],  # Truncate to fit column without overflow
                        conversation_id,
                    )
            except Exception:
                log.exception("orchestrator.failed_status_update_failed")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def save_conversation(
    source: str,
    session_id: str | None,
    name: str | None,
    messages: list[dict],
    started_at: datetime | None,
    ended_at: datetime | None,
) -> int:
    """Store a conversation (L1) immediately and spawn the pipeline as a background task.

    Inserts the conversation row and all messages in a single transaction,
    then fires-and-forgets process_conversation() via asyncio.create_task().

    Args:
        source: 'cc' or 'claude_ai'
        session_id: Unique session identifier (may be None for claude_ai imports)
        name: Human-readable conversation name
        messages: List of dicts with keys: role, content, timestamp (ISO str or datetime), ordinal
        started_at: Conversation start timestamp
        ended_at: Conversation end timestamp

    Returns:
        The new conversation ID.
    """
    pool = get_pool()

    async with pool.acquire() as conn, conn.transaction():
        conversation_id: int = await conn.fetchval(
            """
            INSERT INTO conversations (source, session_id, name, started_at, ended_at, message_count)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id
            """,
            source,
            session_id,
            name,
            started_at,
            ended_at,
            len(messages),
        )

        if messages:
            message_rows = [
                (
                    conversation_id,
                    msg["role"],
                    msg["content"],
                    _ensure_datetime(msg.get("timestamp")),
                    msg["ordinal"],
                )
                for msg in messages
            ]
            await conn.executemany(
                """
                INSERT INTO messages (conversation_id, role, content, timestamp, ordinal)
                VALUES ($1, $2, $3, $4, $5)
                """,
                message_rows,
            )

    logger.info(
        "orchestrator.conversation_saved",
        conversation_id=conversation_id,
        source=source,
        message_count=len(messages),
    )

    # Spawn pipeline as a background task — track for graceful shutdown
    task = asyncio.create_task(
        process_conversation(conversation_id),
        name=f"pipeline:{conversation_id}",
    )
    _running_tasks.add(task)
    task.add_done_callback(_running_tasks.discard)

    return conversation_id


async def startup_sweep() -> int:
    """Re-queue any conversations that didn't complete the pipeline before last shutdown.

    Finds conversations with pipeline_status NOT IN ('complete', 'failed', 'extracted')
    — i.e. 'pending', 'chunked', or 'embedded' — and spawns process_conversation() for each.

    Should be called from the server lifespan after the DB pool and migrations are ready.

    Returns the number of conversations re-queued.
    """
    from app.config import get_settings

    pool = get_pool()

    # In agent mode, don't re-queue embedded conversations — they're waiting for
    # external agent processing, not API extraction.
    terminal = ["complete", "failed", "extracted"]
    if get_settings().pipeline_mode == "agent":
        terminal.append("embedded")

    async with pool.acquire() as conn:
        stalled = await conn.fetch(
            """
            SELECT id FROM conversations
            WHERE pipeline_status != ALL($1::text[])
            ORDER BY id
            """,
            terminal,
        )

    count = len(stalled)
    if count == 0:
        logger.info("orchestrator.startup_sweep_nothing_to_do")
        return 0

    logger.info("orchestrator.startup_sweep", requeuing=count)

    for row in stalled:
        task = asyncio.create_task(
            process_conversation(row["id"]),
            name=f"pipeline:sweep:{row['id']}",
        )
        _running_tasks.add(task)
        task.add_done_callback(_running_tasks.discard)

    return count


async def shutdown(wait_timeout: float = 30) -> None:
    """Wait for all in-flight pipeline tasks to complete, up to wait_timeout seconds.

    Tasks still running after the timeout are cancelled. Cancellation errors
    are suppressed — shutdown proceeds regardless.

    Args:
        wait_timeout: Maximum seconds to wait before cancelling outstanding tasks.
    """
    if not _running_tasks:
        logger.info("orchestrator.shutdown_no_tasks")
        return

    in_flight = list(_running_tasks)
    logger.info("orchestrator.shutdown_waiting", task_count=len(in_flight))

    try:
        done, pending = await asyncio.wait(in_flight, timeout=wait_timeout)
    except Exception:
        logger.exception("orchestrator.shutdown_wait_error")
        pending = set(in_flight)
        done = set()

    if pending:
        logger.warning(
            "orchestrator.shutdown_timeout",
            timed_out=len(pending),
            completed=len(done),
        )
        for task in pending:
            task.cancel()
        # Allow cancellations to propagate
        await asyncio.gather(*pending, return_exceptions=True)
    else:
        logger.info("orchestrator.shutdown_clean", completed=len(done))
