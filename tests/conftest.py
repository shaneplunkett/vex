"""Test infrastructure — testcontainers Postgres fixture with pgvector."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from testcontainers.postgres import PostgresContainer

from app.config import Settings, reset_settings
from app.db import close_pool, create_pool, get_pool, run_migrations


@pytest.fixture(scope="session")
def pg_container() -> PostgresContainer:
    """Session-scoped Postgres container with pgvector extension.

    Starts once for the entire test session, torn down at the end.
    """
    container = PostgresContainer(
        image="pgvector/pgvector:pg16",
        username="test",
        password="test",
        dbname="vex_brain_test",
    )
    with container:
        yield container


@pytest.fixture(scope="session")
def database_url(pg_container: PostgresContainer) -> str:
    """Connection string for the test database."""
    url = pg_container.get_connection_url()
    # testcontainers returns "postgresql+psycopg2://..." — asyncpg needs plain "postgresql://"
    if "+psycopg2" in url:
        url = url.replace("+psycopg2", "")
    return url


@pytest.fixture(scope="session")
async def _run_migrations(database_url: str) -> AsyncIterator[None]:
    """Session-scoped: run migrations once when the container starts."""
    reset_settings()
    import app.config as config_module

    config_module._settings = Settings(database_url=database_url)

    await create_pool()
    try:
        await run_migrations()
        yield
    finally:
        await close_pool()
        reset_settings()


@pytest.fixture(autouse=True)
async def _setup_db(database_url: str, _run_migrations: None) -> AsyncIterator[None]:
    """Per-test fixture: creates pool, yields, then truncates for isolation."""
    reset_settings()
    import app.config as config_module

    config_module._settings = Settings(database_url=database_url)

    await create_pool()
    try:
        yield
    finally:
        # Truncate all data tables for test isolation (preserve schema).
        # Runs in finally so cleanup happens even if test raises.
        try:
            pool = get_pool()
            async with pool.acquire() as conn:
                await conn.execute("""
                    TRUNCATE
                        messages, chunks, entity_chunks, relation_chunks,
                        relations, review_queue, extraction_examples,
                        audit_log, persona
                    CASCADE
                """)
                await conn.execute("TRUNCATE conversations CASCADE")
                await conn.execute("""
                    DELETE FROM entities WHERE id NOT IN (1, 2)
                """)
        finally:
            await close_pool()
            reset_settings()


# ---------------------------------------------------------------------------
# Helper functions for inserting test data
# ---------------------------------------------------------------------------


async def insert_conversation(
    *,
    source: str = "cc",
    session_id: str = "test-session-001",
    name: str = "Test conversation",
    message_count: int = 0,
    pipeline_status: str = "pending",
) -> int:
    """Insert a test conversation and return its ID."""
    pool = get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(
            """
            INSERT INTO conversations (source, session_id, name, message_count, pipeline_status)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
            """,
            source,
            session_id,
            name,
            message_count,
            pipeline_status,
        )


async def insert_message(
    *,
    conversation_id: int,
    role: str = "human",
    content: str = "Hello",
    ordinal: int = 0,
) -> int:
    """Insert a test message and return its ID."""
    pool = get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(
            """
            INSERT INTO messages (conversation_id, role, content, ordinal)
            VALUES ($1, $2, $3, $4)
            RETURNING id
            """,
            conversation_id,
            role,
            content,
            ordinal,
        )


async def insert_chunk(
    *,
    conversation_id: int,
    raw_content: str = "Test chunk content",
    chunk_type: str = "topic",
    significance: int = 3,
    start_ordinal: int = 0,
    end_ordinal: int = 1,
) -> int:
    """Insert a test chunk and return its ID."""
    pool = get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(
            """
            INSERT INTO chunks (conversation_id, content, raw_content, chunk_type, significance,
                                start_ordinal, end_ordinal)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id
            """,
            conversation_id,
            raw_content,
            raw_content,
            chunk_type,
            significance,
            start_ordinal,
            end_ordinal,
        )


async def insert_entity(
    *,
    name: str,
    entity_type: str,
    summary: str | None = None,
) -> int:
    """Insert a test entity and return its ID."""
    pool = get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(
            """
            INSERT INTO entities (name, entity_type, summary)
            VALUES ($1, $2, $3)
            RETURNING id
            """,
            name,
            entity_type,
            summary,
        )
