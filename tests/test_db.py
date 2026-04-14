"""Tests for database schema, migrations, and basic operations."""

from __future__ import annotations

import asyncpg.exceptions
import pytest

from app.db import get_pool
from tests.conftest import insert_conversation, insert_entity, insert_message


async def test_migrations_applied() -> None:
    """Verify that migrations have been applied successfully."""
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT filename FROM applied_migrations ORDER BY id")
        filenames = [r["filename"] for r in rows]
        assert "001_initial_schema.sql" in filenames


async def test_pgvector_extension_available() -> None:
    """Verify pgvector extension is loaded and functional."""
    pool = get_pool()
    async with pool.acquire() as conn:
        # Check extension exists
        result = await conn.fetchval("SELECT count(*) FROM pg_extension WHERE extname = 'vector'")
        assert result == 1

        # Verify vector operations work
        similarity = await conn.fetchval("SELECT 1 - ('[1,0,0]'::vector(3) <=> '[0,1,0]'::vector(3))")
        assert similarity is not None


async def test_singleton_entities_exist() -> None:
    """Verify singleton entities were created by migration."""
    pool = get_pool()
    async with pool.acquire() as conn:
        user = await conn.fetchrow("SELECT * FROM entities WHERE id = 1")
        assert user is not None
        assert user["name"] == "User"
        assert user["entity_type"] == "Person"

        assistant = await conn.fetchrow("SELECT * FROM entities WHERE id = 2")
        assert assistant is not None
        assert assistant["name"] == "Assistant"
        assert assistant["entity_type"] == "Person"


async def test_conversation_insert_roundtrip() -> None:
    """Verify basic conversation + message insert and select."""
    conv_id = await insert_conversation(
        session_id="roundtrip-test",
        name="Test roundtrip",
        message_count=2,
    )

    msg1_id = await insert_message(
        conversation_id=conv_id,
        role="human",
        content="Hello there",
        ordinal=0,
    )
    msg2_id = await insert_message(
        conversation_id=conv_id,
        role="assistant",
        content="Hi! How can I help?",
        ordinal=1,
    )

    pool = get_pool()
    async with pool.acquire() as conn:
        conv = await conn.fetchrow("SELECT * FROM conversations WHERE id = $1", conv_id)
        assert conv["session_id"] == "roundtrip-test"
        assert conv["pipeline_status"] == "pending"

        messages = await conn.fetch(
            "SELECT * FROM messages WHERE conversation_id = $1 ORDER BY ordinal",
            conv_id,
        )
        assert len(messages) == 2
        assert messages[0]["id"] == msg1_id
        assert messages[0]["role"] == "human"
        assert messages[1]["id"] == msg2_id
        assert messages[1]["role"] == "assistant"


async def test_session_id_unique_constraint() -> None:
    """Verify session_id UNIQUE constraint prevents duplicates."""
    await insert_conversation(session_id="unique-test")

    with pytest.raises(asyncpg.exceptions.UniqueViolationError):
        await insert_conversation(session_id="unique-test")


async def test_message_ordinal_unique_per_conversation() -> None:
    """Verify (conversation_id, ordinal) uniqueness."""
    conv_id = await insert_conversation(session_id="ordinal-test")
    await insert_message(conversation_id=conv_id, ordinal=0)

    with pytest.raises(asyncpg.exceptions.UniqueViolationError):
        await insert_message(conversation_id=conv_id, ordinal=0)


async def test_cascade_delete_conversation() -> None:
    """Verify CASCADE: deleting conversation removes its messages."""
    conv_id = await insert_conversation(session_id="cascade-test")
    await insert_message(conversation_id=conv_id, ordinal=0)
    await insert_message(conversation_id=conv_id, ordinal=1)

    pool = get_pool()
    async with pool.acquire() as conn:
        # Verify messages exist
        count = await conn.fetchval(
            "SELECT count(*) FROM messages WHERE conversation_id = $1",
            conv_id,
        )
        assert count == 2

        # Delete conversation
        await conn.execute("DELETE FROM conversations WHERE id = $1", conv_id)

        # Messages should be gone
        count = await conn.fetchval(
            "SELECT count(*) FROM messages WHERE conversation_id = $1",
            conv_id,
        )
        assert count == 0


async def test_entity_unique_constraint() -> None:
    """Verify (name, entity_type) uniqueness on entities."""
    await insert_entity(name="TestEntity", entity_type="Tool")

    with pytest.raises(asyncpg.exceptions.UniqueViolationError):
        await insert_entity(name="TestEntity", entity_type="Tool")


async def test_entity_different_types_allowed() -> None:
    """Same name with different entity_type should be allowed."""
    id1 = await insert_entity(name="Python", entity_type="Tool")
    id2 = await insert_entity(name="Python", entity_type="Skill")
    assert id1 != id2


async def test_relations_restrict_entity_delete() -> None:
    """Verify ON DELETE RESTRICT: can't delete entity with active relations."""
    e1 = await insert_entity(name="Source", entity_type="Person")
    e2 = await insert_entity(name="Target", entity_type="Organisation")

    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO relations (source_id, target_id, relation_type)
            VALUES ($1, $2, 'works_at')
            """,
            e1,
            e2,
        )

        # Should fail — entity has relations
        with pytest.raises(asyncpg.exceptions.ForeignKeyViolationError):
            await conn.execute("DELETE FROM entities WHERE id = $1", e1)


async def test_full_text_search_messages() -> None:
    """Verify FTS index works on messages."""
    conv_id = await insert_conversation(session_id="fts-test")
    await insert_message(
        conversation_id=conv_id,
        role="human",
        content="I love writing Python code for data pipelines",
        ordinal=0,
    )
    await insert_message(
        conversation_id=conv_id,
        role="assistant",
        content="TypeScript is great for frontend work",
        ordinal=1,
    )

    pool = get_pool()
    async with pool.acquire() as conn:
        results = await conn.fetch(
            """
            SELECT content FROM messages
            WHERE to_tsvector('english', content) @@ plainto_tsquery('english', 'python pipeline')
            """
        )
        assert len(results) == 1
        assert "Python" in results[0]["content"]

        # Verify the GIN index actually exists in the catalog
        idx = await conn.fetchval(
            """
            SELECT count(*) FROM pg_indexes
            WHERE tablename = 'messages' AND indexname = 'idx_messages_fts'
            """
        )
        assert idx == 1


async def test_pipeline_status_check_constraint() -> None:
    """Verify pipeline_status CHECK constraint rejects invalid values."""
    pool = get_pool()
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute(
                """
                INSERT INTO conversations (source, session_id, name, pipeline_status)
                VALUES ('cc', 'check-test', 'test', 'invalid_status')
                """
            )


async def test_role_check_constraint() -> None:
    """Verify message role CHECK constraint."""
    conv_id = await insert_conversation(session_id="role-check-test")

    pool = get_pool()
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, ordinal)
                VALUES ($1, 'system', 'not a valid role', 0)
                """,
                conv_id,
            )
