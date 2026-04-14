"""Tests for maintenance operations — reconsolidate and type validation."""

from __future__ import annotations

from app.db import get_pool
from app.pipeline.maintenance import (
    _reconsolidate_heuristic,
    _validate_type_heuristic,
    reconsolidate_entity,
    validate_all_types,
    validate_entity_type,
)
from tests.conftest import insert_chunk, insert_conversation, insert_entity


async def _link_entity_to_chunk(entity_id: int, chunk_id: int) -> None:
    """Helper: create entity_chunks junction record."""
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO entity_chunks (entity_id, chunk_id) VALUES ($1, $2)",
            entity_id,
            chunk_id,
        )


async def _create_relation(source_id: int, target_id: int, relation_type: str) -> int:
    """Helper: create a relation and return its ID."""
    pool = get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(
            """
            INSERT INTO relations (source_id, target_id, relation_type, description)
            VALUES ($1, $2, $3, $4)
            RETURNING id
            """,
            source_id,
            target_id,
            relation_type,
            "test relation",
        )


# ---------------------------------------------------------------------------
# Heuristic reconsolidate
# ---------------------------------------------------------------------------


def test_reconsolidate_heuristic_extracts_mentions() -> None:
    entity = {"name": "Postgres", "entity_type": "Tool", "summary": "old"}
    chunks = [
        {
            "raw_content": "Postgres is the primary database. It handles all persistent storage.",
            "chunk_type": "topic",
            "significance": 3,
        },
        {
            "raw_content": "Postgres performance was slow today. Query times were high.",
            "chunk_type": "topic",
            "significance": 3,
        },
    ]
    result = _reconsolidate_heuristic(entity, chunks)
    assert "Postgres" in result
    assert result.endswith(".")


def test_reconsolidate_heuristic_no_mentions_uses_fallback() -> None:
    entity = {"name": "ObscureThing", "entity_type": "Tool", "summary": "existing summary"}
    chunks = [
        {"raw_content": "Nothing relevant here at all.", "chunk_type": "topic", "significance": 3},
    ]
    result = _reconsolidate_heuristic(entity, chunks)
    assert result == "existing summary"


def test_reconsolidate_heuristic_deduplicates() -> None:
    entity = {"name": "NixOS", "entity_type": "Tool", "summary": ""}
    chunks = [
        {"raw_content": "NixOS is great. NixOS is great.", "chunk_type": "topic", "significance": 3},
    ]
    result = _reconsolidate_heuristic(entity, chunks)
    # Should only appear once after dedup
    assert result.count("NixOS is great") == 1


# ---------------------------------------------------------------------------
# Heuristic type validation
# ---------------------------------------------------------------------------


def test_validate_heuristic_person_from_relations() -> None:
    entity = {"id": 99, "name": "Matt", "entity_type": "Person", "summary": ""}
    relations = [
        {"relation_type": "works_at", "role": "source"},
        {"relation_type": "knows", "role": "source"},
    ]
    result = _validate_type_heuristic(entity, relations)
    assert result["suggested_type"] is None  # Person confirmed
    assert result["current_type"] == "Person"


def test_validate_heuristic_detects_mismatch() -> None:
    entity = {"id": 99, "name": "Acme Corp", "entity_type": "Tool", "summary": ""}
    relations = [
        {"relation_type": "works_at", "role": "target"},
        {"relation_type": "member_of", "role": "target"},
    ]
    result = _validate_type_heuristic(entity, relations)
    assert result["suggested_type"] == "Organisation"


def test_validate_heuristic_name_pattern() -> None:
    entity = {"id": 99, "name": "Melbourne", "entity_type": "Tool", "summary": ""}
    relations = []
    result = _validate_type_heuristic(entity, relations)
    assert result["suggested_type"] == "Place"


def test_validate_heuristic_no_signals() -> None:
    entity = {"id": 99, "name": "SomeRandomThing", "entity_type": "Preference", "summary": ""}
    relations = []
    result = _validate_type_heuristic(entity, relations)
    assert result["suggested_type"] is None
    assert result["confidence"] == 1.0


# ---------------------------------------------------------------------------
# DB integration tests
# ---------------------------------------------------------------------------


async def test_reconsolidate_entity_not_found() -> None:
    result = await reconsolidate_entity(99999)
    assert "error" in result


async def test_reconsolidate_entity_no_chunks() -> None:
    eid = await insert_entity(name="Lonely", entity_type="Tool", summary="old summary")
    result = await reconsolidate_entity(eid, mode="heuristic")
    assert result["chunks_used"] == 0
    assert result["skipped"] == "no linked chunks"


async def test_reconsolidate_entity_with_chunks() -> None:
    conv_id = await insert_conversation(pipeline_status="extracted")
    chunk_id = await insert_chunk(
        conversation_id=conv_id,
        raw_content="Neovim is the primary editor. Neovim handles everything.",
    )
    eid = await insert_entity(name="Neovim", entity_type="Tool", summary="old")
    await _link_entity_to_chunk(eid, chunk_id)

    result = await reconsolidate_entity(eid, mode="heuristic")
    assert result["chunks_used"] == 1
    assert "Neovim" in result["new_summary"]
    assert result["old_summary"] == "old"

    # Verify DB was updated
    pool = get_pool()
    async with pool.acquire() as conn:
        summary = await conn.fetchval("SELECT summary FROM entities WHERE id = $1", eid)
    assert "Neovim" in summary


async def test_validate_entity_type_not_found() -> None:
    result = await validate_entity_type(99999)
    assert "error" in result


async def test_validate_entity_type_with_relations() -> None:
    eid = await insert_entity(name="TestOrg", entity_type="Tool")
    # User (id=1) works_at TestOrg → TestOrg should be Organisation
    await _create_relation(1, eid, "works_at")

    result = await validate_entity_type(eid, mode="heuristic")
    assert result["suggested_type"] == "Organisation"
    assert result["current_type"] == "Tool"


async def test_validate_all_types_finds_mismatches() -> None:
    eid = await insert_entity(name="BadType", entity_type="Medication")
    await _create_relation(1, eid, "works_at")  # Target of works_at → should be Org

    result = await validate_all_types(mode="heuristic")
    # Should include singletons (User, Assistant) + our test entity
    assert result["processed"] >= 1
    mismatch_ids = [m["entity_id"] for m in result["mismatches"]]
    assert eid in mismatch_ids
