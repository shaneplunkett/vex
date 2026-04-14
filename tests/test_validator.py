"""Tests for the validator module — schema validation, singleton enforcement, routing logic."""

from __future__ import annotations

import pytest

from app.pipeline.validator import (
    _is_singleton,
    validate_entity,
    validate_relation,
)

# ---------------------------------------------------------------------------
# Schema Validation — Entities
# ---------------------------------------------------------------------------


class TestValidateEntity:
    """Tests for validate_entity schema checks."""

    def test_valid_entity(self) -> None:
        entity = {
            "name": "Kubernetes",
            "entity_type": "Tool",
            "match": "new",
            "confidence": 0.9,
            "summary": "Container orchestration platform",
            "reasoning": "Mentioned as infrastructure tool",
        }
        assert validate_entity(entity) == []

    def test_missing_name(self) -> None:
        entity = {"entity_type": "Person", "match": "new", "confidence": 0.9}
        errors = validate_entity(entity)
        assert any("name" in e for e in errors)

    def test_missing_entity_type(self) -> None:
        entity = {"name": "Postgres", "match": "new", "confidence": 0.9}
        errors = validate_entity(entity)
        assert any("entity_type" in e for e in errors)

    def test_missing_match(self) -> None:
        entity = {"name": "Postgres", "entity_type": "HealthCondition", "confidence": 0.9}
        errors = validate_entity(entity)
        assert any("match" in e for e in errors)

    def test_missing_confidence(self) -> None:
        entity = {"name": "Postgres", "entity_type": "HealthCondition", "match": "new"}
        errors = validate_entity(entity)
        assert any("confidence" in e for e in errors)

    def test_invalid_entity_type(self) -> None:
        entity = {
            "name": "Postgres",
            "entity_type": "Disease",
            "match": "new",
            "confidence": 0.9,
        }
        errors = validate_entity(entity)
        assert any("entity_type" in e for e in errors)

    def test_invalid_match_value(self) -> None:
        entity = {
            "name": "Postgres",
            "entity_type": "HealthCondition",
            "match": "maybe",
            "confidence": 0.9,
        }
        errors = validate_entity(entity)
        assert any("match" in e for e in errors)

    def test_confidence_out_of_range_high(self) -> None:
        entity = {
            "name": "Postgres",
            "entity_type": "HealthCondition",
            "match": "new",
            "confidence": 1.5,
        }
        errors = validate_entity(entity)
        assert any("confidence" in e for e in errors)

    def test_confidence_out_of_range_low(self) -> None:
        entity = {
            "name": "Postgres",
            "entity_type": "HealthCondition",
            "match": "new",
            "confidence": -0.1,
        }
        errors = validate_entity(entity)
        assert any("confidence" in e for e in errors)

    def test_confidence_not_numeric(self) -> None:
        entity = {
            "name": "Postgres",
            "entity_type": "HealthCondition",
            "match": "new",
            "confidence": "high",
        }
        errors = validate_entity(entity)
        assert any("numeric" in e for e in errors)

    def test_empty_name(self) -> None:
        entity = {
            "name": "  ",
            "entity_type": "HealthCondition",
            "match": "new",
            "confidence": 0.9,
        }
        errors = validate_entity(entity)
        assert any("empty" in e for e in errors)

    def test_all_16_entity_types_accepted(self) -> None:
        """Every entity type in the schema should be valid."""
        from app.pipeline.extractor import ENTITY_TYPES

        for etype in ENTITY_TYPES:
            entity = {
                "name": "Test",
                "entity_type": etype,
                "match": "new",
                "confidence": 0.9,
            }
            assert validate_entity(entity) == [], f"Type {etype} rejected"


# ---------------------------------------------------------------------------
# Schema Validation — Relations
# ---------------------------------------------------------------------------


class TestValidateRelation:
    """Tests for validate_relation schema checks."""

    def test_valid_relation(self) -> None:
        relation = {
            "source": "User",
            "target": "Postgres",
            "relation_type": "uses",
            "description": "Uses Postgres as primary database",
        }
        assert validate_relation(relation) == []

    def test_missing_source(self) -> None:
        relation = {"target": "Postgres", "relation_type": "uses"}
        errors = validate_relation(relation)
        assert any("source" in e for e in errors)

    def test_missing_target(self) -> None:
        relation = {"source": "User", "relation_type": "uses"}
        errors = validate_relation(relation)
        assert any("target" in e for e in errors)

    def test_missing_relation_type(self) -> None:
        relation = {"source": "User", "target": "Postgres"}
        errors = validate_relation(relation)
        assert any("relation_type" in e for e in errors)

    def test_invalid_relation_type(self) -> None:
        relation = {
            "source": "User",
            "target": "Postgres",
            "relation_type": "suffers_from",
        }
        errors = validate_relation(relation)
        assert any("relation_type" in e for e in errors)

    def test_all_21_relation_types_accepted(self) -> None:
        """Every relation type in the schema should be valid."""
        from app.pipeline.extractor import RELATION_TYPES

        for rtype in RELATION_TYPES:
            relation = {
                "source": "A",
                "target": "B",
                "relation_type": rtype,
            }
            assert validate_relation(relation) == [], f"Type {rtype} rejected"


# ---------------------------------------------------------------------------
# Singleton Enforcement
# ---------------------------------------------------------------------------


class TestSingletonEnforcement:
    """Tests for singleton entity detection."""

    def test_human_speaker_is_singleton(self) -> None:
        assert _is_singleton("User") == 1

    def test_assistant_speaker_is_singleton(self) -> None:
        assert _is_singleton("Assistant") == 2

    def test_human_speaker_case_insensitive(self) -> None:
        assert _is_singleton("USER") == 1
        assert _is_singleton("user") == 1

    def test_assistant_speaker_case_insensitive(self) -> None:
        assert _is_singleton("ASSISTANT") == 2
        assert _is_singleton("assistant") == 2

    def test_non_singleton(self) -> None:
        assert _is_singleton("Dr Jennifer Sharpe") is None

    def test_whitespace_stripped(self) -> None:
        assert _is_singleton("  User  ") == 1


# ---------------------------------------------------------------------------
# Routing — DB-dependent tests
# ---------------------------------------------------------------------------


class TestRouteEntity:
    """Tests for route_entity that require the DB."""

    @pytest.fixture(autouse=True)
    async def _setup(self, _setup_db: None) -> None:
        """Ensure DB is ready and settings use test DB."""

    async def test_singleton_routes_to_singleton(self) -> None:
        from app.pipeline.validator import route_entity

        entity = {
            "name": "User",
            "entity_type": "Person",
            "match": "existing",
            "confidence": 0.99,
        }
        result = await route_entity(entity, chunk_id=1)
        assert result["action"] == "singleton"
        assert result["entity_id"] == 1

    async def test_new_entity_high_confidence_creates(self) -> None:
        from app.pipeline.validator import route_entity

        entity = {
            "name": "Redis",
            "entity_type": "Tool",
            "match": "new",
            "confidence": 0.9,
        }
        result = await route_entity(entity, chunk_id=1)
        assert result["action"] == "create_new"

    async def test_new_entity_low_confidence_reviews(self) -> None:
        from app.pipeline.validator import route_entity

        entity = {
            "name": "SomeNewThing",
            "entity_type": "Tool",
            "match": "new",
            "confidence": 0.5,
        }
        result = await route_entity(entity, chunk_id=1)
        assert result["action"] == "review"

    async def test_uncertain_always_reviews(self) -> None:
        from app.pipeline.validator import route_entity

        entity = {
            "name": "UnknownEntity",
            "entity_type": "Person",
            "match": "uncertain",
            "confidence": 0.95,
        }
        result = await route_entity(entity, chunk_id=1)
        assert result["action"] == "review"

    async def test_exact_match_high_confidence_applies(self) -> None:
        from app.db import get_pool
        from app.pipeline.validator import route_entity

        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO entities (name, entity_type, summary) VALUES ($1, $2, $3)",
                "Neovim",
                "Tool",
                "Terminal editor",
            )

        entity = {
            "name": "Neovim",
            "entity_type": "Tool",
            "match": "existing",
            "confidence": 0.9,
        }
        result = await route_entity(entity, chunk_id=1)
        assert result["action"] == "apply_existing"

    async def test_exact_match_low_confidence_reviews(self) -> None:
        from app.db import get_pool
        from app.pipeline.validator import route_entity

        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO entities (name, entity_type, summary) VALUES ($1, $2, $3)",
                "Obsidian",
                "Tool",
                "Note taking app",
            )

        entity = {
            "name": "Obsidian",
            "entity_type": "Tool",
            "match": "existing",
            "confidence": 0.5,
        }
        result = await route_entity(entity, chunk_id=1)
        assert result["action"] == "review"

    async def test_cross_type_match_reviews(self) -> None:
        from app.db import get_pool
        from app.pipeline.validator import route_entity

        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO entities (name, entity_type) VALUES ($1, $2)",
                "Python",
                "Skill",
            )

        entity = {
            "name": "Python",
            "entity_type": "Tool",
            "match": "new",
            "confidence": 0.9,
        }
        result = await route_entity(entity, chunk_id=1)
        assert result["action"] == "review"
        assert any(c["type"] == "Skill" for c in result["candidates"])

    async def test_denylisted_entity_skips(self) -> None:
        from app.db import get_pool
        from app.pipeline.validator import route_entity

        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO extraction_denylist (name, entity_type, reason) VALUES ($1, $2, $3)",
                "Daily Standup",
                "Preference",
                "Too generic for entity extraction",
            )

        entity = {
            "name": "Daily Standup",
            "entity_type": "Preference",
            "match": "new",
            "confidence": 0.9,
        }
        result = await route_entity(entity, chunk_id=1)
        assert result["action"] == "skip"
        assert "Denylisted" in result["reason"]

    async def test_denylist_case_insensitive(self) -> None:
        from app.db import get_pool
        from app.pipeline.validator import route_entity

        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO extraction_denylist (name, entity_type) VALUES ($1, $2)",
                "Rice Cooker",
                "Tool",
            )

        entity = {
            "name": "rice cooker",
            "entity_type": "Tool",
            "match": "new",
            "confidence": 0.9,
        }
        result = await route_entity(entity, chunk_id=1)
        assert result["action"] == "skip"

    async def test_denylist_different_type_not_blocked(self) -> None:
        from app.db import get_pool
        from app.pipeline.validator import route_entity

        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO extraction_denylist (name, entity_type) VALUES ($1, $2)",
                "Boot.dev",
                "Tool",
            )

        # Same name but different type should NOT be blocked
        entity = {
            "name": "Boot.dev",
            "entity_type": "Organisation",
            "match": "new",
            "confidence": 0.9,
        }
        result = await route_entity(entity, chunk_id=1)
        assert result["action"] != "skip"
