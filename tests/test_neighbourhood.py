"""Tests for neighbourhood() — graph traversal via recursive CTE."""

from __future__ import annotations

import pytest

from app.db import get_pool


@pytest.fixture
async def graph_data(_setup_db: None) -> dict[str, int]:
    """Create a small test graph.

    User -[uses]-> Postgres -[built_on]-> pgvector
    User -[uses]-> Redis
    User -[uses]-> Neovim
    Postgres -[part_of]-> Data Platform
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        # User is id=1 (singleton)
        postgres_id = await conn.fetchval(
            "INSERT INTO entities (name, entity_type, summary) VALUES ($1, $2, $3) RETURNING id",
            "Postgres",
            "Tool",
            "Primary relational database",
        )
        redis_id = await conn.fetchval(
            "INSERT INTO entities (name, entity_type, summary) VALUES ($1, $2, $3) RETURNING id",
            "Redis",
            "Tool",
            "In-memory cache and message broker",
        )
        pgvector_id = await conn.fetchval(
            "INSERT INTO entities (name, entity_type, summary) VALUES ($1, $2, $3) RETURNING id",
            "pgvector",
            "Tool",
            "Vector similarity search extension for Postgres",
        )
        neovim_id = await conn.fetchval(
            "INSERT INTO entities (name, entity_type, summary) VALUES ($1, $2, $3) RETURNING id",
            "Neovim",
            "Tool",
            "Terminal text editor",
        )
        platform_id = await conn.fetchval(
            "INSERT INTO entities (name, entity_type, summary) VALUES ($1, $2, $3) RETURNING id",
            "Data Platform",
            "Infrastructure",
            "Core data infrastructure",
        )

        # Relations
        await conn.execute(
            "INSERT INTO relations (source_id, target_id, relation_type, description) VALUES ($1, $2, $3, $4)",
            1,
            postgres_id,
            "uses",
            "Primary database",
        )
        await conn.execute(
            "INSERT INTO relations (source_id, target_id, relation_type, description) VALUES ($1, $2, $3, $4)",
            1,
            redis_id,
            "uses",
            "Caching layer",
        )
        await conn.execute(
            "INSERT INTO relations (source_id, target_id, relation_type, description) VALUES ($1, $2, $3, $4)",
            postgres_id,
            pgvector_id,
            "built_on",
            "Postgres extended with pgvector",
        )
        await conn.execute(
            "INSERT INTO relations (source_id, target_id, relation_type, description) VALUES ($1, $2, $3, $4)",
            1,
            neovim_id,
            "uses",
            "Primary editor",
        )
        await conn.execute(
            "INSERT INTO relations (source_id, target_id, relation_type, description) VALUES ($1, $2, $3, $4)",
            postgres_id,
            platform_id,
            "part_of",
            "Postgres is part of Data Platform",
        )

    return {
        "user": 1,
        "postgres": postgres_id,
        "redis": redis_id,
        "pgvector": pgvector_id,
        "neovim": neovim_id,
        "platform": platform_id,
    }


class TestNeighbourhood:
    """Tests for neighbourhood graph traversal."""

    async def test_one_hop_from_user(self, graph_data: dict[str, int]) -> None:
        from app.tools.query import neighbourhood

        result = await neighbourhood("User", hops=1)
        assert result is not None
        assert result["root"]["name"] == "User"

        # User has 3 direct connections: Postgres, Redis, Neovim
        assert result["total_entities"] == 3
        hop1_names = {e["name"] for e in result["entities_by_depth"][1]}
        assert hop1_names == {"Postgres", "Redis", "Neovim"}

    async def test_two_hops_from_user(self, graph_data: dict[str, int]) -> None:
        from app.tools.query import neighbourhood

        result = await neighbourhood("User", hops=2)
        assert result is not None

        # Hop 1: Postgres, Redis, Neovim
        # Hop 2: pgvector (via Postgres->built_on), Data Platform (via Postgres->part_of)
        assert result["total_entities"] == 5
        hop2_names = {e["name"] for e in result["entities_by_depth"].get(2, [])}
        assert "pgvector" in hop2_names
        assert "Data Platform" in hop2_names

    async def test_relation_type_filter(self, graph_data: dict[str, int]) -> None:
        from app.tools.query import neighbourhood

        result = await neighbourhood("User", hops=1, relation_types=["uses"])
        assert result is not None

        # All direct connections are "uses" relations
        hop1_names = {e["name"] for e in result["entities_by_depth"][1]}
        assert hop1_names == {"Postgres", "Redis", "Neovim"}

    async def test_from_postgres(self, graph_data: dict[str, int]) -> None:
        from app.tools.query import neighbourhood

        result = await neighbourhood("Postgres", hops=1)
        assert result is not None
        assert result["root"]["name"] == "Postgres"

        # Postgres connects to: User (inbound uses), pgvector, Data Platform
        hop1_names = {e["name"] for e in result["entities_by_depth"][1]}
        assert "User" in hop1_names
        assert "pgvector" in hop1_names
        assert "Data Platform" in hop1_names

    @pytest.mark.usefixtures("_setup_db")
    async def test_not_found(self) -> None:
        from app.tools.query import neighbourhood

        result = await neighbourhood("NonexistentEntity")
        assert result is None

    async def test_by_id(self, graph_data: dict[str, int]) -> None:
        from app.tools.query import neighbourhood

        result = await neighbourhood("1", hops=1)
        assert result is not None
        assert result["root"]["name"] == "User"

    async def test_hops_clamped(self, graph_data: dict[str, int]) -> None:
        from app.tools.query import neighbourhood

        # hops=10 should be clamped to 3 — verify by comparing results
        result_clamped = await neighbourhood("User", hops=10)
        result_three = await neighbourhood("User", hops=3)
        assert result_clamped is not None
        assert result_three is not None
        assert result_clamped["total_entities"] == result_three["total_entities"]
        # Also verify no entity appears at depth > 3
        for depth in result_clamped["entities_by_depth"]:
            assert depth <= 3

    async def test_superseded_relations_excluded(self, graph_data: dict[str, int]) -> None:
        from app.db import get_pool
        from app.tools.query import neighbourhood

        # Supersede the Postgres->pgvector relation
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE relations SET superseded_at = now() WHERE source_id = $1 AND target_id = $2",
                graph_data["postgres"],
                graph_data["pgvector"],
            )

        result = await neighbourhood("Postgres", hops=1)
        assert result is not None
        hop1_names = {e["name"] for e in result["entities_by_depth"].get(1, [])}
        assert "pgvector" not in hop1_names

    async def test_relations_between_discovered(self, graph_data: dict[str, int]) -> None:
        from app.tools.query import neighbourhood

        result = await neighbourhood("User", hops=2)
        assert result is not None

        # Should include relations between discovered entities
        relation_types = {r["relation_type"] for r in result["relations"]}
        assert "uses" in relation_types
        assert "built_on" in relation_types
