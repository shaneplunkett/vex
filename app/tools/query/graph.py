"""neighbourhood() — graph traversal outward from an entity via recursive CTE."""

from __future__ import annotations

from typing import Any

import structlog

from app.db import get_pool

logger = structlog.get_logger()


async def neighbourhood(
    entity_name: str,
    *,
    hops: int = 1,
    relation_types: list[str] | None = None,
) -> dict[str, Any] | None:
    """Graph traversal outward from an entity, returning connected entities and relations.

    Uses a recursive CTE to walk the knowledge graph up to `hops` depth.
    Only follows current (non-superseded) relations. Handles cycles via
    a visited set in the CTE.

    Args:
        entity_name: Name or ID of the starting entity.
        hops: Traversal depth (1-3, default 1).
        relation_types: Optional list of relation types to follow.

    Returns:
        Dict with root entity, discovered entities by hop depth, and all
        traversed relations. None if the starting entity is not found.
    """
    hops = max(1, min(3, hops))
    entity_name = entity_name.strip()
    if not entity_name:
        return None

    pool = get_pool()

    async with pool.acquire() as conn:
        # Resolve starting entity
        root = None
        if entity_name.isdigit():
            root = await conn.fetchrow(
                "SELECT id, name, entity_type, summary FROM entities WHERE id = $1",
                int(entity_name),
            )
        if root is None:
            escaped = entity_name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            root = await conn.fetchrow(
                """
                SELECT id, name, entity_type, summary FROM entities
                WHERE name ILIKE $1 ORDER BY access_count DESC LIMIT 1
                """,
                f"%{escaped}%",
            )

        if root is None:
            logger.info("neighbourhood.not_found", entity_name=entity_name)
            return None

        root_id = root["id"]

        # Build relation type filter
        type_filter = ""
        params: list[Any] = [root_id, hops]
        if relation_types:
            type_filter = "AND r.relation_type = ANY($3)"
            params.append(relation_types)

        # Recursive CTE — walks outward from root, tracking depth and visited nodes
        rows = await conn.fetch(
            f"""
            WITH RECURSIVE graph AS (
                -- Base case: root entity
                SELECT
                    $1::int AS entity_id,
                    0 AS depth,
                    ARRAY[$1::int] AS visited
                UNION ALL
                -- Recursive step: follow relations in both directions
                SELECT
                    CASE
                        WHEN r.source_id = g.entity_id THEN r.target_id
                        ELSE r.source_id
                    END AS entity_id,
                    g.depth + 1 AS depth,
                    g.visited || CASE
                        WHEN r.source_id = g.entity_id THEN r.target_id
                        ELSE r.source_id
                    END
                FROM graph g
                JOIN relations r ON (r.source_id = g.entity_id OR r.target_id = g.entity_id)
                    AND r.superseded_at IS NULL
                    {type_filter}
                WHERE g.depth < $2
                    AND NOT (
                        CASE
                            WHEN r.source_id = g.entity_id THEN r.target_id
                            ELSE r.source_id
                        END = ANY(g.visited)
                    )
            )
            SELECT DISTINCT ON (e.id)
                e.id, e.name, e.entity_type, e.summary,
                g.depth
            FROM graph g
            JOIN entities e ON e.id = g.entity_id
            WHERE g.depth > 0
            ORDER BY e.id, g.depth
            """,
            *params,
        )

        # Fetch all relations between discovered entities (including root)
        discovered_ids = [root_id] + [row["id"] for row in rows]
        relation_rows = await conn.fetch(
            f"""
            SELECT r.id, r.source_id, r.target_id, r.relation_type, r.description,
                   r.valid_from,
                   es.name AS source_name, et.name AS target_name
            FROM relations r
            JOIN entities es ON es.id = r.source_id
            JOIN entities et ON et.id = r.target_id
            WHERE r.superseded_at IS NULL
              AND r.source_id = ANY($1) AND r.target_id = ANY($1)
              {"AND r.relation_type = ANY($2)" if relation_types else ""}
            ORDER BY r.created_at DESC
            """,
            discovered_ids,
            *([] if not relation_types else [relation_types]),
        )

    # Group entities by depth
    entities_by_depth: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        depth = row["depth"]
        entities_by_depth.setdefault(depth, []).append(
            {
                "id": row["id"],
                "name": row["name"],
                "entity_type": row["entity_type"],
                "summary": row["summary"],
            }
        )

    relations = [
        {
            "id": row["id"],
            "source_id": row["source_id"],
            "source_name": row["source_name"],
            "target_id": row["target_id"],
            "target_name": row["target_name"],
            "relation_type": row["relation_type"],
            "description": row["description"],
            "valid_from": row["valid_from"],
        }
        for row in relation_rows
    ]

    result: dict[str, Any] = {
        "root": {
            "id": root["id"],
            "name": root["name"],
            "entity_type": root["entity_type"],
            "summary": root["summary"],
        },
        "entities_by_depth": entities_by_depth,
        "relations": relations,
        "total_entities": sum(len(v) for v in entities_by_depth.values()),
        "total_relations": len(relations),
    }

    logger.info(
        "neighbourhood.complete",
        root_id=root_id,
        root_name=root["name"],
        hops=hops,
        entities=result["total_entities"],
        relations=result["total_relations"],
    )
    return result
