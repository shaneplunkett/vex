"""Auditor checks — programmatic graph quality detection.

Each check function returns a list of findings. The run_audit() function
orchestrates all checks, stores results in audit_log, and returns a summary.

All checks are purely programmatic — no LLM calls.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

import structlog

from app.db import get_pool

logger = structlog.get_logger()


async def check_duplicate_entities() -> list[dict[str, Any]]:
    """Detect potential duplicate entities via trigram similarity within same type.

    Returns pairs of entities with similarity > 0.6 that might be the same thing.
    """
    pool = get_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT a.id AS id_a, a.name AS name_a,
                   b.id AS id_b, b.name AS name_b,
                   a.entity_type,
                   similarity(lower(a.name), lower(b.name)) AS sim
            FROM entities a
            JOIN entities b ON a.entity_type = b.entity_type AND a.id < b.id
            WHERE similarity(lower(a.name), lower(b.name)) > 0.6
            ORDER BY sim DESC
            LIMIT 50
            """
        )

    return [
        {
            "check": "duplicate_entity",
            "entity_a": {"id": row["id_a"], "name": row["name_a"]},
            "entity_b": {"id": row["id_b"], "name": row["name_b"]},
            "entity_type": row["entity_type"],
            "similarity": round(float(row["sim"]), 3),
        }
        for row in rows
    ]


async def check_orphan_entities() -> list[dict[str, Any]]:
    """Detect entities with no chunk links (orphans).

    Excludes singletons (configured speakers) which may not have chunk links initially.
    """
    pool = get_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT e.id, e.name, e.entity_type, e.created_at
            FROM entities e
            LEFT JOIN entity_chunks ec ON ec.entity_id = e.id
            WHERE ec.entity_id IS NULL
              AND e.id NOT IN (1, 2)
            ORDER BY e.created_at
            LIMIT 100
            """
        )

    return [
        {
            "check": "orphan_entity",
            "entity_id": row["id"],
            "name": row["name"],
            "entity_type": row["entity_type"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


async def check_broken_relations() -> list[dict[str, Any]]:
    """Detect relations referencing non-existent entities."""
    pool = get_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT r.id, r.relation_type, r.source_id, r.target_id
            FROM relations r
            LEFT JOIN entities es ON es.id = r.source_id
            LEFT JOIN entities et ON et.id = r.target_id
            WHERE (es.id IS NULL OR et.id IS NULL)
              AND r.superseded_at IS NULL
            """
        )

    return [
        {
            "check": "broken_relation",
            "relation_id": row["id"],
            "relation_type": row["relation_type"],
            "source_id": row["source_id"],
            "target_id": row["target_id"],
        }
        for row in rows
    ]


async def check_duplicate_relations() -> list[dict[str, Any]]:
    """Detect duplicate active relations (same source, target, type)."""
    pool = get_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT source_id, target_id, relation_type, COUNT(*) AS cnt,
                   array_agg(id ORDER BY created_at) AS relation_ids
            FROM relations
            WHERE superseded_at IS NULL
            GROUP BY source_id, target_id, relation_type
            HAVING COUNT(*) > 1
            ORDER BY cnt DESC
            LIMIT 50
            """
        )

    return [
        {
            "check": "duplicate_relation",
            "source_id": row["source_id"],
            "target_id": row["target_id"],
            "relation_type": row["relation_type"],
            "count": row["cnt"],
            "relation_ids": list(row["relation_ids"]),
        }
        for row in rows
    ]


async def check_stale_entities(days: int = 90) -> list[dict[str, Any]]:
    """Detect entities not accessed in the specified number of days.

    Only flags entities with low access counts — frequently accessed entities
    are likely still relevant even if not recently touched.
    """
    pool = get_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, name, entity_type, access_count, last_accessed_at, updated_at
            FROM entities
            WHERE (last_accessed_at IS NULL OR last_accessed_at < now() - make_interval(days => $1))
              AND access_count < 3
              AND id NOT IN (1, 2)
            ORDER BY COALESCE(last_accessed_at, updated_at)
            LIMIT 100
            """,
            days,
        )

    return [
        {
            "check": "stale_entity",
            "entity_id": row["id"],
            "name": row["name"],
            "entity_type": row["entity_type"],
            "access_count": row["access_count"],
            "last_accessed_at": row["last_accessed_at"],
        }
        for row in rows
    ]


async def run_audit() -> dict[str, Any]:
    """Run all audit checks and store results in audit_log.

    Returns:
        Summary dict with finding counts and audit_log IDs.
    """
    pool = get_pool()

    _check_fn_t = Callable[[], Coroutine[Any, Any, list[dict[str, Any]]]]  # noqa: N806
    checks: dict[str, _check_fn_t] = {
        "duplicate_entities": check_duplicate_entities,
        "orphan_entities": check_orphan_entities,
        "broken_relations": check_broken_relations,
        "duplicate_relations": check_duplicate_relations,
        "stale_entities": check_stale_entities,
    }

    all_findings: dict[str, list[dict[str, Any]]] = {}
    audit_ids: list[int] = []

    for check_name, check_fn in checks.items():
        try:
            findings = await check_fn()
            all_findings[check_name] = findings

            # Store in audit_log
            async with pool.acquire() as conn:
                audit_id = await conn.fetchval(
                    """
                    INSERT INTO audit_log (audit_type, findings)
                    VALUES ($1, $2)
                    RETURNING id
                    """,
                    check_name,
                    findings,
                )
                audit_ids.append(audit_id)

            logger.info(
                "auditor.check_complete",
                check=check_name,
                findings=len(findings),
            )
        except Exception:
            logger.exception("auditor.check_failed", check=check_name)
            all_findings[check_name] = [{"check": check_name, "error": "check failed"}]

    summary = {
        "checks_run": len(checks),
        "finding_counts": {name: len(findings) for name, findings in all_findings.items()},
        "total_findings": sum(len(f) for f in all_findings.values()),
        "audit_log_ids": audit_ids,
    }

    logger.info("auditor.complete", **summary)
    return summary
