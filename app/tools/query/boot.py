"""boot() — session-start context snapshot."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog

from app.db import get_pool
from app.tools.query._common import MELBOURNE_TZ

logger = structlog.get_logger()


async def boot() -> dict[str, Any]:
    """Return structured session-start context.

    Aggregates recently active entities and pipeline health counts
    into a compact snapshot for session initialisation.

    Returns:
        Dict with keys: today, significant_entities, pipeline_health.
    """
    pool = get_pool()
    now_melb = datetime.now(tz=MELBOURNE_TZ)
    today = now_melb.date()

    async with pool.acquire() as conn:
        # Recently significant entities (updated in last 7 days, access_count >= 3)
        entity_rows = await conn.fetch(
            """
            SELECT id, name, entity_type, summary, access_count, updated_at
            FROM entities
            WHERE updated_at >= now() - INTERVAL '7 days'
              AND access_count >= 3
            ORDER BY access_count DESC, updated_at DESC
            LIMIT 10
            """,
        )
        significant_entities = [
            {
                "id": row["id"],
                "name": row["name"],
                "entity_type": row["entity_type"],
                "summary": row["summary"],
                "access_count": row["access_count"],
            }
            for row in entity_rows
        ]

        # Pipeline health counts (all hit indexed columns)
        review_pending = await conn.fetchval("SELECT COUNT(*) FROM review_queue WHERE status = 'pending'")
        ingestion_pending = await conn.fetchval(
            "SELECT COUNT(*) FROM conversations WHERE pipeline_status NOT IN ('complete', 'failed', 'embedded')"
        )
        pipeline_failed = await conn.fetchval("SELECT COUNT(*) FROM conversations WHERE pipeline_status = 'failed'")

    result: dict[str, Any] = {
        "today": str(today),
        "significant_entities": significant_entities,
        "pipeline_health": {
            "review_pending": review_pending,
            "ingestion_pending": ingestion_pending,
            "pipeline_failed": pipeline_failed,
        },
    }

    logger.info(
        "boot.complete",
        today=str(today),
        significant_entities=len(significant_entities),
        review_pending=review_pending,
        ingestion_pending=ingestion_pending,
        pipeline_failed=pipeline_failed,
    )
    return result
