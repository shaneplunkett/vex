"""Review queue tools — resolve uncertain extractions, build the learning loop.

get_review_queue(): pending items with chunk context and candidates.
resolve_review(): link to existing entity, add alias, store examples.
resolve_review_new(): confirm as genuinely new entity.
dismiss_review(): skip noise.
"""

from __future__ import annotations

from typing import Any

import structlog

from app.db import get_pool

logger = structlog.get_logger()


async def get_review_queue(
    limit: int = 10,
    status: str = "pending",
) -> list[dict[str, Any]]:
    """Return pending review queue items with chunk context and candidates.

    Args:
        limit: Maximum items to return.
        status: Filter by status ('pending', 'resolved', 'dismissed').

    Returns:
        List of review items with proposed entity, candidates, chunk context.
    """
    limit = max(1, min(50, limit))
    pool = get_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT rq.id, rq.chunk_id, rq.proposed, rq.candidates, rq.reason,
                   rq.status, rq.created_at,
                   c.raw_content AS chunk_content,
                   c.chunk_type, c.significance,
                   conv.name AS conversation_name
            FROM review_queue rq
            JOIN chunks c ON c.id = rq.chunk_id
            JOIN conversations conv ON conv.id = c.conversation_id
            WHERE rq.status = $1
            ORDER BY rq.created_at
            LIMIT $2
            """,
            status,
            limit,
        )

    return [
        {
            "id": row["id"],
            "chunk_id": row["chunk_id"],
            "proposed": row["proposed"],
            "candidates": row["candidates"],
            "reason": row["reason"],
            "status": row["status"],
            "created_at": row["created_at"],
            "chunk_content": row["chunk_content"],
            "chunk_type": row["chunk_type"],
            "significance": row["significance"],
            "conversation_name": row["conversation_name"],
        }
        for row in rows
    ]


async def resolve_review(
    review_id: int,
    entity_id: int,
) -> dict[str, Any]:
    """Resolve a review item to an existing entity.

    Atomically claims the review via FOR UPDATE, then creates entity_chunks link,
    adds the proposed name as an alias, stores extraction example, and sets
    status='resolved'.

    Args:
        review_id: ID of the review queue item.
        entity_id: ID of the existing entity to link to.

    Returns:
        Dict with review_id, entity_id, and action taken.
    """
    pool = get_pool()

    async with pool.acquire() as conn, conn.transaction():
        # Atomic claim — FOR UPDATE prevents concurrent resolution
        review = await conn.fetchrow(
            """
            SELECT id, chunk_id, proposed, candidates, status
            FROM review_queue
            WHERE id = $1
            FOR UPDATE
            """,
            review_id,
        )

        if review is None:
            return {"error": f"Review item {review_id} not found"}
        if review["status"] != "pending":
            return {"error": f"Review item {review_id} is already {review['status']}"}

        # Verify entity exists
        entity = await conn.fetchrow(
            "SELECT id, name, aliases FROM entities WHERE id = $1",
            entity_id,
        )
        if entity is None:
            return {"error": f"Entity {entity_id} not found"}

        proposed = review["proposed"]
        proposed_name = proposed.get("name", "") if isinstance(proposed, dict) else ""
        candidates = review["candidates"] or []

        # Build rejected candidates — all candidates except the resolved one
        rejected = [c for c in candidates if c.get("id") != entity_id]

        # Link entity to chunk
        await conn.execute(
            """
            INSERT INTO entity_chunks (entity_id, chunk_id)
            VALUES ($1, $2)
            ON CONFLICT (entity_id, chunk_id) DO NOTHING
            """,
            entity_id,
            review["chunk_id"],
        )

        # Add proposed name as alias if different from entity name
        alias_added = False
        if proposed_name and proposed_name.lower() != entity["name"].lower():
            current_aliases = list(entity["aliases"] or [])
            if proposed_name not in current_aliases:
                await conn.execute(
                    """
                    UPDATE entities
                    SET aliases = array_append(COALESCE(aliases, '{}'), $1),
                        updated_at = now()
                    WHERE id = $2
                    """,
                    proposed_name,
                    entity_id,
                )
                alias_added = True

        # Store extraction example — use the original proposed name (not corrected)
        # so the learning loop captures the surface form that triggered review
        await conn.execute(
            """
            INSERT INTO extraction_examples (entity_name, resolved_to_id, context_snippet)
            VALUES ($1, $2, $3)
            """,
            proposed_name,
            entity_id,
            proposed.get("reasoning", "") if isinstance(proposed, dict) else "",
        )

        # Resolve the review item
        await conn.execute(
            """
            UPDATE review_queue
            SET status = 'resolved',
                resolved_entity_id = $1,
                rejected_candidates = $2,
                resolved_at = now()
            WHERE id = $3
            """,
            entity_id,
            rejected,
            review_id,
        )

    logger.info(
        "review.resolved",
        review_id=review_id,
        entity_id=entity_id,
        entity_name=entity["name"],
        proposed_name=proposed_name,
    )
    return {
        "review_id": review_id,
        "entity_id": entity_id,
        "entity_name": entity["name"],
        "action": "resolved",
        "alias_added": proposed_name if alias_added else None,
    }


async def resolve_review_new(
    review_id: int,
    name: str,
    entity_type: str,
    summary: str | None = None,
) -> dict[str, Any]:
    """Resolve a review item by confirming it as a genuinely new entity.

    Atomically claims the review via FOR UPDATE, then creates the entity,
    links to chunk, stores extraction example, and sets status='resolved'.

    Args:
        review_id: ID of the review queue item.
        name: Entity name (the corrected/canonical form).
        entity_type: Entity type.
        summary: Optional entity summary.

    Returns:
        Dict with review_id, new entity_id, and action taken.
    """
    pool = get_pool()

    async with pool.acquire() as conn, conn.transaction():
        # Atomic claim
        review = await conn.fetchrow(
            """
            SELECT id, chunk_id, proposed, status
            FROM review_queue
            WHERE id = $1
            FOR UPDATE
            """,
            review_id,
        )

        if review is None:
            return {"error": f"Review item {review_id} not found"}
        if review["status"] != "pending":
            return {"error": f"Review item {review_id} is already {review['status']}"}

        proposed = review["proposed"]
        # Store the original proposed name for the learning loop, not the corrected one
        original_name = proposed.get("name", name) if isinstance(proposed, dict) else name

        # Create the entity
        entity_id = await conn.fetchval(
            """
            INSERT INTO entities (name, entity_type, summary)
            VALUES ($1, $2, $3)
            ON CONFLICT (name, entity_type) DO UPDATE SET
                summary = COALESCE(EXCLUDED.summary, entities.summary),
                updated_at = now()
            RETURNING id
            """,
            name.strip(),
            entity_type,
            summary,
        )

        # Link to chunk
        await conn.execute(
            """
            INSERT INTO entity_chunks (entity_id, chunk_id)
            VALUES ($1, $2)
            ON CONFLICT (entity_id, chunk_id) DO NOTHING
            """,
            entity_id,
            review["chunk_id"],
        )

        # Store extraction example with original surface form
        await conn.execute(
            """
            INSERT INTO extraction_examples (entity_name, resolved_to_id, context_snippet)
            VALUES ($1, $2, $3)
            """,
            original_name,
            entity_id,
            proposed.get("reasoning", "") if isinstance(proposed, dict) else "",
        )

        # Resolve the review item
        await conn.execute(
            """
            UPDATE review_queue
            SET status = 'resolved',
                resolved_entity_id = $1,
                resolved_at = now()
            WHERE id = $2
            """,
            entity_id,
            review_id,
        )

    logger.info(
        "review.resolved_new",
        review_id=review_id,
        entity_id=entity_id,
        name=name,
        entity_type=entity_type,
    )
    return {
        "review_id": review_id,
        "entity_id": entity_id,
        "name": name,
        "entity_type": entity_type,
        "action": "created",
    }


async def dismiss_review(review_id: int) -> dict[str, Any]:
    """Dismiss a review item as noise. No entity created, no links.

    Auto-adds the dismissed entity to the extraction denylist so the same
    name+type won't re-enter the review queue from future extractions.

    Args:
        review_id: ID of the review queue item.

    Returns:
        Dict with review_id and action taken.
    """
    pool = get_pool()

    async with pool.acquire() as conn, conn.transaction():
        # Atomic claim
        review = await conn.fetchrow(
            """
            SELECT id, proposed, status
            FROM review_queue
            WHERE id = $1
            FOR UPDATE
            """,
            review_id,
        )

        if review is None:
            return {"error": f"Review item {review_id} not found"}
        if review["status"] != "pending":
            return {"error": f"Review item {review_id} is already {review['status']}"}

        await conn.execute(
            """
            UPDATE review_queue
            SET status = 'dismissed', resolved_at = now()
            WHERE id = $1
            """,
            review_id,
        )

        # Auto-add to denylist so this name+type doesn't come back
        proposed = review["proposed"]
        proposed_name = proposed.get("name") if isinstance(proposed, dict) else None
        proposed_type = proposed.get("entity_type") if isinstance(proposed, dict) else None

        denylisted = False
        if proposed_name and proposed_type:
            await conn.execute(
                """
                INSERT INTO extraction_denylist (name, entity_type, reason)
                VALUES ($1, $2, $3)
                ON CONFLICT (name, entity_type) DO NOTHING
                """,
                proposed_name.strip().lower(),
                proposed_type,
                f"Dismissed from review queue (review_id={review_id})",
            )
            denylisted = True

    logger.info("review.dismissed", review_id=review_id, denylisted=denylisted)
    return {"review_id": review_id, "action": "dismissed", "denylisted": denylisted}
