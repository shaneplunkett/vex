"""Linker — applies validated extraction results to the database.

Creates entities, relations, junction records (entity_chunks, relation_chunks),
and review queue entries. Orchestrates the full validate→route→apply flow
for a single chunk's extraction output.
"""

from __future__ import annotations

from typing import Any

import structlog

from app.db import get_pool
from app.pipeline.validator import route_entity, route_relation, validate_entity, validate_relation

logger = structlog.get_logger()


async def apply_extraction(
    chunk_id: int,
    extraction: dict[str, Any],
) -> dict[str, Any]:
    """Apply a single chunk's extraction results to the database.

    Validates each entity and relation, routes them via the validator,
    then applies the routing decisions (create, link, review queue).

    Args:
        chunk_id: ID of the chunk these extractions came from.
        extraction: Extractor output with 'entities', 'relations', 'flags'.

    Returns:
        Summary dict with counts of applied, created, reviewed, skipped.
    """
    entities = extraction.get("entities", [])
    relations = extraction.get("relations", [])
    flags = extraction.get("flags", [])

    stats = {
        "entities_applied": 0,
        "entities_created": 0,
        "entities_reviewed": 0,
        "entities_rejected": 0,
        "relations_created": 0,
        "relations_superseded": 0,
        "relations_skipped": 0,
        "flags": len(flags),
    }

    # Map of entity name (lowercase) → entity_id for relation resolution
    entity_id_map: dict[str, int] = {}

    pool = get_pool()

    # --- Process entities ---
    for entity in entities:
        errors = validate_entity(entity)
        if errors:
            logger.warning(
                "linker.entity_rejected",
                chunk_id=chunk_id,
                entity_name=entity.get("name"),
                errors=errors,
            )
            stats["entities_rejected"] += 1
            continue

        routing = await route_entity(entity, chunk_id)

        if routing["action"] == "singleton":
            entity_id_map[entity["name"].strip().lower()] = routing["entity_id"]
            # Link singleton to chunk
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO entity_chunks (entity_id, chunk_id)
                    VALUES ($1, $2)
                    ON CONFLICT (entity_id, chunk_id) DO NOTHING
                    """,
                    routing["entity_id"],
                    chunk_id,
                )
            stats["entities_applied"] += 1

        elif routing["action"] == "apply_existing":
            entity_id_map[entity["name"].strip().lower()] = routing["entity_id"]
            async with pool.acquire() as conn, conn.transaction():
                # Update summary if provided
                if entity.get("summary"):
                    await conn.execute(
                        """
                        UPDATE entities SET summary = $1, updated_at = now()
                        WHERE id = $2
                        """,
                        entity["summary"],
                        routing["entity_id"],
                    )
                # Link to chunk
                await conn.execute(
                    """
                    INSERT INTO entity_chunks (entity_id, chunk_id)
                    VALUES ($1, $2)
                    ON CONFLICT (entity_id, chunk_id) DO NOTHING
                    """,
                    routing["entity_id"],
                    chunk_id,
                )
            stats["entities_applied"] += 1

        elif routing["action"] == "create_new":
            async with pool.acquire() as conn, conn.transaction():
                entity_id = await conn.fetchval(
                    """
                    INSERT INTO entities (name, entity_type, summary)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (name, entity_type) DO UPDATE SET
                        summary = COALESCE(EXCLUDED.summary, entities.summary),
                        updated_at = now()
                    RETURNING id
                    """,
                    entity["name"].strip(),
                    entity["entity_type"],
                    entity.get("summary"),
                )
                entity_id_map[entity["name"].strip().lower()] = entity_id

                # Link to chunk
                await conn.execute(
                    """
                    INSERT INTO entity_chunks (entity_id, chunk_id)
                    VALUES ($1, $2)
                    ON CONFLICT (entity_id, chunk_id) DO NOTHING
                    """,
                    entity_id,
                    chunk_id,
                )
            stats["entities_created"] += 1

        elif routing["action"] == "skip":
            logger.debug(
                "linker.entity_skipped",
                chunk_id=chunk_id,
                entity_name=entity.get("name"),
                reason=routing.get("reason"),
            )
            stats["entities_rejected"] += 1

        elif routing["action"] == "review":
            async with pool.acquire() as conn:
                # Check for existing review for this chunk+entity to avoid duplicates on retry
                existing_review = await conn.fetchval(
                    """
                    SELECT id FROM review_queue
                    WHERE chunk_id = $1 AND proposed->>'name' = $2 AND status = 'pending'
                    """,
                    chunk_id,
                    entity.get("name", ""),
                )
                if not existing_review:
                    await conn.execute(
                        """
                        INSERT INTO review_queue (chunk_id, proposed, candidates, reason)
                        VALUES ($1, $2, $3, $4)
                        """,
                        chunk_id,
                        entity,
                        routing.get("candidates", []),
                        routing.get("reason", ""),
                    )
            stats["entities_reviewed"] += 1

    # --- Process relations ---
    for relation in relations:
        errors = validate_relation(relation)
        if errors:
            logger.warning(
                "linker.relation_rejected",
                chunk_id=chunk_id,
                relation=relation,
                errors=errors,
            )
            stats["relations_skipped"] += 1
            continue

        routing = await route_relation(relation, entity_id_map)

        if routing["action"] == "create":
            async with pool.acquire() as conn, conn.transaction():
                # Check for existing duplicate — FOR UPDATE prevents concurrent inserts
                existing_relation = await conn.fetchval(
                    """
                    SELECT id FROM relations
                    WHERE source_id = $1 AND target_id = $2 AND relation_type = $3
                      AND superseded_at IS NULL
                    FOR UPDATE
                    """,
                    routing["source_id"],
                    routing["target_id"],
                    relation["relation_type"],
                )
                if existing_relation:
                    # Just link existing relation to this chunk
                    await conn.execute(
                        """
                        INSERT INTO relation_chunks (relation_id, chunk_id)
                        VALUES ($1, $2)
                        ON CONFLICT (relation_id, chunk_id) DO NOTHING
                        """,
                        existing_relation,
                        chunk_id,
                    )
                    stats["relations_skipped"] += 1
                    continue

                relation_id = await conn.fetchval(
                    """
                    INSERT INTO relations (source_id, target_id, relation_type, description, valid_from)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING id
                    """,
                    routing["source_id"],
                    routing["target_id"],
                    relation["relation_type"],
                    relation.get("description"),
                    relation.get("valid_from"),
                )
                # Link to chunk
                await conn.execute(
                    """
                    INSERT INTO relation_chunks (relation_id, chunk_id)
                    VALUES ($1, $2)
                    ON CONFLICT (relation_id, chunk_id) DO NOTHING
                    """,
                    relation_id,
                    chunk_id,
                )
            stats["relations_created"] += 1

        elif routing["action"] == "supersede":
            async with pool.acquire() as conn, conn.transaction():
                # Re-verify the relation is still active before superseding
                active = await conn.fetchrow(
                    """
                    SELECT id FROM relations
                    WHERE id = $1 AND superseded_at IS NULL
                    FOR UPDATE
                    """,
                    routing["supersede_relation_id"],
                )
                if not active:
                    # Already superseded — find the current active relation and link this chunk
                    current_active = await conn.fetchval(
                        """
                        SELECT id FROM relations
                        WHERE source_id = $1 AND relation_type = $2 AND superseded_at IS NULL
                        """,
                        routing["source_id"],
                        relation["relation_type"],
                    )
                    if current_active:
                        await conn.execute(
                            """
                            INSERT INTO relation_chunks (relation_id, chunk_id)
                            VALUES ($1, $2)
                            ON CONFLICT (relation_id, chunk_id) DO NOTHING
                            """,
                            current_active,
                            chunk_id,
                        )
                    stats["relations_skipped"] += 1
                    continue

                # Create new relation
                new_relation_id = await conn.fetchval(
                    """
                    INSERT INTO relations (source_id, target_id, relation_type, description, valid_from)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING id
                    """,
                    routing["source_id"],
                    routing["target_id"],
                    relation["relation_type"],
                    relation.get("description"),
                    relation.get("valid_from"),
                )
                # Supersede old relation
                await conn.execute(
                    """
                    UPDATE relations
                    SET superseded_at = now(), superseded_by = $1
                    WHERE id = $2
                    """,
                    new_relation_id,
                    routing["supersede_relation_id"],
                )
                # Link to chunk
                await conn.execute(
                    """
                    INSERT INTO relation_chunks (relation_id, chunk_id)
                    VALUES ($1, $2)
                    ON CONFLICT (relation_id, chunk_id) DO NOTHING
                    """,
                    new_relation_id,
                    chunk_id,
                )
            stats["relations_superseded"] += 1

        elif routing["action"] == "skip":
            logger.debug(
                "linker.relation_skipped",
                chunk_id=chunk_id,
                reason=routing.get("reason"),
            )
            stats["relations_skipped"] += 1

    # --- Log flags ---
    if flags:
        logger.info(
            "linker.flags",
            chunk_id=chunk_id,
            flags=flags,
        )

    logger.info(
        "linker.chunk_complete",
        chunk_id=chunk_id,
        **stats,
    )

    return stats


async def apply_conversation_extractions(
    conversation_id: int,
    extraction_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply all extraction results for a conversation.

    Args:
        conversation_id: ID of the conversation.
        extraction_results: List of dicts, each with 'chunk_id' and extraction data.

    Returns:
        Aggregate stats across all chunks.
    """
    totals: dict[str, int] = {
        "chunks_processed": 0,
        "chunks_failed": 0,
        "entities_applied": 0,
        "entities_created": 0,
        "entities_reviewed": 0,
        "entities_rejected": 0,
        "relations_created": 0,
        "relations_superseded": 0,
        "relations_skipped": 0,
        "flags": 0,
    }

    for result in extraction_results:
        try:
            chunk_id = result.get("chunk_id")
            if chunk_id is None:
                totals["chunks_failed"] += 1
                logger.error("linker.missing_chunk_id", result_keys=list(result.keys()))
                continue

            stats = await apply_extraction(chunk_id, result)

            totals["chunks_processed"] += 1
            for key in stats:
                if key in totals:
                    totals[key] += stats[key]
        except Exception:
            totals["chunks_failed"] += 1
            logger.exception("linker.chunk_failed", chunk_id=result.get("chunk_id"))

    # Pipeline status managed by orchestrator (not here) — it checks extraction
    # failure counts to decide whether to advance or leave for retry.

    logger.info(
        "linker.conversation_complete",
        conversation_id=conversation_id,
        **totals,
    )

    return totals
