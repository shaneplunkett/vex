"""Write tools — save_conversation, and entity/relation management.

save_conversation(): store conversation + trigger async pipeline.
add_entity(), add_relation(): manual entity/relation creation.
correct_entity(), correct_relation(): fix extraction errors.
merge_entities(): merge duplicates with full audit trail.
remove_from_denylist(): undo an extraction denylist entry.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog

from app.db import get_pool

logger = structlog.get_logger()


async def save_conversation(
    messages: list[dict[str, Any]],
    source: str = "cc",
    name: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Store a conversation and trigger the async pipeline.

    Returns immediately with conversation_id and pipeline_status='pending'.
    Chunking and embedding run in the background.

    Args:
        messages: List of dicts with keys: role, content, timestamp (ISO str), ordinal.
        source: Source identifier ('cc' or 'claude_ai').
        name: Human-readable conversation name.
        session_id: Unique session identifier.

    Returns:
        Dict with conversation_id and pipeline_status.
    """
    from app.pipeline.orchestrator import save_conversation as _save_conversation

    if not messages:
        return {"error": "No messages provided"}

    # Validate messages and derive timestamps
    timestamps = []
    for i, msg in enumerate(messages):
        for key in ("role", "content", "ordinal"):
            if key not in msg:
                return {"error": f"Message {i} missing required key '{key}'"}

        ts = msg.get("timestamp")
        if ts is not None:
            if isinstance(ts, str):
                try:
                    timestamps.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
                except ValueError:
                    return {"error": f"Message {i} has invalid timestamp: {ts}"}
            elif isinstance(ts, datetime):
                timestamps.append(ts)
            else:
                return {"error": f"Message {i} has unsupported timestamp type: {type(ts).__name__}"}

    started_at = min(timestamps) if timestamps else None
    ended_at = max(timestamps) if timestamps else None

    conversation_id = await _save_conversation(
        source=source,
        session_id=session_id,
        name=name,
        messages=messages,
        started_at=started_at,
        ended_at=ended_at,
    )

    logger.info(
        "save_conversation.queued",
        conversation_id=conversation_id,
        message_count=len(messages),
        source=source,
    )
    return {"conversation_id": conversation_id, "pipeline_status": "pending"}


async def add_entity(
    name: str,
    entity_type: str,
    summary: str | None = None,
) -> dict[str, Any]:
    """Manually create an entity. Escape hatch for when extraction misses something.

    Args:
        name: Entity name.
        entity_type: Entity type (e.g. Person, Place, Concept).
        summary: Optional description.

    Returns:
        Dict with entity id and name.
    """
    pool = get_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO entities (name, entity_type, summary)
            VALUES ($1, $2, $3)
            ON CONFLICT (name, entity_type) DO UPDATE SET
                summary = COALESCE(EXCLUDED.summary, entities.summary),
                updated_at = now()
            RETURNING id, (xmax = 0) AS inserted
            """,
            name,
            entity_type,
            summary,
        )

    action = "created" if row["inserted"] else "updated"
    logger.info(f"add_entity.{action}", entity_id=row["id"], name=name, entity_type=entity_type)
    return {"id": row["id"], "name": name, "entity_type": entity_type, "action": action}


async def add_relation(
    source_entity: str,
    target_entity: str,
    relation_type: str,
    description: str | None = None,
    valid_from: str | None = None,
) -> dict[str, Any]:
    """Manually create a relation between two entities.

    Entities are looked up by name (case-insensitive). Both must exist.

    Args:
        source_entity: Name of the source entity.
        target_entity: Name of the target entity.
        relation_type: Type of relationship (e.g. "lives_in", "works_at").
        description: Optional description of the relationship.
        valid_from: ISO datetime string for when the relation became valid.

    Returns:
        Dict with relation id and entity names.
    """
    pool = get_pool()
    parsed_valid_from = None
    if valid_from:
        try:
            parsed_valid_from = datetime.fromisoformat(valid_from.replace("Z", "+00:00"))
        except ValueError:
            return {"error": f"Invalid valid_from format: {valid_from}"}

    async with pool.acquire() as conn:
        source_matches = await conn.fetch(
            "SELECT id, name, entity_type FROM entities WHERE lower(name) = lower($1)",
            source_entity,
        )
        target_matches = await conn.fetch(
            "SELECT id, name, entity_type FROM entities WHERE lower(name) = lower($1)",
            target_entity,
        )

        if not source_matches:
            return {"error": f"Source entity '{source_entity}' not found"}
        if not target_matches:
            return {"error": f"Target entity '{target_entity}' not found"}
        if len(source_matches) > 1:
            options = [f"{r['name']} ({r['entity_type']}, id={r['id']})" for r in source_matches]
            return {"error": f"Ambiguous source entity '{source_entity}': {options}. Use add_relation with entity IDs."}
        if len(target_matches) > 1:
            options = [f"{r['name']} ({r['entity_type']}, id={r['id']})" for r in target_matches]
            return {"error": f"Ambiguous target entity '{target_entity}': {options}. Use add_relation with entity IDs."}

        source_id = source_matches[0]["id"]
        target_id = target_matches[0]["id"]

        relation_id = await conn.fetchval(
            """
            INSERT INTO relations (source_id, target_id, relation_type, description, valid_from)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
            """,
            source_id,
            target_id,
            relation_type,
            description,
            parsed_valid_from,
        )

    logger.info(
        "add_relation.created",
        relation_id=relation_id,
        source=source_entity,
        target=target_entity,
        relation_type=relation_type,
    )
    return {
        "id": relation_id,
        "source": source_entity,
        "target": target_entity,
        "relation_type": relation_type,
    }


async def correct_entity(
    entity_id: int,
    name: str | None = None,
    summary: str | None = None,
    entity_type: str | None = None,
    add_alias: str | None = None,
) -> dict[str, Any]:
    """Fix extraction errors on an entity.

    Args:
        entity_id: ID of the entity to correct.
        name: New name (if renaming).
        summary: Updated summary.
        entity_type: Updated type.
        add_alias: Alias to add to the aliases array.

    Returns:
        Dict with entity id and fields updated.
    """
    import asyncpg

    updates: list[str] = []
    params: list[Any] = []
    param_idx = 1
    updated_fields: list[str] = []

    if name is not None:
        updates.append(f"name = ${param_idx}")
        params.append(name)
        param_idx += 1
        updated_fields.append("name")

    if summary is not None:
        updates.append(f"summary = ${param_idx}")
        params.append(summary)
        param_idx += 1
        updated_fields.append("summary")

    if entity_type is not None:
        updates.append(f"entity_type = ${param_idx}")
        params.append(entity_type)
        param_idx += 1
        updated_fields.append("entity_type")

    if add_alias is not None:
        updates.append(f"aliases = array_append(COALESCE(aliases, '{{}}'), ${param_idx})")
        params.append(add_alias)
        param_idx += 1
        updated_fields.append("add_alias")

    if not updates:
        return {"error": "No fields to update"}

    updates.append("updated_at = now()")
    params.append(entity_id)

    # Atomic UPDATE ... RETURNING — no separate existence check needed
    query = f"UPDATE entities SET {', '.join(updates)} WHERE id = ${param_idx} RETURNING id"

    pool = get_pool()
    async with pool.acquire() as conn:
        try:
            row = await conn.fetchrow(query, *params)
        except asyncpg.UniqueViolationError:
            return {"error": "Name/type conflict: an entity with that name and type already exists"}

    if row is None:
        return {"error": f"Entity {entity_id} not found"}

    logger.info("correct_entity.updated", entity_id=entity_id, fields=updated_fields)
    return {"id": entity_id, "updated_fields": updated_fields}


async def correct_relation(
    relation_id: int,
    description: str | None = None,
    supersede: bool = False,
) -> dict[str, Any]:
    """Fix or supersede a relation.

    Args:
        relation_id: ID of the relation to correct.
        description: Updated description.
        supersede: If True, mark the relation as superseded (soft-delete).

    Returns:
        Dict with relation id and action taken.
    """
    pool = get_pool()

    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT id FROM relations WHERE id = $1",
            relation_id,
        )
        if existing is None:
            return {"error": f"Relation {relation_id} not found"}

        if supersede:  # Supersede takes priority — no point updating description on a retired relation
            await conn.execute(
                "UPDATE relations SET superseded_at = now() WHERE id = $1",
                relation_id,
            )
            logger.info("correct_relation.superseded", relation_id=relation_id)
            return {"id": relation_id, "action": "superseded"}

        if description is not None:
            await conn.execute(
                "UPDATE relations SET description = $1 WHERE id = $2",
                description,
                relation_id,
            )
            logger.info("correct_relation.updated", relation_id=relation_id)
            return {"id": relation_id, "action": "updated", "updated_fields": ["description"]}

        return {"error": "No changes specified"}


async def merge_entities(
    source_ids: list[int],
    target_id: int,
) -> dict[str, Any]:
    """Merge duplicate entities into a single target.

    Reassigns all relations and chunk links from source entities to the target,
    merges aliases, deletes source entities, and logs the merge in audit_log.

    Args:
        source_ids: IDs of the duplicate entities to merge away.
        target_id: ID of the entity to keep.

    Returns:
        Dict with merge details and audit log id.
    """
    if not source_ids:
        return {"error": "No source IDs provided"}

    if target_id in source_ids:
        return {"error": "Target ID cannot be in source IDs"}

    pool = get_pool()

    async with pool.acquire() as conn, conn.transaction():
        # Lock all entities in one ordered query to prevent deadlocks
        all_ids = sorted({target_id, *source_ids})
        all_rows = await conn.fetch(
            "SELECT id, name, aliases FROM entities WHERE id = ANY($1) ORDER BY id FOR UPDATE",
            all_ids,
        )
        rows_by_id = {r["id"]: r for r in all_rows}

        if target_id not in rows_by_id:
            return {"error": f"Target entity {target_id} not found"}
        target = rows_by_id[target_id]

        missing = set(source_ids) - set(rows_by_id.keys())
        if missing:
            return {"error": f"Source entities not found: {missing}"}
        sources = [rows_by_id[sid] for sid in source_ids]

        # Merge aliases from sources into target
        all_aliases = list(target["aliases"] or [])
        for source in sources:
            all_aliases.append(source["name"])  # Source name becomes alias
            all_aliases.extend(source["aliases"] or [])
        # Deduplicate
        unique_aliases = list(dict.fromkeys(all_aliases))

        await conn.execute(
            "UPDATE entities SET aliases = $1, updated_at = now() WHERE id = $2",
            unique_aliases,
            target_id,
        )

        # Reassign relations (source_id references)
        await conn.execute(
            "UPDATE relations SET source_id = $1 WHERE source_id = ANY($2)",
            target_id,
            source_ids,
        )
        # Reassign relations (target_id references)
        await conn.execute(
            "UPDATE relations SET target_id = $1 WHERE target_id = ANY($2)",
            target_id,
            source_ids,
        )

        # Clean up self-referencing relations created by the merge
        # (e.g. if source A had a relation to source B, both now point to target)
        await conn.execute(
            "DELETE FROM relations WHERE source_id = $1 AND target_id = $1",
            target_id,
        )

        # Reassign entity_chunks — handle conflicts from duplicate (entity_id, chunk_id) pairs
        await conn.execute(
            """
            INSERT INTO entity_chunks (entity_id, chunk_id)
            SELECT $1, chunk_id FROM entity_chunks WHERE entity_id = ANY($2)
            ON CONFLICT (entity_id, chunk_id) DO NOTHING
            """,
            target_id,
            source_ids,
        )
        await conn.execute(
            "DELETE FROM entity_chunks WHERE entity_id = ANY($1)",
            source_ids,
        )

        # Delete source entities (relations already reassigned, so RESTRICT won't fire)
        await conn.execute(
            "DELETE FROM entities WHERE id = ANY($1)",
            source_ids,
        )

        # Audit log
        source_names = [s["name"] for s in sources]
        audit_id = await conn.fetchval(
            """
            INSERT INTO audit_log (audit_type, findings, actions_taken)
            VALUES ('entity_merge', $1, $2)
            RETURNING id
            """,
            {"source_ids": source_ids, "source_names": source_names, "target_id": target_id},
            {
                "merged_aliases": unique_aliases,
                "target_name": target["name"],
                "sources_deleted": source_ids,
            },
        )

    logger.info(
        "merge_entities.complete",
        target_id=target_id,
        source_ids=source_ids,
        audit_id=audit_id,
    )
    return {
        "target_id": target_id,
        "target_name": target["name"],
        "merged_from": [s["name"] for s in sources],
        "audit_log_id": audit_id,
    }


async def remove_from_denylist(
    name: str,
    entity_type: str,
) -> dict[str, Any]:
    """Remove an entity from the extraction denylist.

    Use when a previous dismissal was wrong and the entity should be
    allowed through extraction again.

    Args:
        name: Entity name to un-deny.
        entity_type: Entity type to un-deny.

    Returns:
        Dict confirming removal or indicating it wasn't found.
    """
    pool = get_pool()

    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM extraction_denylist WHERE lower(name) = lower($1) AND entity_type = $2",
            name.strip(),
            entity_type,
        )

    removed = result == "DELETE 1"
    if removed:
        logger.info("denylist.removed", name=name, entity_type=entity_type)
    else:
        logger.info("denylist.not_found", name=name, entity_type=entity_type)

    return {
        "name": name,
        "entity_type": entity_type,
        "action": "removed" if removed else "not_found",
    }
