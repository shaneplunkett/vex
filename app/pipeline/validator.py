"""Validator — programmatic validation and routing of extraction output.

No LLM calls. Takes extractor output and routes each entity/relation to one of:
- auto-apply (confident match or confident new)
- review queue (uncertain, fuzzy match, below threshold)
- reject (schema violation, invalid)

Singleton enforcement: configured speaker names → id=1 and id=2. Always.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog

from app.config import get_settings
from app.db import get_pool
from app.pipeline.extractor import ENTITY_TYPES, RELATION_TYPES

logger = structlog.get_logger()


def _get_singletons() -> dict[str, int]:
    """Build singleton map from config settings."""
    settings = get_settings()
    return {
        settings.human_speaker.lower(): 1,
        settings.assistant_speaker.lower(): 2,
    }


def _is_singleton(name: str) -> int | None:
    """Return singleton entity ID if name matches a configured speaker, else None."""
    return _get_singletons().get(name.strip().lower())


# ---------------------------------------------------------------------------
# Schema Validation
# ---------------------------------------------------------------------------


def validate_entity(entity: dict[str, Any]) -> list[str]:
    """Validate a single extracted entity. Returns list of error strings (empty = valid)."""
    errors = []

    for field in ("name", "entity_type", "match", "confidence"):
        if field not in entity:
            errors.append(f"Missing required field: {field}")

    if not errors:
        if not isinstance(entity["name"], str) or not entity["name"].strip():
            errors.append("name must be a non-empty string")

        if not isinstance(entity["entity_type"], str):
            errors.append("entity_type must be a string")
        elif entity["entity_type"] not in ENTITY_TYPES:
            errors.append(f"Invalid entity_type: {entity['entity_type']}")

        if entity["match"] not in ("existing", "new", "uncertain"):
            errors.append(f"Invalid match value: {entity['match']}")

        if not isinstance(entity["confidence"], (int, float)):
            errors.append(f"confidence must be numeric, got {type(entity['confidence']).__name__}")
        elif not 0.0 <= entity["confidence"] <= 1.0:
            errors.append(f"confidence out of range: {entity['confidence']}")

        # Optional text fields must be strings if present (they go into text columns)
        for opt_field in ("summary", "reasoning"):
            val = entity.get(opt_field)
            if val is not None and not isinstance(val, str):
                errors.append(f"{opt_field} must be a string if provided")

    return errors


def validate_relation(relation: dict[str, Any]) -> list[str]:
    """Validate a single extracted relation. Returns list of error strings (empty = valid)."""
    errors = []

    for field in ("source", "target", "relation_type"):
        if field not in relation:
            errors.append(f"Missing required field: {field}")

    if not errors:
        for field in ("source", "target"):
            if not isinstance(relation[field], str) or not relation[field].strip():
                errors.append(f"{field} must be a non-empty string")

        if not isinstance(relation["relation_type"], str):
            errors.append("relation_type must be a string")
        elif relation["relation_type"] not in RELATION_TYPES:
            errors.append(f"Invalid relation_type: {relation['relation_type']}")

        val = relation.get("description")
        if val is not None and not isinstance(val, str):
            errors.append("description must be a string if provided")

    return errors


# ---------------------------------------------------------------------------
# Exclusive Relation Types
# ---------------------------------------------------------------------------

_EXCLUSIVE_RELATIONS = {"works_at", "member_of", "lives_in"}


# ---------------------------------------------------------------------------
# Routing Decision
# ---------------------------------------------------------------------------


async def route_entity(
    entity: dict[str, Any],
    chunk_id: int,
) -> dict[str, Any]:
    """Route a validated entity to auto-apply, auto-create, or review queue.

    Returns a routing decision dict with:
        action: 'apply_existing' | 'create_new' | 'review' | 'singleton'
        entity_id: (for apply_existing/singleton)
        candidates: (for review, list of fuzzy matches)
        reason: human-readable explanation
    """
    settings = get_settings()
    name = entity["name"].strip()
    entity_type = entity["entity_type"]
    confidence = entity["confidence"]
    match = entity["match"]

    # --- Singleton enforcement ---
    singleton_id = _is_singleton(name)
    if singleton_id is not None:
        return {
            "action": "singleton",
            "entity_id": singleton_id,
            "reason": f"Singleton entity: {name} → id={singleton_id}",
        }

    pool = get_pool()

    async with pool.acquire() as conn:
        # --- Denylist check (before any matching) ---
        denied = await conn.fetchval(
            "SELECT 1 FROM extraction_denylist WHERE lower(name) = lower($1) AND entity_type = $2",
            name,
            entity_type,
        )
        if denied:
            return {
                "action": "skip",
                "reason": f"Denylisted: {name} ({entity_type})",
            }

        # --- Check for exact name+type match ---
        exact_match = await conn.fetchrow(
            "SELECT id, name, entity_type FROM entities WHERE lower(name) = lower($1) AND entity_type = $2",
            name,
            entity_type,
        )

        if exact_match:
            if confidence >= settings.validator_existing_threshold:
                return {
                    "action": "apply_existing",
                    "entity_id": exact_match["id"],
                    "reason": f"Exact match: {exact_match['name']} (id={exact_match['id']}), conf={confidence}",
                }
            candidate = {"id": exact_match["id"], "name": exact_match["name"], "type": exact_match["entity_type"]}
            return {
                "action": "review",
                "candidates": [candidate],
                "reason": f"Exact match but low confidence: conf={confidence}",
            }

        # --- Check for exact name match across different types ---
        cross_type = await conn.fetch(
            "SELECT id, name, entity_type FROM entities WHERE lower(name) = lower($1)",
            name,
        )
        if cross_type:
            candidates = [{"id": r["id"], "name": r["name"], "type": r["entity_type"]} for r in cross_type]
            return {
                "action": "review",
                "candidates": candidates,
                "reason": f"Name matches existing entity with different type: {[c['type'] for c in candidates]}",
            }

        # --- Alias check (deterministic, before fuzzy) ---
        alias_match = await conn.fetchrow(
            """
            SELECT id, name, entity_type
            FROM entities
            WHERE lower($1) = ANY(SELECT lower(unnest(aliases)))
              AND entity_type = $2
            LIMIT 1
            """,
            name,
            entity_type,
        )

        if alias_match:
            if confidence >= settings.validator_existing_threshold:
                return {
                    "action": "apply_existing",
                    "entity_id": alias_match["id"],
                    "reason": f"Alias match: '{name}' is alias of {alias_match['name']} (id={alias_match['id']})",
                }
            candidate = {"id": alias_match["id"], "name": alias_match["name"], "type": alias_match["entity_type"]}
            return {
                "action": "review",
                "candidates": [candidate],
                "reason": f"Alias match but low confidence: conf={confidence}",
            }

        # --- Fuzzy match within same type ---
        fuzzy_matches = await conn.fetch(
            """
            SELECT id, name, entity_type,
                   similarity(lower(name), lower($1)) AS sim
            FROM entities
            WHERE entity_type = $2
              AND similarity(lower(name), lower($1)) > $3
            ORDER BY sim DESC
            LIMIT 5
            """,
            name,
            entity_type,
            settings.validator_fuzzy_threshold,
        )

        if fuzzy_matches:
            candidates = [
                {"id": r["id"], "name": r["name"], "type": r["entity_type"], "similarity": float(r["sim"])}
                for r in fuzzy_matches
            ]
            return {
                "action": "review",
                "candidates": candidates,
                "reason": f"Fuzzy match(es) found: {[f'{c["name"]} ({c["similarity"]:.2f})' for c in candidates]}",
            }

        # --- New entity: confidence check ---
        if match == "new" and confidence >= settings.validator_new_threshold:
            return {
                "action": "create_new",
                "reason": f"New entity, no matches, conf={confidence} >= {settings.validator_new_threshold}",
            }

        # --- Below threshold → review ---
        return {
            "action": "review",
            "candidates": [],
            "reason": f"Below threshold: match={match}, conf={confidence}",
        }


async def route_relation(
    relation: dict[str, Any],
    entity_id_map: dict[str, int],
) -> dict[str, Any]:
    """Route a validated relation for application.

    Args:
        relation: Extracted relation dict.
        entity_id_map: Mapping of entity name → entity_id for entities already applied in this chunk.

    Returns:
        Routing decision dict with action and resolved entity IDs.
    """
    source_name = relation["source"].strip()
    target_name = relation["target"].strip()
    relation_type = relation["relation_type"]

    # Resolve source and target to entity IDs
    source_id = _resolve_entity_name(source_name, entity_id_map)
    target_id = _resolve_entity_name(target_name, entity_id_map)

    if source_id is None:
        return {"action": "skip", "reason": f"Source entity '{source_name}' not resolved"}
    if target_id is None:
        return {"action": "skip", "reason": f"Target entity '{target_name}' not resolved"}

    if source_id == target_id:
        return {"action": "skip", "reason": f"Self-referencing relation: {source_name} → {target_name}"}

    # Check for exclusive relation supersession
    if relation_type in _EXCLUSIVE_RELATIONS:
        pool = get_pool()
        async with pool.acquire() as conn:
            existing = await conn.fetchrow(
                """
                SELECT id, target_id, valid_from
                FROM relations
                WHERE source_id = $1
                  AND relation_type = $2
                  AND superseded_at IS NULL
                """,
                source_id,
                relation_type,
            )

            if existing and existing["target_id"] != target_id:
                # Only supersede if the incoming relation is newer (or existing has no date)
                incoming_valid_from = _parse_datetime(relation.get("valid_from"))
                existing_valid_from = existing["valid_from"]  # already datetime from DB

                # Both have dates — only supersede if incoming is newer
                if existing_valid_from and incoming_valid_from and incoming_valid_from <= existing_valid_from:
                    return {
                        "action": "skip",
                        "reason": (
                            f"Exclusive {relation_type}: incoming is older than existing "
                            f"(incoming={incoming_valid_from}, existing={existing_valid_from})"
                        ),
                    }

                return {
                    "action": "supersede",
                    "source_id": source_id,
                    "target_id": target_id,
                    "supersede_relation_id": existing["id"],
                    "reason": f"Exclusive {relation_type}: supersedes existing (id={existing['id']})",
                }

    return {
        "action": "create",
        "source_id": source_id,
        "target_id": target_id,
        "reason": "Valid relation",
    }


def _parse_datetime(value: Any) -> datetime | None:
    """Coerce a value to datetime, handling strings and None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _resolve_entity_name(name: str, entity_id_map: dict[str, int]) -> int | None:
    """Resolve an entity name to an ID via the map or singletons."""
    # Check singletons first
    singleton_id = _is_singleton(name)
    if singleton_id is not None:
        return singleton_id

    # Check the entity_id_map (entities applied in this chunk)
    return entity_id_map.get(name.lower())
