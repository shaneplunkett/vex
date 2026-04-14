"""Entity maintenance operations — reconsolidate summaries and validate types.

Post-extraction maintenance that runs on existing entities, not as part of
the ingestion pipeline. Each operation supports dual-mode:
  - api: LLM call (Haiku) for higher quality
  - heuristic: rule-based, zero cost, good enough for bulk passes
"""

from __future__ import annotations

from typing import Any, Literal

import structlog

from app.db import get_pool

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Relation-type → entity-type inference rules for heuristic validation
# ---------------------------------------------------------------------------

# If entity appears as SOURCE of these relation types, it's likely this type
_SOURCE_TYPE_HINTS: dict[str, str] = {
    "has_condition": "Person",
    "works_at": "Person",
    "member_of": "Person",
    "lives_in": "Person",
    "knows": "Person",
    "identifies_as": "Person",
    "follows": "Person",
    "prefers": "Person",
    "avoids": "Person",
    "uses": "Person",
    "treats": "Person",
    "built_on": "Project",
    "part_of": "Infrastructure",
    "manages": "HealthCondition",
}

# If entity appears as TARGET of these relation types, it's likely this type
_TARGET_TYPE_HINTS: dict[str, str] = {
    "has_condition": "HealthCondition",
    "treated_by": "Medication",
    "prescribed_by": "Person",
    "works_at": "Organisation",
    "member_of": "Organisation",
    "lives_in": "Place",
    "knows": "Person",
    "identifies_as": "Identity",
    "follows": "Routine",
    "prefers": "Preference",
    "uses": "Tool",
    "built_on": "Tool",
    "treats": "Person",
    "manages": "Routine",
}

# Name-pattern hints (lowercased prefix/suffix matching)
_NAME_PATTERNS: list[tuple[str, str]] = [
    ("dr ", "Person"),
    ("dr.", "Person"),
    (" mg", "Medication"),
    ("adhd", "HealthCondition"),
    ("asd", "HealthCondition"),
    ("pots", "HealthCondition"),
    ("anxiety", "PsychologicalPattern"),
    ("depression", "PsychologicalPattern"),
    ("schema", "PsychologicalPattern"),
    ("nix", "Tool"),
    ("neovim", "Tool"),
    ("vim", "Tool"),
    ("docker", "Tool"),
    ("melbourne", "Place"),
    ("australia", "Place"),
    ("autograb", "Organisation"),
]


# ---------------------------------------------------------------------------
# Reconsolidate
# ---------------------------------------------------------------------------


async def reconsolidate_entity(
    entity_id: int,
    mode: Literal["api", "heuristic"] = "api",
) -> dict[str, Any]:
    """Regenerate an entity's summary from all linked chunk content.

    API mode: Haiku summarisation call with all linked chunks.
    Heuristic mode: Takes most recent chunk excerpts, deduplicates sentences.

    Returns: {entity_id, name, old_summary, new_summary, chunks_used, mode}
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        entity = await conn.fetchrow(
            "SELECT id, name, entity_type, summary FROM entities WHERE id = $1",
            entity_id,
        )
        if not entity:
            return {"error": f"Entity {entity_id} not found"}

        chunks = await conn.fetch(
            """
            SELECT c.raw_content, c.chunk_type, c.significance
            FROM chunks c
            JOIN entity_chunks ec ON ec.chunk_id = c.id
            WHERE ec.entity_id = $1
            ORDER BY c.id DESC
            """,
            entity_id,
        )

    if not chunks:
        return {
            "entity_id": entity_id,
            "name": entity["name"],
            "old_summary": entity["summary"],
            "new_summary": entity["summary"],
            "chunks_used": 0,
            "mode": mode,
            "skipped": "no linked chunks",
        }

    old_summary = entity["summary"] or ""

    if mode == "api":
        new_summary = await _reconsolidate_api(entity, chunks)
    else:
        new_summary = _reconsolidate_heuristic(entity, chunks)

    # Update if changed
    if new_summary != old_summary:
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE entities SET summary = $1, updated_at = now() WHERE id = $2",
                new_summary,
                entity_id,
            )

    return {
        "entity_id": entity_id,
        "name": entity["name"],
        "old_summary": old_summary,
        "new_summary": new_summary,
        "chunks_used": len(chunks),
        "mode": mode,
    }


async def _reconsolidate_api(
    entity: Any,
    chunks: list[Any],
) -> str:
    """Use Haiku to summarise entity from linked chunks."""
    import anthropic

    from app.config import get_settings

    settings = get_settings()
    if not settings.anthropic_api_key:
        return _reconsolidate_heuristic(entity, chunks)

    # Build context from chunks (limit to avoid token overflow)
    chunk_texts = []
    for chunk in chunks[:20]:
        chunk_texts.append(chunk["raw_content"][:500])
    context = "\n---\n".join(chunk_texts)

    prompt = (
        f"Summarise the entity '{entity['name']}' (type: {entity['entity_type']}) "
        f"based on these conversation excerpts. Write 1-3 sentences capturing the "
        f"key facts and current state. Be specific and factual.\n\n{context}"
    )

    try:
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
        response = await client.messages.create(
            model=settings.maintenance_model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        if not response.content:
            logger.warning("maintenance.reconsolidate_empty_response", entity=entity["name"])
            return _reconsolidate_heuristic(entity, chunks)
        block = response.content[0]
        return str(getattr(block, "text", "")).strip()
    except anthropic.APIError as e:
        logger.warning("maintenance.reconsolidate_api_error", entity=entity["name"], error=str(e))
        return _reconsolidate_heuristic(entity, chunks)


def _reconsolidate_heuristic(
    entity: Any,
    chunks: list[Any],
) -> str:
    """Build entity summary from chunk content without LLM.

    Takes sentences mentioning the entity name from the most recent chunks,
    deduplicates, and concatenates.
    """
    name_lower = entity["name"].lower()
    sentences: list[str] = []
    seen: set[str] = set()

    for chunk in chunks[:10]:
        content = chunk["raw_content"]
        # Extract sentences that mention the entity
        for sentence in content.replace("\n", " ").split(". "):
            sentence = sentence.strip().rstrip(".")
            if not sentence:
                continue
            normalised = sentence.lower()
            if name_lower in normalised and normalised not in seen:
                seen.add(normalised)
                sentences.append(sentence)
                if len(sentences) >= 5:
                    break
        if len(sentences) >= 5:
            break

    if not sentences:
        # Fallback: use existing summary or first chunk excerpt
        return str(entity["summary"] or chunks[0]["raw_content"][:200])

    return ". ".join(sentences) + "."


async def reconsolidate_all(
    mode: Literal["api", "heuristic"] = "api",
    entity_type: str | None = None,
    min_chunks: int = 2,
) -> dict[str, Any]:
    """Reconsolidate all entities matching filters.

    Args:
        mode: 'api' for LLM summarisation, 'heuristic' for rule-based.
        entity_type: Filter by entity type (optional).
        min_chunks: Only reconsolidate entities with >= this many linked chunks.

    Returns: {processed, updated, skipped, errors}
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        if entity_type:
            rows = await conn.fetch(
                """
                SELECT e.id FROM entities e
                JOIN entity_chunks ec ON ec.entity_id = e.id
                WHERE e.entity_type = $1
                GROUP BY e.id
                HAVING count(ec.chunk_id) >= $2
                ORDER BY e.id
                """,
                entity_type,
                min_chunks,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT e.id FROM entities e
                JOIN entity_chunks ec ON ec.entity_id = e.id
                GROUP BY e.id
                HAVING count(ec.chunk_id) >= $1
                ORDER BY e.id
                """,
                min_chunks,
            )

    processed = 0
    updated = 0
    skipped = 0
    errors = 0

    for row in rows:
        try:
            result = await reconsolidate_entity(row["id"], mode=mode)
            processed += 1
            if result.get("skipped"):
                skipped += 1
            elif result.get("old_summary") != result.get("new_summary"):
                updated += 1
        except Exception:
            logger.exception("maintenance.reconsolidate_error", entity_id=row["id"])
            errors += 1

    return {"processed": processed, "updated": updated, "skipped": skipped, "errors": errors, "mode": mode}


# ---------------------------------------------------------------------------
# Type Validation
# ---------------------------------------------------------------------------


async def validate_entity_type(
    entity_id: int,
    mode: Literal["api", "heuristic"] = "heuristic",
) -> dict[str, Any]:
    """Validate that an entity's type assignment is correct.

    API mode: Haiku classification with entity name + summary + sample chunks.
    Heuristic mode: Relation-type inference + name pattern matching.

    Returns: {entity_id, name, current_type, suggested_type, confidence, reasoning}
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        entity = await conn.fetchrow(
            "SELECT id, name, entity_type, summary FROM entities WHERE id = $1",
            entity_id,
        )
        if not entity:
            return {"error": f"Entity {entity_id} not found"}

        # Fetch relations involving this entity
        relations = await conn.fetch(
            """
            SELECT r.relation_type,
                   CASE WHEN r.source_id = $1 THEN 'source' ELSE 'target' END AS role
            FROM relations r
            WHERE (r.source_id = $1 OR r.target_id = $1)
              AND r.superseded_at IS NULL
            """,
            entity_id,
        )

        # Fetch sample chunk content for API mode
        chunks = await conn.fetch(
            """
            SELECT c.raw_content
            FROM chunks c
            JOIN entity_chunks ec ON ec.chunk_id = c.id
            WHERE ec.entity_id = $1
            ORDER BY c.id DESC
            LIMIT 3
            """,
            entity_id,
        )

    if mode == "api":
        return await _validate_type_api(entity, relations, chunks)
    return _validate_type_heuristic(entity, relations)


def _validate_type_heuristic(
    entity: Any,
    relations: list[Any],
) -> dict[str, Any]:
    """Validate entity type using relation inference and name patterns."""
    current_type = entity["entity_type"]
    name = entity["name"]
    name_lower = name.lower()

    # Collect type votes from relations
    type_votes: dict[str, int] = {}
    reasons: list[str] = []

    for rel in relations:
        hint_map = _SOURCE_TYPE_HINTS if rel["role"] == "source" else _TARGET_TYPE_HINTS
        suggested = hint_map.get(rel["relation_type"])
        if suggested:
            type_votes[suggested] = type_votes.get(suggested, 0) + 1

    # Name pattern matching
    for pattern, suggested_type in _NAME_PATTERNS:
        if pattern in name_lower:
            type_votes[suggested_type] = type_votes.get(suggested_type, 0) + 2
            reasons.append(f"name matches pattern '{pattern}' → {suggested_type}")

    if not type_votes:
        return {
            "entity_id": entity["id"],
            "name": name,
            "current_type": current_type,
            "suggested_type": None,
            "confidence": 1.0,
            "reasoning": "No relation or name signals — current type stands",
        }

    # Find the most-voted type
    top_type = max(type_votes, key=lambda t: type_votes[t])
    top_votes = type_votes[top_type]
    total_votes = sum(type_votes.values())

    # Build reasoning
    if relations:
        rel_summary = ", ".join(f"{r['role']} of {r['relation_type']}" for r in relations[:5])
        reasons.append(f"relations: {rel_summary}")

    reasoning = "; ".join(reasons) if reasons else f"type votes: {type_votes}"

    if top_type == current_type:
        return {
            "entity_id": entity["id"],
            "name": name,
            "current_type": current_type,
            "suggested_type": None,
            "confidence": min(1.0, top_votes / max(total_votes, 1)),
            "reasoning": f"Current type confirmed — {reasoning}",
        }

    return {
        "entity_id": entity["id"],
        "name": name,
        "current_type": current_type,
        "suggested_type": top_type,
        "confidence": min(1.0, top_votes / max(total_votes, 1)),
        "reasoning": f"Suggested {top_type} (currently {current_type}) — {reasoning}",
    }


async def _validate_type_api(
    entity: Any,
    relations: list[Any],
    chunks: list[Any],
) -> dict[str, Any]:
    """Use Haiku to classify entity type."""
    import anthropic

    from app.config import get_settings
    from app.pipeline.extractor import ENTITY_TYPES

    settings = get_settings()
    if not settings.anthropic_api_key:
        return _validate_type_heuristic(entity, relations)

    rel_context = ", ".join(f"{r['role']} of {r['relation_type']}" for r in relations[:10])
    chunk_context = "\n---\n".join(c["raw_content"][:300] for c in chunks[:3])

    prompt = (
        f"What entity type best fits '{entity['name']}'?\n"
        f"Current type: {entity['entity_type']}\n"
        f"Summary: {entity['summary'] or 'none'}\n"
        f"Relations: {rel_context or 'none'}\n"
        f"Context:\n{chunk_context or 'none'}\n\n"
        f"Valid types: {', '.join(ENTITY_TYPES)}\n\n"
        f"Reply with ONLY the type name, nothing else."
    )

    try:
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
        response = await client.messages.create(
            model=settings.maintenance_model,
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}],
        )
        if not response.content:
            return _validate_type_heuristic(entity, relations)
        block = response.content[0]
        suggested = str(getattr(block, "text", "")).strip()
    except anthropic.APIError:
        return _validate_type_heuristic(entity, relations)

    # Validate the response is a known type
    if suggested not in ENTITY_TYPES:
        return {
            "entity_id": entity["id"],
            "name": entity["name"],
            "current_type": entity["entity_type"],
            "suggested_type": None,
            "confidence": 0.0,
            "reasoning": f"API returned invalid type: {suggested}",
        }

    matches = suggested == entity["entity_type"]
    return {
        "entity_id": entity["id"],
        "name": entity["name"],
        "current_type": entity["entity_type"],
        "suggested_type": None if matches else suggested,
        "confidence": 0.9 if matches else 0.8,
        "reasoning": f"API classification: {suggested}" + (" (matches current)" if matches else ""),
    }


async def validate_all_types(
    mode: Literal["api", "heuristic"] = "heuristic",
) -> dict[str, Any]:
    """Run type validation across all entities.

    Returns: {processed, mismatches: [{entity_id, name, current_type, suggested_type, reasoning}], confirmed}
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id FROM entities ORDER BY id",
        )

    processed = 0
    confirmed = 0
    errors = 0
    mismatches: list[dict[str, Any]] = []

    for row in rows:
        try:
            result = await validate_entity_type(row["id"], mode=mode)
            if result.get("error"):
                continue
            processed += 1
            if result["suggested_type"]:
                mismatches.append(
                    {
                        "entity_id": result["entity_id"],
                        "name": result["name"],
                        "current_type": result["current_type"],
                        "suggested_type": result["suggested_type"],
                        "confidence": result["confidence"],
                        "reasoning": result["reasoning"],
                    }
                )
            else:
                confirmed += 1
        except Exception:
            logger.exception("maintenance.validate_type_error", entity_id=row["id"])
            errors += 1

    return {
        "processed": processed,
        "confirmed": confirmed,
        "mismatches": mismatches,
        "errors": errors,
        "mode": mode,
    }
