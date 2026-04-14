"""Persona tools — serve and update the assistant persona definition.

get_persona(): returns the full persona for session boot.
update_persona(): update a single section without redeployment.
"""

from __future__ import annotations

from typing import Any

import structlog

from app.db import get_pool

logger = structlog.get_logger()


async def get_persona() -> dict[str, Any]:
    """Return the full persona definition.

    Returns all persona sections concatenated, keyed by section name.
    Called at session start to load the configured assistant persona.

    Returns:
        Dict with 'sections' (list of {key, content}) and 'full_text' (concatenated).
    """
    pool = get_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT key, content, updated_at FROM persona ORDER BY key",
        )

    if not rows:
        return {"sections": [], "full_text": "", "error": "No persona sections found — run seed first"}

    sections = [{"key": row["key"], "content": row["content"], "updated_at": row["updated_at"]} for row in rows]

    # Concatenate with section headers for full_text
    full_text = "\n\n".join(row["content"] for row in rows)

    return {"sections": sections, "full_text": full_text}


async def update_persona(key: str, content: str) -> dict[str, Any]:
    """Create or update a persona section.

    Args:
        key: Section key (e.g. 'identity', 'exec-function', 'interaction').
        content: Full markdown content for this section.

    Returns:
        Dict with key and action taken.
    """
    clean_key = key.strip()
    clean_content = content.strip()
    if not clean_key:
        return {"error": "Key must not be empty"}
    if not clean_content:
        return {"error": "Content must not be empty"}

    pool = get_pool()

    async with pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            INSERT INTO persona (key, content, updated_at)
            VALUES ($1, $2, now())
            ON CONFLICT (key) DO UPDATE SET
                content = EXCLUDED.content,
                updated_at = now()
            RETURNING (xmax = 0) AS inserted
            """,
            clean_key,
            clean_content,
        )

    action = "created" if result["inserted"] else "updated"
    logger.info("persona.updated", key=clean_key, action=action)
    return {"key": clean_key, "action": action}


async def delete_persona_section(key: str) -> dict[str, Any]:
    """Delete a persona section.

    Args:
        key: Section key to delete.

    Returns:
        Dict with key and action taken.
    """
    pool = get_pool()

    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM persona WHERE key = $1",
            key.strip(),
        )

    deleted = result == "DELETE 1"
    if deleted:
        logger.info("persona.deleted", key=key)
    return {"key": key, "action": "deleted" if deleted else "not_found"}
