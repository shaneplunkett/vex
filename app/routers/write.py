"""Write tool registrations."""

from fastmcp import FastMCP


def register(mcp: FastMCP) -> None:
    """Register all write tools on the MCP app."""

    @mcp.tool()
    async def save_conversation(
        messages: list[dict],
        source: str = "cc",
        name: str | None = None,
        session_id: str | None = None,
    ) -> dict:
        """Store a conversation and trigger the async pipeline.

        Returns immediately with conversation_id. Chunking and embedding run in background.

        Args:
            messages: List of dicts with keys: role, content, timestamp (ISO str), ordinal.
            source: Source identifier ('cc' or 'claude_ai').
            name: Human-readable conversation name.
            session_id: Unique session identifier.
        """
        from app.tools.write import save_conversation as _save_conversation

        return await _save_conversation(messages, source=source, name=name, session_id=session_id)

    @mcp.tool()
    async def add_entity(
        name: str,
        entity_type: str,
        summary: str | None = None,
    ) -> dict:
        """Manually create an entity. Escape hatch for when extraction misses something.

        Args:
            name: Entity name.
            entity_type: Entity type (e.g. Person, Place, Concept).
            summary: Optional description.
        """
        from app.tools.write import add_entity as _add_entity

        return await _add_entity(name, entity_type, summary=summary)

    @mcp.tool()
    async def add_relation(
        source_entity: str,
        target_entity: str,
        relation_type: str,
        description: str | None = None,
        valid_from: str | None = None,
    ) -> dict:
        """Manually create a relation between two entities.

        Entities are looked up by name (case-insensitive). Both must exist.

        Args:
            source_entity: Name of the source entity.
            target_entity: Name of the target entity.
            relation_type: Type of relationship (e.g. "lives_in", "works_at").
            description: Optional description of the relationship.
            valid_from: ISO datetime string for when the relation became valid.
        """
        from app.tools.write import add_relation as _add_relation

        return await _add_relation(
            source_entity, target_entity, relation_type, description=description, valid_from=valid_from
        )

    @mcp.tool()
    async def correct_entity(
        entity_id: int,
        name: str | None = None,
        summary: str | None = None,
        entity_type: str | None = None,
        add_alias: str | None = None,
    ) -> dict:
        """Fix extraction errors on an entity.

        Args:
            entity_id: ID of the entity to correct.
            name: New name (if renaming).
            summary: Updated summary.
            entity_type: Updated type.
            add_alias: Alias to add to the aliases array.
        """
        from app.tools.write import correct_entity as _correct_entity

        return await _correct_entity(
            entity_id, name=name, summary=summary, entity_type=entity_type, add_alias=add_alias
        )

    @mcp.tool()
    async def correct_relation(
        relation_id: int,
        description: str | None = None,
        supersede: bool = False,
    ) -> dict:
        """Fix or supersede a relation.

        Args:
            relation_id: ID of the relation to correct.
            description: Updated description.
            supersede: If True, mark the relation as superseded (soft-delete).
        """
        from app.tools.write import correct_relation as _correct_relation

        return await _correct_relation(relation_id, description=description, supersede=supersede)

    @mcp.tool()
    async def merge_entities(
        source_ids: list[int],
        target_id: int,
    ) -> dict:
        """Merge duplicate entities into a single target.

        Reassigns all relations and chunk links from sources to the target,
        merges aliases, deletes source entities, and logs in audit_log.

        Args:
            source_ids: IDs of the duplicate entities to merge away.
            target_id: ID of the entity to keep.
        """
        from app.tools.write import merge_entities as _merge_entities

        return await _merge_entities(source_ids, target_id)

    @mcp.tool()
    async def remove_from_denylist(name: str, entity_type: str) -> dict:
        """Remove an entity from the extraction denylist.

        Use when a previous dismissal was wrong and the entity should be
        allowed through extraction again.

        Args:
            name: Entity name to un-deny.
            entity_type: Entity type (e.g. Person, Tool, HealthCondition).
        """
        from app.tools.write import remove_from_denylist as _remove_from_denylist

        return await _remove_from_denylist(name, entity_type)
