"""Query tool registrations."""

from fastmcp import FastMCP


def register(mcp: FastMCP) -> None:
    """Register all query tools on the MCP app."""

    @mcp.tool()
    async def boot() -> dict:
        """Return session-start context snapshot.

        Aggregates recently active entities and pipeline health counts.
        Call this at the start of every session.
        """
        from app.tools.query import boot as _boot

        return await _boot()

    @mcp.tool()
    async def recall(
        query: str,
        depth: int = 1,
        limit: int = 10,
        chunk_type: str | None = None,
    ) -> list:
        """Search Vex's memory using hybrid semantic + keyword search.

        Args:
            query: Natural language search query
            depth: Detail level (1=entities only, 2=+chunk content, 3=+conversation segments)
            limit: Maximum results
            chunk_type: Optional filter (topic, decision, correction, emotional, moment)
        """
        from app.tools.query import recall as _recall

        return await _recall(query, depth=depth, limit=limit, chunk_type=chunk_type)

    @mcp.tool()
    async def search(
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        source: str | None = None,
        limit: int = 10,
    ) -> list:
        """Keyword search across raw message content.

        Args:
            query: Search terms (exact phrase matching)
            date_from: ISO date filter start (e.g. "2026-01-01")
            date_to: ISO date filter end (e.g. "2026-03-25")
            source: Filter by source ("cc" or "claude_ai")
            limit: Maximum results
        """
        from app.tools.query import search as _search

        return await _search(query, date_from=date_from, date_to=date_to, source=source, limit=limit)

    @mcp.tool()
    async def get_conversation(conversation_id: int) -> dict | None:
        """Return the full transcript and metadata for a single conversation.

        Args:
            conversation_id: Primary key of the conversation.
        """
        from app.tools.query import get_conversation as _get_conversation

        return await _get_conversation(conversation_id)

    @mcp.tool()
    async def recent_conversations(days: int = 7, limit: int = 10) -> list:
        """Return a list of recent conversations ordered by start time descending.

        Args:
            days: Include conversations started within this many days ago (Melbourne timezone).
            limit: Maximum number of conversations to return.
        """
        from app.tools.query import recent_conversations as _recent_conversations

        return await _recent_conversations(days=days, limit=limit)

    @mcp.tool()
    async def get_entity(name_or_id: str) -> dict | None:
        """Return an entity with its current relations and linked chunk summaries.

        Tries numeric ID first, then falls back to case-insensitive name matching.

        Args:
            name_or_id: Entity ID (numeric string) or name (partial match supported).
        """
        from app.tools.query import get_entity as _get_entity

        return await _get_entity(name_or_id)

    @mcp.tool()
    async def neighbourhood(
        entity_name: str,
        hops: int = 1,
        relation_types: list[str] | None = None,
    ) -> dict | None:
        """Graph traversal outward from an entity, returning connected entities and relations.

        Walks the knowledge graph outward from a starting entity up to `hops` depth.
        Only follows current (non-superseded) relations. Handles cycles.

        Args:
            entity_name: Name or ID of the starting entity.
            hops: Traversal depth (1-3, default 1).
            relation_types: Optional list of relation types to follow (e.g. ["has_condition", "treated_by"]).
        """
        from app.tools.query import neighbourhood as _neighbourhood

        return await _neighbourhood(entity_name, hops=hops, relation_types=relation_types)
