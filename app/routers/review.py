"""Review tool registrations."""

from fastmcp import FastMCP


def register(mcp: FastMCP) -> None:
    """Register all review tools on the MCP app."""

    @mcp.tool()
    async def get_review_queue(limit: int = 10, status: str = "pending") -> list:
        """Return pending review queue items with chunk context and candidates.

        Args:
            limit: Maximum items to return.
            status: Filter by status ('pending', 'resolved', 'dismissed').
        """
        from app.tools.review import get_review_queue as _get_review_queue

        return await _get_review_queue(limit=limit, status=status)

    @mcp.tool()
    async def resolve_review(review_id: int, entity_id: int) -> dict:
        """Resolve a review item to an existing entity.

        Creates entity_chunks link, adds proposed name as alias,
        stores extraction example for learning, sets status='resolved'.

        Args:
            review_id: ID of the review queue item.
            entity_id: ID of the existing entity to link to.
        """
        from app.tools.review import resolve_review as _resolve_review

        return await _resolve_review(review_id, entity_id)

    @mcp.tool()
    async def resolve_review_new(
        review_id: int,
        name: str,
        entity_type: str,
        summary: str | None = None,
    ) -> dict:
        """Resolve a review item by confirming it as a genuinely new entity.

        Creates the entity, links to chunk, stores extraction example.

        Args:
            review_id: ID of the review queue item.
            name: Entity name.
            entity_type: Entity type.
            summary: Optional entity summary.
        """
        from app.tools.review import resolve_review_new as _resolve_review_new

        return await _resolve_review_new(review_id, name, entity_type, summary=summary)

    @mcp.tool()
    async def dismiss_review(review_id: int) -> dict:
        """Dismiss a review item as noise. No entity created, no links.

        Auto-adds to extraction denylist so the same name+type won't come back.

        Args:
            review_id: ID of the review queue item.
        """
        from app.tools.review import dismiss_review as _dismiss_review

        return await _dismiss_review(review_id)
