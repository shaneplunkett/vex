"""Admin tool registrations."""

from fastmcp import FastMCP


def register(mcp: FastMCP) -> None:
    """Register all admin tools on the MCP app."""

    @mcp.tool()
    async def stats() -> dict:
        """Return system-wide statistics.

        Counts by layer (conversations, messages, chunks, entities, relations),
        entity type distribution, pipeline status breakdown, and queue depth.
        """
        from app.tools.admin import stats as _stats

        return await _stats()

    @mcp.tool()
    async def get_audit_report(latest: bool = True) -> list:
        """Return audit findings from audit_log.

        Args:
            latest: If True, return only the most recent report per audit_type.
        """
        from app.tools.admin import get_audit_report as _get_audit_report

        return await _get_audit_report(latest=latest)

    @mcp.tool()
    async def reprocess_conversation(conversation_id: int) -> dict:
        """Reset a conversation's pipeline and re-queue for processing.

        Args:
            conversation_id: ID of the conversation to reprocess.
        """
        from app.tools.admin import reprocess_conversation as _reprocess

        return await _reprocess(conversation_id)

    @mcp.tool()
    async def reembed_all() -> dict:
        """Nullify all chunk embeddings to trigger re-embedding.

        Use after changing embedding model or dimensions. Resets affected
        conversations to 'chunked' status for the pipeline to pick up.
        """
        from app.tools.admin import reembed_all as _reembed_all

        return await _reembed_all()

    @mcp.tool()
    async def reconsolidate_entity(
        entity_id: int,
        mode: str = "heuristic",
    ) -> dict:
        """Regenerate an entity's summary from all linked chunk content.

        Use when an entity summary looks stale or incomplete. Reads all chunks
        linked to the entity and builds a fresh summary.

        Args:
            entity_id: ID of the entity to reconsolidate.
            mode: 'api' for LLM summarisation (Haiku), 'heuristic' for rule-based.
        """
        from app.pipeline.maintenance import reconsolidate_entity as _reconsolidate

        return await _reconsolidate(entity_id, mode=mode)  # type: ignore[arg-type]

    @mcp.tool()
    async def validate_entity_type(
        entity_id: int,
        mode: str = "heuristic",
    ) -> dict:
        """Check whether an entity's type assignment is correct.

        Uses relation patterns and name matching (heuristic) or LLM classification
        (api) to suggest whether the entity type should be changed.

        Args:
            entity_id: ID of the entity to validate.
            mode: 'api' for LLM classification (Haiku), 'heuristic' for rule-based.
        """
        from app.pipeline.maintenance import validate_entity_type as _validate

        return await _validate(entity_id, mode=mode)  # type: ignore[arg-type]
