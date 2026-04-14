"""Vex MCP server — knowledge management and conversational memory backend."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import get_settings
from app.db import close_pool, create_pool, get_pool, run_migrations

logger = structlog.get_logger()


def configure_logging(log_level: str) -> None:
    """Configure structlog with the given log level. Called from main(), not at import time."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if log_level == "DEBUG" else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, log_level)),
    )


@asynccontextmanager
async def lifespan(_app: FastMCP) -> AsyncIterator[None]:
    """Manage server lifecycle — pool creation, migration, shutdown."""
    settings = get_settings()
    logger.info("vex.starting", host=settings.host, port=settings.port)
    await create_pool()
    try:
        await run_migrations()
        from app.pipeline.orchestrator import startup_sweep

        await startup_sweep()
        yield
    finally:
        try:
            from app.pipeline.orchestrator import shutdown as _shutdown

            await _shutdown()
        except Exception:
            logger.exception("vex.shutdown_error")
        await close_pool()
        logger.info("vex.stopped")


mcp = FastMCP(
    "Vex",
    instructions=(
        "Vex — knowledge management and conversational memory backend. "
        "Query tools: boot (session start context), "
        "recall (semantic search), search (keyword search), neighbourhood (graph traversal), "
        "get_conversation (full transcript), recent_conversations (conversation list), "
        "get_entity (entity + relations + chunks). "
        "Write tools: save_conversation, "
        "add_entity, add_relation, correct_entity, correct_relation, merge_entities, "
        "remove_from_denylist. "
        "Review tools: get_review_queue, resolve_review, resolve_review_new, dismiss_review. "
        "Persona tools: get_persona (session boot), update_persona, delete_persona_section. "
        "Admin tools: stats, get_audit_report, reprocess_conversation, reembed_all."
    ),
    lifespan=lifespan,
)


@mcp.custom_route("/health", methods=["GET"])
async def health_check(_request: Request) -> JSONResponse:
    """Health check endpoint for MCPHub and Docker."""
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return JSONResponse({"status": "healthy", "service": "vex-brain"})
    except Exception:
        logger.error("health_check_failed", exc_info=True)
        return JSONResponse({"status": "unhealthy", "service": "vex-brain"}, status_code=503)


# Register tool routers
from app.routers import admin as admin_router  # noqa: E402
from app.routers import query as query_router  # noqa: E402
from app.routers import review as review_router  # noqa: E402
from app.routers import write as write_router  # noqa: E402

query_router.register(mcp)
write_router.register(mcp)
review_router.register(mcp)
admin_router.register(mcp)


# ---------------------------------------------------------------------------
# Persona tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_persona() -> dict:
    """Return the full persona definition for session boot.

    Returns all persona sections. Call at session start to load
    the configured assistant persona.
    """
    from app.tools.persona import get_persona as _get_persona

    return await _get_persona()


@mcp.tool()
async def update_persona(key: str, content: str) -> dict:
    """Create or update a persona section.

    Args:
        key: Section key (e.g. 'identity', 'exec-function', 'interaction').
        content: Full markdown content for this section.
    """
    from app.tools.persona import update_persona as _update_persona

    return await _update_persona(key, content)


@mcp.tool()
async def delete_persona_section(key: str) -> dict:
    """Delete a persona section.

    Args:
        key: Section key to delete.
    """
    from app.tools.persona import delete_persona_section as _delete_persona_section

    return await _delete_persona_section(key)


@mcp.tool()
async def run_audit() -> dict:
    """Run all graph quality checks and store results in audit_log.

    Checks for: duplicate entities, orphan entities, broken relations,
    duplicate relations, stale entities. Returns finding counts.
    """
    from app.auditor.checks import run_audit as _run_audit

    return await _run_audit()


def main() -> None:
    """Run the Vex MCP server."""
    settings = get_settings()
    configure_logging(settings.log_level)
    mcp.run(transport="http", host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
