"""Vex CLI — import, admin, and maintenance commands."""

import asyncio
import sys
from pathlib import Path

import click
import structlog

from app.config import get_settings
from app.main import configure_logging

logger = structlog.get_logger()


@click.group()
def main() -> None:
    """Vex — knowledge management CLI."""
    # Force line-buffered stdout so output appears in docker exec -T
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if not sys.stdout.isatty() and callable(reconfigure):
        reconfigure(line_buffering=True)


@main.group(name="import")
def import_cmd() -> None:
    """Import conversations from various sources."""


@import_cmd.command("cc")
@click.option(
    "--source",
    required=True,
    type=click.Path(exists=True),
    help="Path to .claude/ directory or projects/ subdirectory",
)
@click.option("--dry-run", is_flag=True, help="Parse and report without writing to database")
def import_cc(source: str, dry_run: bool) -> None:
    """Import Claude Code JSONL sessions."""
    settings = get_settings()
    configure_logging(settings.log_level)
    asyncio.run(_import_cc(Path(source), dry_run))


async def _import_cc(source: Path, dry_run: bool) -> None:
    """Async implementation of CC import."""
    from app.importers.cc import parse_all

    # If pointed at .claude/, look for projects/ subdirectory
    projects_dir = source
    if (source / "projects").is_dir():
        projects_dir = source / "projects"

    conversations = parse_all(projects_dir)

    if not conversations:
        click.echo("No conversations found.")
        return

    click.echo(f"Parsed {len(conversations)} conversations")

    if dry_run:
        for conv in conversations:
            click.echo(f"  {conv.session_id[:8]}  {conv.message_count:3d} msgs  {conv.name}")
        click.echo(f"\nDry run — {len(conversations)} conversations would be imported")
        return

    # Store to database
    from app.db import close_pool, create_pool, run_migrations
    from app.importers.store import store_conversations

    await create_pool()
    try:
        await run_migrations()
        imported, skipped = await store_conversations(conversations, source="cc")
        click.echo(f"Imported {imported}, skipped {skipped} (already existed)")
    finally:
        await close_pool()


@import_cmd.command("archive")
@click.argument("path", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Parse and report without writing to database")
def import_archive(path: str, dry_run: bool) -> None:
    """Import claude.ai conversations-clean.json archive."""
    settings = get_settings()
    configure_logging(settings.log_level)
    asyncio.run(_import_archive(Path(path), dry_run))


async def _import_archive(path: Path, dry_run: bool) -> None:
    """Async implementation of archive import."""
    from app.importers.archive import parse_archive

    conversations = parse_archive(path)

    if not conversations:
        click.echo("No conversations found.")
        return

    click.echo(f"Parsed {len(conversations)} conversations")

    if dry_run:
        for conv in conversations:
            click.echo(f"  {conv.session_id[:8]}  {conv.message_count:3d} msgs  {conv.name}")
        click.echo(f"\nDry run — {len(conversations)} conversations would be imported")
        return

    from app.db import close_pool, create_pool, run_migrations
    from app.importers.store import store_conversations

    await create_pool()
    try:
        await run_migrations()
        imported, skipped = await store_conversations(conversations, source="claude_ai")
        click.echo(f"Imported {imported}, skipped {skipped} (already existed)")
    finally:
        await close_pool()


@main.command()
@click.argument("directory", type=click.Path(exists=True))
def seed_persona(directory: str) -> None:
    """Seed the persona table from a directory of markdown files.

    Each .md file becomes a persona section with the filename (minus extension) as the key.
    """
    settings = get_settings()
    configure_logging(settings.log_level)

    # Read files synchronously before entering async context (ruff ASYNC240)
    dir_path = Path(directory)
    md_files = sorted(dir_path.glob("*.md"))
    file_contents = [(f.stem, f.read_text().strip()) for f in md_files]

    if not file_contents:
        click.echo(f"No .md files found in {directory}")
        return

    asyncio.run(_seed_persona(file_contents))


async def _seed_persona(file_contents: list[tuple[str, str]]) -> None:
    """Async implementation of persona seeding."""
    from app.db import close_pool, create_pool, run_migrations

    await create_pool()
    try:
        await run_migrations()
        from app.tools.persona import update_persona

        seeded = 0
        for key, content in file_contents:
            if not content:
                click.echo(f"  Skipping {key} (empty)")
                continue
            result = await update_persona(key, content)
            click.echo(f"  {result['action']}: {key}")
            seeded += 1

        click.echo(f"\nSeeded {seeded} persona sections")
    finally:
        await close_pool()


@main.command()
def stats() -> None:
    """Show system statistics."""
    settings = get_settings()
    configure_logging(settings.log_level)
    asyncio.run(_stats())


async def _stats() -> None:
    """Async implementation of stats."""
    from app.db import close_pool, create_pool, run_migrations

    await create_pool()
    try:
        await run_migrations()
        from app.tools.admin import stats as get_stats

        result = await get_stats()

        click.echo("\n=== Vex Stats ===\n")
        click.echo("Layers:")
        for key, val in result["layers"].items():
            click.echo(f"  {key}: {val}")
        click.echo(f"\nPipeline status: {result['pipeline_status']}")
        click.echo(f"Entity types: {result['entity_types']}")
        click.echo(f"Review queue: {result['review_queue']}")
        click.echo(f"Denylist entries: {result['denylist_count']}")
        click.echo(f"Links: {result['links']}")
    finally:
        await close_pool()


@main.command()
def audit() -> None:
    """Run all graph quality checks."""
    settings = get_settings()
    configure_logging(settings.log_level)
    asyncio.run(_audit())


async def _audit() -> None:
    """Async implementation of audit."""
    from app.db import close_pool, create_pool, run_migrations

    await create_pool()
    try:
        await run_migrations()
        from app.auditor.checks import run_audit

        result = await run_audit()

        click.echo("\n=== Audit Complete ===")
        click.echo(f"Checks run: {result['checks_run']}")
        click.echo(f"Total findings: {result['total_findings']}")
        for check, count in result["finding_counts"].items():
            click.echo(f"  {check}: {count}")
    finally:
        await close_pool()


@main.group()
def extraction() -> None:
    """Agent extraction workflow — export, import, and status."""


@extraction.command("export")
@click.option(
    "--output-dir",
    type=click.Path(),
    default="agent-extraction",
    help="Directory to write chunk files and prompt to",
)
@click.option(
    "--conversation-id",
    type=int,
    multiple=True,
    help="Specific conversation IDs (default: all embedded)",
)
def extraction_export(output_dir: str, conversation_id: tuple[int, ...]) -> None:
    """Export embedded conversations as JSONL for agent extraction."""
    settings = get_settings()
    configure_logging(settings.log_level)
    ids = list(conversation_id) if conversation_id else None
    asyncio.run(_extraction_export(Path(output_dir), ids))


async def _extraction_export(output_dir: Path, conversation_ids: list[int] | None) -> None:
    from app.db import close_pool, create_pool, run_migrations
    from app.pipeline.agent_io import export_chunks

    await create_pool()
    try:
        await run_migrations()
        result = await export_chunks(output_dir, conversation_ids)
        click.echo(f"\nExported {result['exported']} conversations ({result['total_chunks']} chunks)")
        if result["skipped"]:
            click.echo(f"Skipped {result['skipped']} (already extracted or no chunks)")
        click.echo(f"Output: {result['output_dir']}")
        click.echo(f"\nPrompt copied to {output_dir}/prompt.md")
        click.echo("Run CC subagents with the prompt + conv_*.jsonl files, write result_*.json, then:")
        click.echo(f"  vex-brain extraction import {output_dir}")
    finally:
        await close_pool()


@extraction.command("import")
@click.argument("results_dir", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Validate results without applying")
def extraction_import(results_dir: str, dry_run: bool) -> None:
    """Import agent extraction results through validator/linker."""
    settings = get_settings()
    configure_logging(settings.log_level)
    asyncio.run(_extraction_import(Path(results_dir), dry_run))


async def _extraction_import(results_dir: Path, dry_run: bool) -> None:
    from app.db import close_pool, create_pool, run_migrations
    from app.pipeline.agent_io import import_results

    await create_pool()
    try:
        await run_migrations()
        result = await import_results(results_dir, dry_run)

        if "error" in result:
            click.echo(f"Error: {result['error']}")
            return

        prefix = "[DRY RUN] " if dry_run else ""
        click.echo(f"\n{prefix}Import complete:")
        click.echo(f"  Conversations: {result['conversations']}")
        click.echo(f"  Chunks processed: {result['chunks_processed']}")
        if result.get("chunks_failed"):
            click.echo(f"  Chunks failed: {result['chunks_failed']}")
        if not dry_run:
            click.echo(f"  Entities applied: {result['entities_applied']}")
            click.echo(f"  Entities created: {result['entities_created']}")
            click.echo(f"  Entities reviewed: {result['entities_reviewed']}")
            click.echo(f"  Relations created: {result['relations_created']}")
        if result.get("errors"):
            click.echo(f"\n  File errors ({len(result['errors'])}):")
            for err in result["errors"]:
                click.echo(f"    {err['file']}: {err['error']}")
    finally:
        await close_pool()


@extraction.command("status")
def extraction_status() -> None:
    """Show pipeline status breakdown across all conversations."""
    settings = get_settings()
    configure_logging(settings.log_level)
    asyncio.run(_extraction_status())


async def _extraction_status() -> None:
    from app.db import close_pool, create_pool, run_migrations

    await create_pool()
    try:
        await run_migrations()
        from app.db import get_pool

        pool = get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT pipeline_status, count(*) as count
                FROM conversations
                GROUP BY pipeline_status
                ORDER BY pipeline_status
                """
            )
        total = sum(r["count"] for r in rows)
        click.echo(f"\nPipeline status ({total} conversations):")
        for row in rows:
            click.echo(f"  {row['pipeline_status']:12s} {row['count']:5d}")

        from app.config import get_settings as _get_settings

        click.echo(f"\nCurrent mode: {_get_settings().pipeline_mode}")
    finally:
        await close_pool()


@main.command()
@click.argument("conversation_id", type=int)
def reprocess(conversation_id: int) -> None:
    """Re-run pipeline on an existing conversation."""
    settings = get_settings()
    configure_logging(settings.log_level)
    asyncio.run(_reprocess(conversation_id))


async def _reprocess(conversation_id: int) -> None:
    """Async implementation of reprocess."""
    from app.db import close_pool, create_pool, run_migrations

    await create_pool()
    try:
        await run_migrations()
        from app.tools.admin import reprocess_conversation

        result = await reprocess_conversation(conversation_id)
        if "error" in result:
            click.echo(f"Error: {result['error']}")
        else:
            click.echo(f"Requeued conversation {conversation_id}: {result['old_status']} → pending")
    finally:
        await close_pool()


@main.group()
def maintain() -> None:
    """Graph maintenance operations."""


@maintain.command("reconsolidate")
@click.option("--entity-id", type=int, help="Reconsolidate a specific entity")
@click.option("--entity-type", type=str, help="Filter by entity type")
@click.option(
    "--mode",
    type=click.Choice(["api", "heuristic"]),
    default="heuristic",
    help="Summarisation mode (default heuristic; api uses Haiku)",
)
@click.option("--min-chunks", type=int, default=2, help="Min linked chunks to qualify")
def reconsolidate(entity_id: int | None, entity_type: str | None, mode: str, min_chunks: int) -> None:
    """Reconsolidate entity summaries from linked chunk content."""
    settings = get_settings()
    configure_logging(settings.log_level)
    asyncio.run(_reconsolidate(entity_id, entity_type, mode, min_chunks))


async def _reconsolidate(entity_id: int | None, entity_type: str | None, mode: str, min_chunks: int) -> None:
    from app.db import close_pool, create_pool, run_migrations
    from app.pipeline.maintenance import reconsolidate_all, reconsolidate_entity

    await create_pool()
    try:
        await run_migrations()
        if entity_id:
            result = await reconsolidate_entity(entity_id, mode=mode)  # type: ignore[arg-type]
            if "error" in result:
                click.echo(f"Error: {result['error']}")
            else:
                click.echo(f"Entity: {result['name']} ({result['mode']})")
                click.echo(f"  Chunks used: {result['chunks_used']}")
                click.echo(f"  Old: {result['old_summary']}")
                click.echo(f"  New: {result['new_summary']}")
        else:
            result = await reconsolidate_all(
                mode=mode,  # type: ignore[arg-type]
                entity_type=entity_type,
                min_chunks=min_chunks,
            )
            click.echo(f"\nReconsolidate complete ({result['mode']}):")
            click.echo(f"  Processed: {result['processed']}")
            click.echo(f"  Updated: {result['updated']}")
            click.echo(f"  Skipped: {result['skipped']}")
            if result["errors"]:
                click.echo(f"  Errors: {result['errors']}")
    finally:
        await close_pool()


@maintain.command("validate-types")
@click.option("--entity-id", type=int, help="Validate a specific entity")
@click.option(
    "--mode",
    type=click.Choice(["api", "heuristic"]),
    default="heuristic",
    help="Validation mode",
)
def validate_types(entity_id: int | None, mode: str) -> None:
    """Validate entity type assignments."""
    settings = get_settings()
    configure_logging(settings.log_level)
    asyncio.run(_validate_types(entity_id, mode))


async def _validate_types(entity_id: int | None, mode: str) -> None:
    from app.db import close_pool, create_pool, run_migrations
    from app.pipeline.maintenance import validate_all_types, validate_entity_type

    await create_pool()
    try:
        await run_migrations()
        if entity_id:
            result = await validate_entity_type(entity_id, mode=mode)  # type: ignore[arg-type]
            if "error" in result:
                click.echo(f"Error: {result['error']}")
            else:
                click.echo(f"Entity: {result['name']}")
                click.echo(f"  Current type: {result['current_type']}")
                if result["suggested_type"]:
                    click.echo(f"  Suggested: {result['suggested_type']}")
                else:
                    click.echo("  Type confirmed")
                click.echo(f"  Confidence: {result['confidence']:.2f}")
                click.echo(f"  Reasoning: {result['reasoning']}")
        else:
            result = await validate_all_types(mode=mode)  # type: ignore[arg-type]
            click.echo(f"\nType validation complete ({result['mode']}):")
            click.echo(f"  Processed: {result['processed']}")
            click.echo(f"  Confirmed: {result['confirmed']}")
            click.echo(f"  Mismatches: {len(result['mismatches'])}")
            for m in result["mismatches"]:
                click.echo(f"    {m['name']}: {m['current_type']} → {m['suggested_type']} ({m['reasoning']})")
    finally:
        await close_pool()


if __name__ == "__main__":
    main()
