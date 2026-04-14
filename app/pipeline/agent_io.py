"""Agent I/O — export chunks for external agent extraction, import results through linker.

Formalises the proven pattern from /tmp/vex-extraction/ into the codebase.
The export/import cycle is the agent-mode counterpart to API extraction:

  1. export_chunks() writes JSONL chunk files + prompt for CC subagents
  2. Agents process externally, produce result_*.json files
  3. import_results() feeds those results through the existing validator/linker
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from app.db import get_pool
from app.pipeline.prompts import get_agent_prompt

logger = structlog.get_logger()


async def get_exportable_conversations(
    status_filter: str = "embedded",
) -> list[dict[str, Any]]:
    """Find conversations ready for agent extraction.

    Returns list of dicts with id, name, source, message_count, pipeline_status.
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, name, source, message_count, pipeline_status
            FROM conversations
            WHERE pipeline_status = $1
            ORDER BY id
            """,
            status_filter,
        )
    return [dict(r) for r in rows]


def _prepare_output_dir(output_dir: Path) -> None:
    """Create output directory and write agent prompt (sync — avoids ASYNC240)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt.md").write_text(get_agent_prompt(), encoding="utf-8")


def _write_chunk_file(path: Path, chunks: list[dict[str, Any]]) -> None:
    """Write chunk data as JSONL (sync — avoids ASYNC240)."""
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def _list_result_files(results_dir: Path) -> list[Path]:
    """List result_*.json files in directory (sync — avoids ASYNC240)."""
    return sorted(results_dir.glob("result_*.json"))


async def export_chunks(
    output_dir: Path,
    conversation_ids: list[int] | None = None,
    status_filter: str = "embedded",
) -> dict[str, Any]:
    """Export chunk data as JSONL files for agent extraction.

    Creates one file per conversation: conv_{id}.jsonl
    Each line is a JSON object with: id, raw_content, chunk_type, significance.
    Also copies the agent extraction prompt to output_dir/prompt.md.

    Args:
        output_dir: Directory to write files to (created if needed).
        conversation_ids: Specific IDs to export. If None, exports all matching status_filter.
        status_filter: Pipeline status to filter on (default: "embedded").

    Returns:
        Stats dict: {"exported": int, "skipped": int, "total_chunks": int, "output_dir": str}
    """
    _prepare_output_dir(output_dir)

    pool = get_pool()
    exported = 0
    skipped = 0
    total_chunks = 0

    # Collect all data from DB first, then write files after releasing connection
    export_payloads: list[tuple[int, list[dict[str, Any]]]] = []

    async with pool.acquire() as conn:
        if conversation_ids:
            conversations = await conn.fetch(
                """
                SELECT id, pipeline_status FROM conversations
                WHERE id = ANY($1::int[])
                ORDER BY id
                """,
                conversation_ids,
            )
        else:
            conversations = await conn.fetch(
                """
                SELECT id, pipeline_status FROM conversations
                WHERE pipeline_status = $1
                ORDER BY id
                """,
                status_filter,
            )

        for conv in conversations:
            conv_id = conv["id"]

            if conv["pipeline_status"] in ("extracted", "complete"):
                logger.debug("agent_io.skip_already_extracted", conversation_id=conv_id)
                skipped += 1
                continue

            chunks = await conn.fetch(
                """
                SELECT id, raw_content, chunk_type, significance
                FROM chunks
                WHERE conversation_id = $1
                ORDER BY start_ordinal
                """,
                conv_id,
            )

            if not chunks:
                logger.warning("agent_io.no_chunks", conversation_id=conv_id)
                skipped += 1
                continue

            chunk_dicts = [
                {
                    "id": chunk["id"],
                    "raw_content": chunk["raw_content"],
                    "chunk_type": chunk["chunk_type"],
                    "significance": chunk["significance"],
                }
                for chunk in chunks
            ]
            export_payloads.append((conv_id, chunk_dicts))

    # Write files outside the DB connection
    for conv_id, chunk_dicts in export_payloads:
        _write_chunk_file(output_dir / f"conv_{conv_id}.jsonl", chunk_dicts)
        exported += 1
        total_chunks += len(chunk_dicts)
        logger.info("agent_io.exported", conversation_id=conv_id, chunk_count=len(chunk_dicts))

    return {
        "exported": exported,
        "skipped": skipped,
        "total_chunks": total_chunks,
        "output_dir": str(output_dir),
    }


async def import_results(
    results_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Import agent extraction results through the validator/linker.

    Reads result_*.json files from results_dir. Each file must contain:
    {"conversation_id": int, "chunks": [{"chunk_id": int, "entities": [...], "relations": [...], "flags": [...]}]}

    Validates chunk ownership (chunk must belong to conversation_id).
    Calls apply_extraction() per chunk — same path as API extraction.
    Updates conversation pipeline_status to 'extracted' on success.

    Args:
        results_dir: Directory containing result_*.json files.
        dry_run: If True, validate files and report stats without applying.

    Returns:
        Aggregate stats dict with per-conversation and total counts.
    """
    from app.pipeline.linker import apply_extraction

    files = _list_result_files(results_dir)
    if not files:
        return {"error": "No result_*.json files found", "results_dir": str(results_dir)}

    pool = get_pool()
    totals: dict[str, int] = {
        "conversations": 0,
        "chunks_processed": 0,
        "chunks_failed": 0,
        "entities_applied": 0,
        "entities_created": 0,
        "entities_reviewed": 0,
        "entities_rejected": 0,
        "relations_created": 0,
        "relations_superseded": 0,
        "relations_skipped": 0,
    }
    errors: list[dict[str, Any]] = []

    for path in files:
        filename = path.name

        # Read file with per-file error isolation
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as e:
            errors.append({"file": filename, "error": str(e)})
            continue

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            errors.append({"file": filename, "error": str(e)})
            continue

        conv_id = data.get("conversation_id")
        chunks = data.get("chunks", [])

        if not conv_id or not chunks:
            errors.append({"file": filename, "error": "Missing conversation_id or chunks"})
            continue

        if dry_run:
            totals["conversations"] += 1
            totals["chunks_processed"] += len(chunks)
            logger.info(
                "agent_io.dry_run",
                file=filename,
                conversation_id=conv_id,
                chunk_count=len(chunks),
                entity_count=sum(len(c.get("entities", [])) for c in chunks),
                relation_count=sum(len(c.get("relations", [])) for c in chunks),
            )
            continue

        # Validate chunk ownership — all chunk_ids must belong to this conversation
        async with pool.acquire() as conn:
            valid_chunk_ids = {
                row["id"]
                for row in await conn.fetch(
                    "SELECT id FROM chunks WHERE conversation_id = $1",
                    conv_id,
                )
            }

        # Apply each chunk's extraction through the linker
        conv_chunks_ok = 0
        conv_chunks_failed = 0

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            if not chunk_id or chunk_id not in valid_chunk_ids:
                if chunk_id and chunk_id not in valid_chunk_ids:
                    logger.warning(
                        "agent_io.chunk_ownership_mismatch",
                        chunk_id=chunk_id,
                        conversation_id=conv_id,
                    )
                conv_chunks_failed += 1
                continue

            extraction = {
                "entities": chunk.get("entities", []),
                "relations": chunk.get("relations", []),
                "flags": chunk.get("flags", []),
            }

            if not extraction["entities"] and not extraction["relations"]:
                conv_chunks_ok += 1
                continue

            try:
                stats = await apply_extraction(chunk_id, extraction)
                conv_chunks_ok += 1
                for key in stats:
                    if key in totals:
                        totals[key] += stats[key]
            except Exception as e:
                logger.error(
                    "agent_io.chunk_import_error",
                    chunk_id=chunk_id,
                    conversation_id=conv_id,
                    error=str(e),
                )
                conv_chunks_failed += 1

        totals["conversations"] += 1
        totals["chunks_processed"] += conv_chunks_ok
        totals["chunks_failed"] += conv_chunks_failed

        # Update pipeline status if all chunks succeeded
        if conv_chunks_failed == 0:
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE conversations SET pipeline_status = 'extracted' WHERE id = $1",
                    conv_id,
                )
        else:
            logger.warning(
                "agent_io.partial_import",
                conversation_id=conv_id,
                ok=conv_chunks_ok,
                failed=conv_chunks_failed,
            )

        logger.info(
            "agent_io.imported",
            conversation_id=conv_id,
            chunks_ok=conv_chunks_ok,
            chunks_failed=conv_chunks_failed,
        )

    result: dict[str, Any] = {**totals, "dry_run": dry_run}
    if errors:
        result["errors"] = errors
    return result
