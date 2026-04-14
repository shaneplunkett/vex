"""Tests for agent I/O — export chunks and import extraction results."""

from __future__ import annotations

import json
from pathlib import Path

from app.pipeline.agent_io import export_chunks, get_exportable_conversations, import_results
from tests.conftest import insert_chunk, insert_conversation


async def _create_embedded_conversation(
    session_id: str = "test-embedded-001",
    chunk_contents: list[str] | None = None,
) -> tuple[int, list[int]]:
    """Helper: create a conversation at 'embedded' status with chunks."""
    conv_id = await insert_conversation(
        session_id=session_id,
        pipeline_status="embedded",
    )
    contents = chunk_contents or ["First chunk content", "Second chunk content"]
    chunk_ids = []
    for i, content in enumerate(contents):
        cid = await insert_chunk(
            conversation_id=conv_id,
            raw_content=content,
            chunk_type="topic",
            significance=3,
            start_ordinal=i * 2,
            end_ordinal=i * 2 + 1,
        )
        chunk_ids.append(cid)
    return conv_id, chunk_ids


# ---------------------------------------------------------------------------
# get_exportable_conversations
# ---------------------------------------------------------------------------


async def test_get_exportable_finds_embedded() -> None:
    await _create_embedded_conversation()
    await insert_conversation(session_id="test-extracted-001", pipeline_status="extracted")

    result = await get_exportable_conversations()
    assert len(result) == 1
    assert result[0]["pipeline_status"] == "embedded"


async def test_get_exportable_empty_when_none() -> None:
    result = await get_exportable_conversations()
    assert result == []


# ---------------------------------------------------------------------------
# export_chunks
# ---------------------------------------------------------------------------


async def test_export_writes_jsonl(tmp_path: Path) -> None:
    conv_id, chunk_ids = await _create_embedded_conversation()

    result = await export_chunks(tmp_path / "export")
    assert result["exported"] == 1
    assert result["total_chunks"] == 2

    jsonl_file = tmp_path / "export" / f"conv_{conv_id}.jsonl"
    assert jsonl_file.exists()

    lines = jsonl_file.read_text().strip().split("\n")
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["id"] == chunk_ids[0]
    assert first["raw_content"] == "First chunk content"
    assert first["chunk_type"] == "topic"
    assert first["significance"] == 3


async def test_export_copies_prompt(tmp_path: Path) -> None:
    await _create_embedded_conversation()
    await export_chunks(tmp_path / "export")

    prompt_file = tmp_path / "export" / "prompt.md"
    assert prompt_file.exists()
    assert "entity_type" in prompt_file.read_text()


async def test_export_skips_extracted(tmp_path: Path) -> None:
    conv_id = await insert_conversation(session_id="test-done-001", pipeline_status="extracted")

    result = await export_chunks(tmp_path / "export", conversation_ids=[conv_id])
    # Should skip — conversation is already extracted
    assert result["exported"] == 0


async def test_export_specific_ids(tmp_path: Path) -> None:
    conv1, _ = await _create_embedded_conversation(session_id="test-a")
    await _create_embedded_conversation(session_id="test-b")

    result = await export_chunks(tmp_path / "export", conversation_ids=[conv1])
    assert result["exported"] == 1


# ---------------------------------------------------------------------------
# import_results
# ---------------------------------------------------------------------------


async def test_import_no_files(tmp_path: Path) -> None:
    result = await import_results(tmp_path)
    assert "error" in result


async def test_import_dry_run(tmp_path: Path) -> None:
    conv_id, chunk_ids = await _create_embedded_conversation()

    result_data = {
        "conversation_id": conv_id,
        "chunks": [
            {
                "chunk_id": chunk_ids[0],
                "entities": [
                    {
                        "name": "TestEntity",
                        "entity_type": "Tool",
                        "match": "new",
                        "confidence": 0.9,
                        "reasoning": "test",
                    }
                ],
                "relations": [],
                "flags": [],
            }
        ],
    }
    (tmp_path / f"result_{conv_id}.json").write_text(json.dumps(result_data))

    result = await import_results(tmp_path, dry_run=True)
    assert result["dry_run"] is True
    assert result["conversations"] == 1
    assert result["chunks_processed"] == 1


async def test_import_applies_through_linker(tmp_path: Path) -> None:
    conv_id, chunk_ids = await _create_embedded_conversation()

    result_data = {
        "conversation_id": conv_id,
        "chunks": [
            {
                "chunk_id": chunk_ids[0],
                "entities": [
                    {"name": "NixOS", "entity_type": "Tool", "match": "new", "confidence": 0.95, "reasoning": "test"}
                ],
                "relations": [],
                "flags": [],
            }
        ],
    }
    (tmp_path / f"result_{conv_id}.json").write_text(json.dumps(result_data))

    result = await import_results(tmp_path)
    assert result["conversations"] == 1
    assert result["chunks_processed"] == 1
    assert result["chunks_failed"] == 0

    # Verify conversation status was updated
    from app.db import get_pool

    pool = get_pool()
    async with pool.acquire() as conn:
        status = await conn.fetchval("SELECT pipeline_status FROM conversations WHERE id = $1", conv_id)
    assert status == "extracted"


async def test_import_handles_malformed_json(tmp_path: Path) -> None:
    (tmp_path / "result_999.json").write_text("not valid json{{{")

    result = await import_results(tmp_path)
    assert result["conversations"] == 0
    assert len(result.get("errors", [])) == 1


async def test_import_handles_missing_chunk_id(tmp_path: Path) -> None:
    conv_id, _ = await _create_embedded_conversation()

    result_data = {
        "conversation_id": conv_id,
        "chunks": [
            {
                "entities": [{"name": "X", "entity_type": "Tool", "match": "new", "confidence": 0.9}],
                "relations": [],
                "flags": [],
            }
        ],
    }
    (tmp_path / f"result_{conv_id}.json").write_text(json.dumps(result_data))

    result = await import_results(tmp_path)
    assert result["chunks_failed"] == 1
