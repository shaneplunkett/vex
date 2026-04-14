"""CC JSONL session parser — converts Claude Code session files to conversations + messages.

CC stores sessions as JSONL files under ~/.claude/projects/<project-slug>/<session-id>.jsonl.
Each line is a JSON record with a `type` field. We care about:
  - "user" records with string content in .message.content (actual human messages)
  - "assistant" records with array content in .message.content (model responses)

We skip: file-history-snapshot, progress, queue-operation, system, pr-link, last-prompt,
and user records that are tool results (content is array with tool_result blocks).

Conversation tree is reconstructed via parentUuid chains. Sidechain branches
(isSidechain: true) are filtered out. Subagent files (in subagents/ dirs) are
skipped at the file discovery level.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

# Record types we care about for conversation content
_CONTENT_TYPES = {"user", "assistant"}


@dataclass
class ParsedMessage:
    """A single message extracted from a CC JSONL session."""

    role: str  # "human" or "assistant" — mapped from CC's "user"/"assistant"
    content: str
    timestamp: datetime
    uuid: str
    parent_uuid: str | None
    ordinal: int = 0  # set during linearisation


@dataclass
class ParsedConversation:
    """A conversation extracted from a CC JSONL session file."""

    session_id: str
    name: str
    messages: list[ParsedMessage] = field(default_factory=list)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    source_path: str = ""

    @property
    def message_count(self) -> int:
        return len(self.messages)


def _extract_text_content(message: dict[str, Any]) -> str | None:
    """Extract text content from a CC message, discarding thinking/tool blocks.

    User messages have string content. Assistant messages have an array of
    content blocks — we keep only type: "text" blocks.
    """
    content = message.get("content")
    if content is None:
        return None

    if isinstance(content, str):
        clean = content.replace("\x00", "").strip()
        return clean if clean else None

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "").replace("\x00", "").strip()
                if text:
                    text_parts.append(text)
        return "\n\n".join(text_parts) if text_parts else None

    return None


def _generate_name(messages: list[ParsedMessage]) -> str:
    """Generate a conversation name from the first human message, truncated to 80 chars."""
    for msg in messages:
        if msg.role == "human":
            name = msg.content.replace("\n", " ").strip()
            if len(name) > 80:
                # Truncate at word boundary
                name = name[:77].rsplit(" ", 1)[0] + "..."
            return name
    return "Untitled conversation"


def _parse_timestamp(ts: str) -> datetime | None:
    """Parse an ISO 8601 timestamp from CC records.

    Returns None for malformed timestamps so a single bad record
    doesn't abort the entire file.
    """
    try:
        # CC uses format like "2026-02-25T07:04:20.086Z"
        ts = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(ts)
    except ValueError:
        logger.warning("cc.bad_timestamp", timestamp=ts)
        return None


def _linearise(records: list[dict[str, Any]]) -> list[ParsedMessage]:
    """Reconstruct the linear conversation from a tree of records.

    CC stores messages as a tree via parentUuid. We walk from roots to leaves,
    always choosing the non-sidechain path when branches exist.
    """
    by_uuid: dict[str, dict[str, Any]] = {}
    children: dict[str | None, list[str]] = {}

    for record in records:
        uuid = record.get("uuid")
        if not uuid:
            continue
        by_uuid[uuid] = record
        parent = record.get("parentUuid")
        children.setdefault(parent, []).append(uuid)

    # Find roots — records whose parent is null or not in our filtered set.
    # We need to walk from the absolute root of the conversation tree, not just
    # from content records. But we only have content records. So a record is a
    # root if its parent isn't another content record we're tracking.
    roots = [
        r["uuid"] for r in records if r.get("uuid") and (r.get("parentUuid") is None or r["parentUuid"] not in by_uuid)
    ]

    if not roots:
        return []

    messages: list[ParsedMessage] = []
    visited: set[str] = set()

    def walk(uuid: str) -> None:
        if uuid in visited:
            return
        visited.add(uuid)

        record = by_uuid.get(uuid)
        if record is None:
            return

        rec_type = record.get("type")
        message_data = record.get("message")

        if rec_type in _CONTENT_TYPES and message_data is not None:
            text = _extract_text_content(message_data)
            ts_raw = record.get("timestamp")
            timestamp = _parse_timestamp(ts_raw) if ts_raw else None
            if text and timestamp is not None:
                role = "human" if rec_type == "user" else "assistant"
                messages.append(
                    ParsedMessage(
                        role=role,
                        content=text,
                        timestamp=timestamp,
                        uuid=uuid,
                        parent_uuid=record.get("parentUuid"),
                    )
                )

        # Walk children, preferring non-sidechain, ordered by timestamp
        child_uuids = children.get(uuid, [])
        child_uuids.sort(
            key=lambda u: (
                by_uuid.get(u, {}).get("isSidechain", False),
                by_uuid.get(u, {}).get("timestamp", ""),
            )
        )

        for child_uuid in child_uuids:
            child = by_uuid.get(child_uuid, {})
            if child.get("isSidechain", False):
                continue
            walk(child_uuid)

    for root_uuid in roots:
        walk(root_uuid)

    # Assign ordinals
    for i, msg in enumerate(messages):
        msg.ordinal = i

    return messages


def parse_session_file(path: Path) -> ParsedConversation | None:
    """Parse a single CC JSONL session file into a ParsedConversation.

    Returns None if the file contains no extractable messages (e.g., all tool calls).
    """
    records: list[dict[str, Any]] = []
    session_id: str | None = None

    try:
        with path.open() as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "cc.parse_skip_line",
                        path=str(path),
                        line=line_num,
                        reason="invalid JSON",
                    )
                    continue

                if session_id is None and record.get("sessionId"):
                    session_id = record["sessionId"]

                rec_type = record.get("type", "")
                if rec_type not in _CONTENT_TYPES:
                    continue

                # Skip user records that are tool results (content is an array
                # containing tool_result blocks, not actual human text)
                if rec_type == "user":
                    message = record.get("message", {})
                    content = message.get("content") if isinstance(message, dict) else None
                    if isinstance(content, list) and any(
                        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
                    ):
                        continue

                records.append(record)

    except OSError:
        logger.exception("cc.parse_file_error", path=str(path))
        return None

    if not session_id:
        session_id = path.stem

    messages = _linearise(records)

    if not messages:
        logger.debug("cc.parse_skip_empty", path=str(path), session_id=session_id)
        return None

    name = _generate_name(messages)
    started_at = messages[0].timestamp
    ended_at = messages[-1].timestamp

    return ParsedConversation(
        session_id=session_id,
        name=name,
        messages=messages,
        started_at=started_at,
        ended_at=ended_at,
        source_path=str(path),
    )


def discover_session_files(root: Path) -> list[Path]:
    """Find all CC JSONL session files under a root directory.

    Skips files in subagents/ directories.
    """
    files: list[Path] = []
    for jsonl_file in root.rglob("*.jsonl"):
        if "subagents" in jsonl_file.parts:
            continue
        files.append(jsonl_file)

    files.sort()
    logger.info("cc.discovered_files", root=str(root), count=len(files))
    return files


def parse_all(root: Path) -> list[ParsedConversation]:
    """Parse all CC JSONL files under a root directory.

    Returns a list of parsed conversations, skipping empty ones.
    """
    files = discover_session_files(root)
    conversations: list[ParsedConversation] = []

    for session_file in files:
        conversation = parse_session_file(session_file)
        if conversation is not None:
            conversations.append(conversation)

    logger.info(
        "cc.parse_complete",
        total_files=len(files),
        parsed=len(conversations),
        skipped=len(files) - len(conversations),
    )
    return conversations
