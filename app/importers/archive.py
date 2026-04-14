"""Claude.ai archive parser — converts conversations-clean.json to L1 format.

The archive is a JSON array of conversations, each with:
  - uuid: unique conversation identifier
  - name: conversation title
  - created_at: ISO 8601 timestamp
  - messages: array of {role: "human"|"assistant", text: string}
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from app.importers.cc import ParsedConversation, ParsedMessage

logger = structlog.get_logger()


def _parse_timestamp(ts: Any) -> datetime | None:
    """Parse an ISO 8601 timestamp from archive records."""
    if not isinstance(ts, str) or not ts:
        logger.warning("archive.bad_timestamp", timestamp=ts)
        return None
    try:
        ts = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(ts)
    except ValueError:
        logger.warning("archive.bad_timestamp", timestamp=ts)
        return None


def _parse_conversation(data: dict[str, Any]) -> ParsedConversation | None:
    """Parse a single conversation from the archive format."""
    uuid = data.get("uuid")
    if not uuid:
        logger.warning("archive.missing_uuid")
        return None

    name = data.get("name") or "Untitled conversation"
    raw_messages = data.get("messages", [])
    if not isinstance(raw_messages, list):
        logger.warning("archive.invalid_messages", uuid=uuid[:8])
        return None

    if not raw_messages:
        logger.debug("archive.skip_empty", uuid=uuid[:8])
        return None

    messages: list[ParsedMessage] = []
    created_at = _parse_timestamp(data.get("created_at", ""))

    prev_uuid: str | None = None
    for i, msg in enumerate(raw_messages):
        if not isinstance(msg, dict):
            continue

        role = msg.get("role", "")
        text = msg.get("text", "").replace("\x00", "").strip() if isinstance(msg.get("text"), str) else ""

        if not text or role not in ("human", "assistant"):
            continue

        msg_uuid = f"{uuid}-{i}"
        messages.append(
            ParsedMessage(
                role=role,
                content=text,
                timestamp=created_at or datetime.now().astimezone(),
                uuid=msg_uuid,
                parent_uuid=prev_uuid,
                ordinal=len(messages),
            )
        )
        prev_uuid = msg_uuid

    if not messages:
        logger.debug("archive.skip_no_content", uuid=uuid[:8])
        return None

    return ParsedConversation(
        session_id=uuid,
        name=name,
        messages=messages,
        started_at=messages[0].timestamp,
        ended_at=messages[-1].timestamp,
        source_path="",
    )


def parse_archive(path: Path) -> list[ParsedConversation]:
    """Parse a conversations-clean.json archive file.

    Returns a list of parsed conversations, skipping empty ones.
    """
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        logger.exception("archive.parse_error", path=str(path))
        return []

    if not isinstance(data, list):
        logger.error("archive.invalid_format", path=str(path), reason="expected JSON array")
        return []

    conversations: list[ParsedConversation] = []

    for entry in data:
        if not isinstance(entry, dict):
            continue
        conversation = _parse_conversation(entry)
        if conversation is not None:
            conversations.append(conversation)

    logger.info(
        "archive.parse_complete",
        total=len(data),
        parsed=len(conversations),
        skipped=len(data) - len(conversations),
    )
    return conversations
