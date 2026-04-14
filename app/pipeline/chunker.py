"""Turn-group chunker — breaks conversations into semantically meaningful chunks.

Strategy:
  1. Group messages into turn pairs (human + assistant)
  2. Apply boundary detection (time gaps, boot sequence)
  3. Merge small adjacent pairs, split oversized responses
  4. Classify chunk type and assign significance
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import structlog

logger = structlog.get_logger()

# Boundary thresholds
_TIME_GAP_MINUTES = 30
_MERGE_TOKEN_THRESHOLD = 150
_SPLIT_TOKEN_THRESHOLD = 600

# Rough token estimate: ~4 chars per token
_CHARS_PER_TOKEN = 4

# Chunk type detection patterns
_DECISION_PATTERNS = [
    re.compile(r"\b(decided|going with|chose|choosing|settled on|went with)\b", re.IGNORECASE),
]
_CORRECTION_PATTERNS = [
    re.compile(r"\b(no not that|don't do|instead|not that|wrong|actually no)\b", re.IGNORECASE),
]
_EMOTIONAL_WORDS = re.compile(
    r"\b(feeling|felt|scared|anxious|happy|sad|overwhelmed|frustrated|exhausted"
    r"|love|hate|angry|hurt|rough|struggling|excited|proud"
    r"|spiralling|shutdown|meltdown|burnout|dysregulated)\b",
    re.IGNORECASE,
)
_GREETING_PATTERNS = [
    re.compile(r"^(hey|hi|hello|morning|afternoon|evening|g'day)\b", re.IGNORECASE),
]


@dataclass
class MessageRef:
    """Reference to a message in the database."""

    id: int | None  # DB id, None if not yet persisted
    role: str
    content: str
    timestamp: datetime | None
    ordinal: int


@dataclass
class Chunk:
    """A chunk ready for storage."""

    content: str  # will have coreferences resolved later
    raw_content: str
    start_message_id: int | None
    end_message_id: int | None
    start_ordinal: int
    end_ordinal: int
    chunk_type: str = "topic"
    significance: int = 3
    messages: list[MessageRef] = field(default_factory=list)


def _estimate_tokens(text: str) -> int:
    """Rough token count estimate."""
    return len(text) // _CHARS_PER_TOKEN


def _classify_chunk_type(text: str, messages: list[MessageRef]) -> str:
    """Classify a chunk's type based on content heuristics."""
    # Check for corrections (human messages)
    human_text = " ".join(m.content for m in messages if m.role == "human")
    for pattern in _CORRECTION_PATTERNS:
        if pattern.search(human_text):
            return "correction"

    # Check for decisions
    for pattern in _DECISION_PATTERNS:
        if pattern.search(text):
            return "decision"

    # Check emotional density — count individual marker words
    emotional_hits = len(_EMOTIONAL_WORDS.findall(text))
    if emotional_hits >= 3 and _estimate_tokens(text) < 200:
        return "moment"
    if emotional_hits >= 2:
        return "emotional"

    return "topic"


def _calculate_significance(
    chunk_type: str,
    messages: list[MessageRef],
    is_boot: bool,
) -> int:
    """Calculate significance score (1-5)."""
    sig = 3

    if chunk_type in ("decision", "correction", "emotional"):
        sig += 1

    if chunk_type == "moment":
        sig += 1

    # Boot penalty (boot already implies greeting, so no separate greeting check needed).
    # Non-boot greetings mid-conversation also get penalised.
    if is_boot or _is_greeting(messages):
        sig -= 1

    # Clamp
    return max(1, min(5, sig))


def _is_greeting(group: list[MessageRef]) -> bool:
    """Check if a group starts with a greeting message."""
    return bool(group and group[0].role == "human" and any(p.search(group[0].content) for p in _GREETING_PATTERNS))


def _build_chunk_text(messages: list[MessageRef]) -> str:
    """Build chunk text from messages."""
    parts = []
    for msg in messages:
        prefix = "Human" if msg.role == "human" else "Assistant"
        # Strip null bytes — Postgres text columns reject \x00
        content = msg.content.replace("\x00", "")
        parts.append(f"{prefix}: {content}")
    return "\n\n".join(parts)


def chunk_conversation(
    messages: list[dict[str, Any]],
    conversation_id: int,
) -> list[Chunk]:
    """Break a conversation's messages into chunks.

    Args:
        messages: List of message dicts from DB (id, role, content, timestamp, ordinal)
        conversation_id: The conversation's DB id (for logging)

    Returns:
        List of Chunk objects ready for storage.
    """
    if not messages:
        return []

    # Convert to MessageRef
    msg_refs = [
        MessageRef(
            id=m.get("id"),
            role=m["role"],
            content=m["content"],
            timestamp=m.get("timestamp"),
            ordinal=m["ordinal"],
        )
        for m in messages
    ]

    # Step 1: Group into turn pairs
    turn_groups = _group_into_turns(msg_refs)

    # Step 2: Apply boundary detection
    bounded_groups = _apply_boundaries(turn_groups)

    # Step 3: Merge small groups, split large ones
    sized_groups = _merge_and_split(bounded_groups)

    # Step 4: Build chunks with classification
    chunks: list[Chunk] = []
    is_first = True

    for group in sized_groups:
        if not group:
            continue

        # Boot = first chunk AND starts with a greeting. A substantive first
        # message ("I got diagnosed today") is not a boot sequence.
        is_boot = is_first and _is_greeting(group)
        is_first = False

        text = _build_chunk_text(group)
        chunk_type = _classify_chunk_type(text, group)
        significance = _calculate_significance(chunk_type, group, is_boot)

        chunks.append(
            Chunk(
                content=text,
                raw_content=text,
                start_message_id=group[0].id,
                end_message_id=group[-1].id,
                start_ordinal=group[0].ordinal,
                end_ordinal=group[-1].ordinal,
                chunk_type=chunk_type,
                significance=significance,
                messages=group,
            )
        )

    logger.info(
        "chunker.complete",
        conversation_id=conversation_id,
        message_count=len(messages),
        chunk_count=len(chunks),
        types={c.chunk_type for c in chunks},
    )
    return chunks


def _group_into_turns(messages: list[MessageRef]) -> list[list[MessageRef]]:
    """Group messages into turn pairs (human + assistant responses).

    A turn starts with a human message and includes all following assistant
    messages until the next human message. Consecutive assistant messages
    (common in CC) are grouped with the preceding human message.
    """
    groups: list[list[MessageRef]] = []
    current: list[MessageRef] = []

    for msg in messages:
        if msg.role == "human" and current:
            groups.append(current)
            current = []
        current.append(msg)

    if current:
        groups.append(current)

    return groups


def _apply_boundaries(groups: list[list[MessageRef]]) -> list[list[MessageRef]]:
    """Apply hard boundaries at time gaps, splitting groups where needed.

    A time gap > 30 minutes between any two adjacent messages forces
    a chunk boundary.
    """
    result: list[list[MessageRef]] = []

    for group in groups:
        current: list[MessageRef] = []

        for msg in group:
            if (
                current
                and msg.timestamp is not None
                and current[-1].timestamp is not None
                and (msg.timestamp - current[-1].timestamp) > timedelta(minutes=_TIME_GAP_MINUTES)
            ):
                result.append(current)
                current = []
            current.append(msg)

        if current:
            result.append(current)

    return result


def _merge_and_split(groups: list[list[MessageRef]]) -> list[list[MessageRef]]:
    """Merge small adjacent groups and split oversized ones."""
    # First pass: merge small groups (boot sequence stays isolated)
    merged: list[list[MessageRef]] = []
    first_is_boot = bool(groups) and _is_greeting(groups[0])

    for i, group in enumerate(groups):
        text = _build_chunk_text(group)
        tokens = _estimate_tokens(text)

        # Never merge into/out of boot chunk
        is_boot_boundary = first_is_boot and i <= 1
        if merged and tokens < _MERGE_TOKEN_THRESHOLD and not is_boot_boundary:
            prev_text = _build_chunk_text(merged[-1])
            prev_tokens = _estimate_tokens(prev_text)
            # Don't merge across time gaps
            prev_last = merged[-1][-1]
            curr_first = group[0]
            has_time_gap = (
                prev_last.timestamp is not None
                and curr_first.timestamp is not None
                and (curr_first.timestamp - prev_last.timestamp) > timedelta(minutes=_TIME_GAP_MINUTES)
            )
            if prev_tokens + tokens < _MERGE_TOKEN_THRESHOLD and not has_time_gap:
                merged[-1].extend(group)
                continue

        merged.append(list(group))

    # Second pass: split oversized groups
    result: list[list[MessageRef]] = []

    for group in merged:
        text = _build_chunk_text(group)
        if _estimate_tokens(text) <= _SPLIT_TOKEN_THRESHOLD:
            result.append(group)
            continue

        # First try splitting at message boundaries
        sub_groups = _split_at_message_boundaries(group)

        # Then split any remaining oversized sub-groups at paragraph boundaries
        for sub_group in sub_groups:
            sub_text = _build_chunk_text(sub_group)
            if _estimate_tokens(sub_text) <= _SPLIT_TOKEN_THRESHOLD:
                result.append(sub_group)
            else:
                result.extend(_split_at_paragraph_boundaries(sub_group))

    return result


def _split_at_message_boundaries(group: list[MessageRef]) -> list[list[MessageRef]]:
    """Split a group at message boundaries when it exceeds the token threshold."""
    sub_groups: list[list[MessageRef]] = []
    current: list[MessageRef] = []
    current_tokens = 0

    for msg in group:
        msg_tokens = _estimate_tokens(msg.content)

        if current and current_tokens + msg_tokens > _SPLIT_TOKEN_THRESHOLD:
            sub_groups.append(current)
            current = []
            current_tokens = 0

        current.append(msg)
        current_tokens += msg_tokens

    if current:
        sub_groups.append(current)

    return sub_groups


def _split_at_paragraph_boundaries(group: list[MessageRef]) -> list[list[MessageRef]]:
    """Split an oversized group by splitting long messages at paragraph boundaries.

    Creates synthetic MessageRef objects for each fragment. Both fragments
    share the same id, ordinal, and timestamp as the original message —
    provenance traces back to the same source message.

    See SPIKE issue for downstream implications of shared ordinals.
    """
    # Find the oversized message(s) and split their content
    expanded: list[MessageRef] = []
    for msg in group:
        if _estimate_tokens(msg.content) > _SPLIT_TOKEN_THRESHOLD:
            fragments = _split_text_at_paragraphs(msg.content, _SPLIT_TOKEN_THRESHOLD)
            for fragment in fragments:
                expanded.append(
                    MessageRef(
                        id=msg.id,
                        role=msg.role,
                        content=fragment,
                        timestamp=msg.timestamp,
                        ordinal=msg.ordinal,
                    )
                )
        else:
            expanded.append(msg)

    # Now re-split the expanded list at message boundaries
    return _split_at_message_boundaries(expanded)


def _split_text_at_paragraphs(text: str, threshold: int) -> list[str]:
    """Split text at paragraph boundaries when it exceeds token threshold."""
    if _estimate_tokens(text) <= threshold:
        return [text]

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _estimate_tokens(para)

        if current and current_tokens + para_tokens > threshold:
            chunks.append("\n\n".join(current))
            current = [para]
            current_tokens = para_tokens
        else:
            current.append(para)
            current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks
