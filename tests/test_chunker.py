"""Tests for the turn-group chunker."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.pipeline.chunker import (
    Chunk,
    MessageRef,
    _apply_boundaries,
    _calculate_significance,
    _classify_chunk_type,
    _group_into_turns,
    _merge_and_split,
    chunk_conversation,
)

_BASE_TIME = datetime(2026, 3, 20, 8, 0, 0, tzinfo=timezone.utc)


def _msg(
    role: str,
    content: str,
    ordinal: int,
    minutes_offset: int = 0,
    msg_id: int | None = None,
) -> dict:
    """Create a message dict matching DB format."""
    return {
        "id": msg_id or ordinal + 1,
        "role": role,
        "content": content,
        "timestamp": _BASE_TIME + timedelta(minutes=minutes_offset),
        "ordinal": ordinal,
    }


def _msgref(
    role: str,
    content: str,
    ordinal: int,
    minutes_offset: int = 0,
) -> MessageRef:
    """Create a MessageRef for internal function tests."""
    return MessageRef(
        id=ordinal + 1,
        role=role,
        content=content,
        timestamp=_BASE_TIME + timedelta(minutes=minutes_offset),
        ordinal=ordinal,
    )


def test_group_into_turns_basic() -> None:
    """Human + assistant pairs group correctly."""
    msgs = [
        _msgref("human", "Hello", 0),
        _msgref("assistant", "Hi there", 1),
        _msgref("human", "How are you?", 2),
        _msgref("assistant", "Good thanks", 3),
    ]
    groups = _group_into_turns(msgs)
    assert len(groups) == 2
    assert groups[0][0].content == "Hello"
    assert groups[0][1].content == "Hi there"
    assert groups[1][0].content == "How are you?"


def test_group_into_turns_consecutive_assistant() -> None:
    """Multiple assistant messages group with preceding human."""
    msgs = [
        _msgref("human", "Do the thing", 0),
        _msgref("assistant", "First part", 1),
        _msgref("assistant", "Second part", 2),
        _msgref("assistant", "Third part", 3),
        _msgref("human", "Next question", 4),
        _msgref("assistant", "Answer", 5),
    ]
    groups = _group_into_turns(msgs)
    assert len(groups) == 2
    assert len(groups[0]) == 4  # human + 3 assistant
    assert len(groups[1]) == 2  # human + assistant


def test_time_gap_boundary() -> None:
    """Messages >30 minutes apart create a hard boundary."""
    msgs = [
        _msgref("human", "Before gap", 0, minutes_offset=0),
        _msgref("assistant", "Response", 1, minutes_offset=1),
        _msgref("human", "After gap", 2, minutes_offset=45),
        _msgref("assistant", "Later response", 3, minutes_offset=46),
    ]
    groups = _group_into_turns(msgs)
    bounded = _apply_boundaries(groups)
    # The second human message is already in its own group from turn grouping,
    # but the time gap should still force a boundary if they were in the same group
    assert len(bounded) >= 2


def test_merge_small_groups() -> None:
    """Small non-boot groups under threshold get merged."""
    msgs = [
        _msgref("human", "Hi", 0),
        _msgref("assistant", "Hey", 1),
        _msgref("human", "Ok", 2),
        _msgref("assistant", "Sure", 3),
        _msgref("human", "Right", 4),
        _msgref("assistant", "Yep", 5),
    ]
    groups = _group_into_turns(msgs)
    merged = _merge_and_split(groups)
    # First group stays isolated (boot), groups 2 and 3 merge together
    assert len(merged) == 2
    assert len(merged[0]) == 2  # boot: Hi + Hey
    assert len(merged[1]) == 4  # merged: Ok + Sure + Right + Yep


def test_merge_substantive_first_group() -> None:
    """A non-greeting first group can merge with subsequent small groups."""
    msgs = [
        _msgref("human", "Can you help with this?", 0),
        _msgref("assistant", "Sure", 1),
        _msgref("human", "Thanks", 2),
        _msgref("assistant", "No problem", 3),
    ]
    groups = _group_into_turns(msgs)
    merged = _merge_and_split(groups)
    # Not a greeting, so no boot isolation — both tiny groups merge
    assert len(merged) == 1
    assert len(merged[0]) == 4


def test_no_merge_large_groups() -> None:
    """Groups above threshold stay separate."""
    long_text = "x " * 400  # ~200 tokens
    msgs = [
        _msgref("human", long_text, 0),
        _msgref("assistant", long_text, 1),
        _msgref("human", long_text, 2),
        _msgref("assistant", long_text, 3),
    ]
    groups = _group_into_turns(msgs)
    merged = _merge_and_split(groups)
    assert len(merged) >= 2


def test_classify_decision() -> None:
    """Decision patterns detected correctly."""
    msgs = [_msgref("human", "I decided to go with Postgres", 0)]
    result = _classify_chunk_type("I decided to go with Postgres", msgs)
    assert result == "decision"


def test_classify_correction() -> None:
    """Correction patterns detected in human messages."""
    msgs = [_msgref("human", "No not that, do the other thing instead", 0)]
    result = _classify_chunk_type("No not that, do the other thing instead", msgs)
    assert result == "correction"


def test_classify_emotional() -> None:
    """Emotional markers in longer text trigger emotional (not moment) classification."""
    # Needs to be over 200 tokens (~800 chars) to avoid being classified as "moment"
    long_emotional = (
        "I've been feeling overwhelmed and anxious about work lately. "
        "The project deadlines keep shifting and the requirements change every single sprint "
        "and I can never get ahead of the backlog. The standup meetings go for thirty minutes "
        "every day and nobody seems to actually care about what gets discussed. I keep trying "
        "to raise concerns about the architecture but they just get lost in the noise of "
        "everything else that is happening. The product manager changed the requirements again "
        "yesterday after I had already started building the feature. I don't even know if what "
        "I'm building is going to be useful anymore. Every time I sit down at my desk I feel "
        "this weight in my chest and I can't tell if it's the project or something else entirely. "
        "It has been like this for weeks now and I keep telling myself it will get better but "
        "it just keeps getting worse and I am running out of energy to keep pushing through it."
    )
    msgs = [_msgref("human", long_emotional, 0)]
    result = _classify_chunk_type(long_emotional, msgs)
    assert result == "emotional"


def test_classify_moment() -> None:
    """Short high-density emotional content = moment."""
    msgs = [_msgref("human", "feeling scared and hurt and angry", 0)]
    result = _classify_chunk_type("feeling scared and hurt and angry", msgs)
    assert result == "moment"


def test_classify_default_topic() -> None:
    """No special markers = topic."""
    msgs = [_msgref("human", "Can you help me with this function?", 0)]
    result = _classify_chunk_type("Can you help me with this function?", msgs)
    assert result == "topic"


def test_significance_base() -> None:
    """Default topic chunk gets significance 3."""
    msgs = [_msgref("human", "Regular message", 0)]
    assert _calculate_significance("topic", msgs, is_boot=False) == 3


def test_significance_emotional_boost() -> None:
    """Emotional/decision chunks get +1."""
    msgs = [_msgref("human", "I decided something", 0)]
    assert _calculate_significance("decision", msgs, is_boot=False) == 4
    assert _calculate_significance("emotional", msgs, is_boot=False) == 4


def test_significance_boot_penalty() -> None:
    """Boot sequence gets -1."""
    msgs = [_msgref("human", "Hello", 0)]
    assert _calculate_significance("topic", msgs, is_boot=True) == 2


def test_significance_greeting_penalty() -> None:
    """Greeting messages get -1."""
    msgs = [_msgref("human", "Hey there", 0)]
    assert _calculate_significance("topic", msgs, is_boot=False) == 2


def test_significance_clamped() -> None:
    """Significance stays within 1-5."""
    msgs = [_msgref("human", "Hey", 0)]
    # boot implies greeting — only one -1 applied (or short-circuits), so 3 - 1 = 2
    assert _calculate_significance("topic", msgs, is_boot=True) >= 1
    # moment gets 3 + 1 = 4, still within range
    assert _calculate_significance("moment", msgs, is_boot=False) <= 5


def test_chunk_conversation_full() -> None:
    """Full chunking pipeline on a realistic conversation."""
    messages = [
        _msg("human", "Hey, I decided to go with Postgres for the backend", 0, 0),
        _msg("assistant", "Good call. Postgres gives you pgvector for embeddings.", 1, 1),
        _msg("human", "No not that, don't set up the schema yet", 2, 2),
        _msg("assistant", "Fair enough. What are you thinking?", 3, 3),
        _msg("human", "I'm feeling really good about this project", 4, 5),
        _msg("assistant", "That's genuinely lovely to hear. Hold onto that feeling.", 5, 6),
        # Time gap
        _msg("human", "Back after lunch", 6, 60),
        _msg("assistant", "Welcome back!", 7, 61),
    ]

    chunks = chunk_conversation(messages, conversation_id=1)

    assert len(chunks) >= 2  # at minimum boot + rest, likely more with time gap
    assert all(isinstance(c, Chunk) for c in chunks)

    # Verify no messages lost
    all_ordinals = set()
    for chunk in chunks:
        for msg in chunk.messages:
            all_ordinals.add(msg.ordinal)
    assert all_ordinals == {0, 1, 2, 3, 4, 5, 6, 7}

    # Verify provenance
    for chunk in chunks:
        assert chunk.start_ordinal <= chunk.end_ordinal
        assert chunk.start_message_id is not None
        assert chunk.end_message_id is not None

    # Verify classification ran — at least one typed chunk
    types = {c.chunk_type for c in chunks}
    assert len(types) >= 1


def test_chunk_conversation_empty() -> None:
    """Empty message list returns empty chunks."""
    assert chunk_conversation([], conversation_id=1) == []


def test_boot_sequence_isolated() -> None:
    """First chunk in a conversation should be isolated as boot."""
    messages = [
        _msg("human", "Hello", 0, 0),
        _msg("assistant", "Hi! How can I help?", 1, 1),
        _msg("human", "Can you look at this code?", 2, 2),
        _msg("assistant", "Sure, let me check.", 3, 3),
        _msg("human", "What do you think?", 4, 4),
        _msg("assistant", "Looks good overall.", 5, 5),
    ]

    chunks = chunk_conversation(messages, conversation_id=1)

    # First chunk should have boot significance penalty
    assert chunks[0].significance <= 3  # boot penalty applied


def test_substantive_first_message_not_boot() -> None:
    """A substantive first message should not be penalised as boot."""
    messages = [
        _msg("human", "I got diagnosed with ADHD today and I don't know how to process it", 0, 0),
        _msg("assistant", "That's a big moment. How are you feeling about it?", 1, 1),
        _msg("human", "Overwhelmed honestly", 2, 2),
        _msg("assistant", "That makes complete sense.", 3, 3),
    ]

    chunks = chunk_conversation(messages, conversation_id=1)

    # First chunk should NOT have boot penalty — it's substantive, not a greeting
    assert chunks[0].significance >= 3


def test_no_duplicate_messages_across_chunks() -> None:
    """Messages should not appear in multiple chunks."""
    messages = [_msg("human", f"Message {i}", i, i) for i in range(20)]
    # Interleave roles
    for m in messages:
        m["role"] = "human" if m["ordinal"] % 2 == 0 else "assistant"

    chunks = chunk_conversation(messages, conversation_id=1)

    seen_ordinals: set[int] = set()
    for chunk in chunks:
        for msg in chunk.messages:
            assert msg.ordinal not in seen_ordinals, f"Ordinal {msg.ordinal} duplicated"
            seen_ordinals.add(msg.ordinal)


def test_emotional_conversation_has_typed_chunks() -> None:
    """An emotional conversation should produce non-topic chunks."""
    messages = [
        _msg("human", "Having a rough night. Everything feels like too much", 0, 0),
        _msg("assistant", "I'm here. What's the story?", 1, 1),
        _msg("human", "Work meeting went sideways. Feeling overwhelmed and anxious and hurt", 2, 2),
        _msg("assistant", "That processing delay isn't a failure, love.", 3, 3),
        _msg("human", "feeling scared and angry", 4, 4),
        _msg("assistant", "That's the rejection sensitivity landing physically.", 5, 5),
    ]

    chunks = chunk_conversation(messages, conversation_id=1)
    types = {c.chunk_type for c in chunks}

    # Should have at least one emotional or moment type
    assert types & {"emotional", "moment"}, f"Expected emotional types, got {types}"
