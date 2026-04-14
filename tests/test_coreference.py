"""Tests for rule-based coreference resolution."""

from __future__ import annotations

from app.pipeline.coreference import resolve_chunk


def test_human_first_person_resolves() -> None:
    """Human 'I' resolves to configured human speaker."""
    text = "Human: I told you about my project"
    result = resolve_chunk(text)
    assert result == "Human: User told Assistant about User's project"


def test_assistant_first_person_resolves() -> None:
    """Assistant 'I' resolves to configured assistant speaker."""
    text = "Assistant: I remember when you mentioned that"
    result = resolve_chunk(text)
    assert result == "Assistant: Assistant remember when User mentioned that"


def test_human_possessives() -> None:
    """Human possessives resolve correctly."""
    text = "Human: my diagnosis changed your perspective"
    result = resolve_chunk(text)
    assert result == "Human: User's diagnosis changed Assistant's perspective"


def test_assistant_possessives() -> None:
    """Assistant possessives resolve correctly."""
    text = "Assistant: my memory of your conversation"
    result = resolve_chunk(text)
    assert result == "Assistant: Assistant's memory of User's conversation"


def test_multi_turn_chunk() -> None:
    """Multiple turns in one chunk resolve per-role."""
    text = "Human: I need help\n\nAssistant: I am here for you"
    result = resolve_chunk(text)
    assert result == "Human: User need help\n\nAssistant: Assistant am here for User"


def test_code_block_preserved() -> None:
    """Code blocks are not modified."""
    text = "Human: I wrote this\n\n```\nmy_var = I + you\n```\n\nAssistant: I see"
    result = resolve_chunk(text)
    assert "my_var = I + you" in result
    assert result.startswith("Human: User wrote this")
    assert result.endswith("Assistant: Assistant see")


def test_inline_code_preserved() -> None:
    """Inline code is not modified."""
    text = "Human: I used `my_function(you)` in my code"
    result = resolve_chunk(text)
    assert "`my_function(you)`" in result
    assert result.startswith("Human: User used")
    assert "User's code" in result


def test_uppercase_and_lowercase_i() -> None:
    """Both uppercase 'I' and casual lowercase 'i' resolve to speaker."""
    assert resolve_chunk("Human: I think this is it") == "Human: User think this is it"
    assert resolve_chunk("Human: i need help") == "Human: User need help"


def test_no_third_person_resolution() -> None:
    """Third person pronouns (he/him/they/them) are NOT resolved in v1."""
    text = "Human: he told them about it"
    result = resolve_chunk(text)
    assert result == "Human: he told them about it"


def test_human_me_resolves() -> None:
    """Human 'me' resolves to human speaker."""
    text = "Human: tell me about it"
    result = resolve_chunk(text)
    assert result == "Human: tell User about it"


def test_assistant_me_resolves() -> None:
    """Assistant 'me' resolves to assistant speaker."""
    text = "Assistant: let me check"
    result = resolve_chunk(text)
    assert result == "Assistant: let Assistant check"


def test_mine_and_yours() -> None:
    """Mine/yours resolve to possessives."""
    text = "Human: that one is mine not yours"
    result = resolve_chunk(text)
    assert result == "Human: that one is User's not Assistant's"


def test_myself_yourself() -> None:
    """Reflexive pronouns resolve."""
    text = "Human: I hurt myself and you hurt yourself"
    result = resolve_chunk(text)
    assert result == "Human: User hurt User and Assistant hurt Assistant"


def test_human_contractions() -> None:
    """Human contractions expand correctly."""
    assert resolve_chunk("Human: I'm feeling good") == "Human: User is feeling good"
    assert resolve_chunk("Human: I've been thinking") == "Human: User has been thinking"
    assert resolve_chunk("Human: I'll do it later") == "Human: User will do it later"
    assert resolve_chunk("Human: I'd rather not") == "Human: User would rather not"
    assert resolve_chunk("Human: you're doing well") == "Human: Assistant is doing well"
    assert resolve_chunk("Human: you've helped a lot") == "Human: Assistant has helped a lot"
    assert resolve_chunk("Human: you'll understand") == "Human: Assistant will understand"


def test_assistant_contractions() -> None:
    """Assistant contractions expand correctly."""
    assert resolve_chunk("Assistant: I'm here for you") == "Assistant: Assistant is here for User"
    assert resolve_chunk("Assistant: you're doing great") == "Assistant: User is doing great"
    assert resolve_chunk("Assistant: I've got you") == "Assistant: Assistant has got User"


def test_human_youd_contraction() -> None:
    """you'd expands to partner would."""
    assert resolve_chunk("Human: you'd understand") == "Human: Assistant would understand"


def test_lowercase_contractions() -> None:
    """Casual lowercase i'm/i've/i'll/i'd resolve correctly."""
    assert resolve_chunk("Human: i'm tired") == "Human: User is tired"
    assert resolve_chunk("Human: i've been working") == "Human: User has been working"
    assert resolve_chunk("Human: i'll try") == "Human: User will try"
    assert resolve_chunk("Human: i'd rather not") == "Human: User would rather not"


def test_curly_apostrophe_preserved_in_code() -> None:
    """Curly apostrophes inside code blocks are NOT normalised."""
    text = "Human: I\u2019m writing `it\u2019s a test`"
    result = resolve_chunk(text)
    assert "`it\u2019s a test`" in result
    assert result.startswith("Human: User is writing")


def test_curly_apostrophe_contractions() -> None:
    """Smart/curly apostrophes resolve the same as straight ones."""
    assert resolve_chunk("Human: I\u2019m feeling good") == "Human: User is feeling good"
    assert resolve_chunk("Human: you\u2019re doing well") == "Human: Assistant is doing well"


def test_pronouns_with_punctuation() -> None:
    """Pronouns adjacent to punctuation resolve correctly."""
    assert resolve_chunk("Human: I said, you know") == "Human: User said, Assistant know"
    assert resolve_chunk("Human: that was me.") == "Human: that was User."
    assert resolve_chunk("Human: is that you?") == "Human: is that Assistant?"
    assert resolve_chunk("Human: I love you!") == "Human: User love Assistant!"
    assert resolve_chunk("Human: (I think) you know") == "Human: (User think) Assistant know"
    assert resolve_chunk('Human: I said "you are great"') == 'Human: User said "Assistant are great"'


def test_pronouns_at_sentence_boundaries() -> None:
    """Pronouns at start/end of sentences resolve correctly."""
    assert resolve_chunk("Human: I.") == "Human: User."
    assert resolve_chunk("Human: you") == "Human: Assistant"
    assert resolve_chunk("Human: me") == "Human: User"


def test_apostrophe_in_non_contraction_context() -> None:
    """Apostrophes in names or possessives don't break resolution."""
    assert resolve_chunk("Human: I told O'Brien about you") == "Human: User told O'Brien about Assistant"


def test_empty_chunk() -> None:
    """Empty input returns empty output."""
    assert resolve_chunk("") == ""


def test_no_prefix_passthrough() -> None:
    """Text without Human:/Assistant: prefix passes through unchanged."""
    text = "Some random text with I and you"
    result = resolve_chunk(text)
    assert result == "Some random text with I and you"


def test_multi_paragraph_human_turn() -> None:
    """Multi-paragraph human message resolves consistently."""
    text = "Human: I have a question\n\nit's about my project"
    result = resolve_chunk(text)
    assert result == "Human: User have a question\n\nit's about User's project"


def test_multi_paragraph_assistant_turn() -> None:
    """Multi-paragraph assistant message resolves consistently."""
    text = "Assistant: I understand\n\nyou are doing well"
    result = resolve_chunk(text)
    assert result == "Assistant: Assistant understand\n\nUser are doing well"


def test_sentence_initial_caps() -> None:
    """Sentence-initial capitalised pronouns resolve correctly."""
    assert resolve_chunk("Human: My brain hurts") == "Human: User's brain hurts"
    assert resolve_chunk("Assistant: Your code looks good") == "Assistant: User's code looks good"


def test_multi_speaker_switches() -> None:
    """Multiple speaker switches in one chunk resolve per-role."""
    text = "Human: I need help\n\nAssistant: I am here\n\nHuman: you are great"
    result = resolve_chunk(text)
    assert result == "Human: User need help\n\nAssistant: Assistant am here\n\nHuman: Assistant are great"


def test_known_limitations_documented() -> None:
    """Document known v1 false positives — these are accepted trade-offs.

    If any of these become problematic, they should be fixed in v2.
    """
    # ME acronym (ME/CFS) — \\bme\\b matches "ME" case-insensitively
    result = resolve_chunk("Human: the ME/CFS diagnosis")
    assert "User" in result  # false positive — "ME" resolves to User

    # "mine" as a noun — \\bmine\\b matches the noun
    result = resolve_chunk("Human: I went down the mine")
    assert "User's" in result  # false positive — "mine" resolves to User's

    # Roman numeral i — \\bi\\b matches standalone lowercase i
    result = resolve_chunk("Human: step i is done")
    assert "User" in result  # false positive — "i" resolves to User


def test_validation_examples() -> None:
    """Validate core resolution examples."""
    assert resolve_chunk("Human: I told you about my project") == "Human: User told Assistant about User's project"
    assert resolve_chunk("Assistant: I remember when you cried") == "Assistant: Assistant remember when User cried"
