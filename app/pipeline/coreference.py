"""Rule-based coreference resolution (v1).

Resolves first/second person pronouns to named entities based on speaker role.
Operates on chunk text before embedding to improve search quality.

Speaker names are configured via VEX_BRAIN_HUMAN_SPEAKER and VEX_BRAIN_ASSISTANT_SPEAKER
environment variables (default: "User" and "Assistant").

Rules:
  Human turns:  I/me/my/mine → human_speaker, you/your/yours → assistant_speaker
  Assistant turns: I/me/my/mine → assistant_speaker, you/your/yours → human_speaker

Skips text inside code blocks (``` ... ```) and inline code (` ... `).
"""

from __future__ import annotations

import re

from app.config import get_settings

# Regex to match code blocks and inline code — these are protected from resolution
_CODE_BLOCK = re.compile(r"```[\s\S]*?```")
_INLINE_CODE = re.compile(r"`[^`]+`")


def _build_replacements(speaker: str, partner: str) -> list[tuple[re.Pattern, str]]:
    """Build ordered replacement rules for a given speaker/partner pair.

    Order matters — possessives before base forms to avoid partial matches.
    """
    return [
        # Contractions (must come before base pronoun forms).
        # Case-insensitive to catch casual "i'm" as well as "I'm".
        (re.compile(r"\bI'm\b", re.IGNORECASE), f"{speaker} is"),
        (re.compile(r"\bI've\b", re.IGNORECASE), f"{speaker} has"),
        (re.compile(r"\bI'll\b", re.IGNORECASE), f"{speaker} will"),
        (re.compile(r"\bI'd\b", re.IGNORECASE), f"{speaker} would"),
        (re.compile(r"\byou're\b", re.IGNORECASE), f"{partner} is"),
        (re.compile(r"\byou've\b", re.IGNORECASE), f"{partner} has"),
        (re.compile(r"\byou'll\b", re.IGNORECASE), f"{partner} will"),
        (re.compile(r"\byou'd\b", re.IGNORECASE), f"{partner} would"),
        # Possessive pronouns (before base forms)
        (re.compile(r"\bmine\b", re.IGNORECASE), f"{speaker}'s"),
        (re.compile(r"\byours\b", re.IGNORECASE), f"{partner}'s"),
        (re.compile(r"\bmy\b", re.IGNORECASE), f"{speaker}'s"),
        (re.compile(r"\byour\b", re.IGNORECASE), f"{partner}'s"),
        # Reflexive (before base forms)
        (re.compile(r"\bmyself\b", re.IGNORECASE), speaker),
        (re.compile(r"\byourself\b", re.IGNORECASE), partner),
        # Object pronouns
        (re.compile(r"\bme\b", re.IGNORECASE), speaker),
        # Subject pronouns — match both "I" and casual lowercase "i"
        (re.compile(r"\bI\b"), speaker),
        (re.compile(r"\bi\b"), speaker),
        (re.compile(r"\byou\b", re.IGNORECASE), partner),
    ]


def _get_rules() -> tuple[list[tuple[re.Pattern, str]], list[tuple[re.Pattern, str]]]:
    """Build rules from current config settings."""
    settings = get_settings()
    human = settings.human_speaker
    assistant = settings.assistant_speaker
    human_rules = _build_replacements(human, assistant)
    assistant_rules = _build_replacements(assistant, human)
    return human_rules, assistant_rules


def _protect_code(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Replace code blocks and inline code with placeholders.

    Returns the modified text and a list of (placeholder, original) pairs
    for restoration after resolution.
    """
    replacements: list[tuple[str, str]] = []
    counter = 0

    def make_placeholder(match: re.Match) -> str:
        nonlocal counter
        placeholder = f"\x00CODE{counter}\x00"
        counter += 1
        replacements.append((placeholder, match.group(0)))
        return placeholder

    # Code blocks first (they may contain inline code)
    text = _CODE_BLOCK.sub(make_placeholder, text)
    text = _INLINE_CODE.sub(make_placeholder, text)

    return text, replacements


def _restore_code(text: str, replacements: list[tuple[str, str]]) -> str:
    """Restore code blocks from placeholders."""
    for placeholder, original in replacements:
        text = text.replace(placeholder, original)
    return text


def _apply_rules(text: str, rules: list[tuple[re.Pattern, str]]) -> str:
    """Apply replacement rules to text."""
    for pattern, replacement in rules:
        text = pattern.sub(replacement, text)
    return text


def resolve_chunk(chunk_text: str) -> str:
    """Resolve coreferences in a chunk's text.

    The chunk text has the format:
        Human: message content

        Assistant: response content

    Each turn is resolved according to its speaker role.
    """
    human_rules, assistant_rules = _get_rules()

    # Protect code blocks FIRST — before apostrophe normalisation
    text, code_replacements = _protect_code(chunk_text)

    # Normalise curly/smart apostrophes to straight (outside code blocks)
    text = text.replace("\u2019", "'").replace("\u2018", "'")

    # Split into paragraphs and resolve each according to speaker role.
    # Track current role across blank-line splits so multi-paragraph
    # messages (where only the first paragraph has the "Human: " prefix)
    # are resolved consistently.
    parts = text.split("\n\n")
    resolved_parts = []
    current_rules: list[tuple[re.Pattern, str]] | None = None

    for part in parts:
        if part.startswith("Human: "):
            prefix = "Human: "
            content = part[len(prefix):]
            current_rules = human_rules
            resolved_parts.append(prefix + _apply_rules(content, current_rules))
        elif part.startswith("Assistant: "):
            prefix = "Assistant: "
            content = part[len(prefix):]
            current_rules = assistant_rules
            resolved_parts.append(prefix + _apply_rules(content, current_rules))
        elif current_rules is not None:
            # Continuation paragraph — same speaker as previous
            resolved_parts.append(_apply_rules(part, current_rules))
        else:
            # No speaker context yet — leave unchanged
            resolved_parts.append(part)

    result = "\n\n".join(resolved_parts)

    # Restore code blocks
    return _restore_code(result, code_replacements)
