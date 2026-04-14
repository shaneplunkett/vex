"""Agent extraction prompts — version-controlled alongside the API extraction prompt."""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def get_agent_prompt() -> str:
    """Return the current agent extraction prompt for CC subagent use."""
    return (_PROMPTS_DIR / "agent_extraction.md").read_text(encoding="utf-8")
