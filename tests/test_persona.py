"""Tests for persona tools — get, update, delete."""

from __future__ import annotations

import pytest


class TestPersona:
    """Tests for persona CRUD operations."""

    @pytest.fixture(autouse=True)
    async def _setup(self, _setup_db: None) -> None:
        """Ensure DB is ready."""

    async def test_get_persona_empty(self) -> None:
        from app.tools.persona import get_persona

        result = await get_persona()
        assert result["sections"] == []
        assert result["full_text"] == ""
        assert "error" in result

    async def test_update_creates_section(self) -> None:
        from app.tools.persona import get_persona, update_persona

        result = await update_persona("identity", "# Identity\nI am an AI assistant.")
        assert result["action"] == "created"
        assert result["key"] == "identity"

        persona = await get_persona()
        assert len(persona["sections"]) == 1
        assert persona["sections"][0]["key"] == "identity"
        assert "I am an AI assistant" in persona["full_text"]

    async def test_update_overwrites_section(self) -> None:
        from app.tools.persona import get_persona, update_persona

        await update_persona("identity", "Version 1")
        result = await update_persona("identity", "Version 2")
        assert result["action"] == "updated"

        persona = await get_persona()
        assert len(persona["sections"]) == 1
        assert "Version 2" in persona["full_text"]

    async def test_multiple_sections_ordered(self) -> None:
        from app.tools.persona import get_persona, update_persona

        await update_persona("02-profile", "Profile content")
        await update_persona("01-identity", "Identity content")

        persona = await get_persona()
        assert len(persona["sections"]) == 2
        assert persona["sections"][0]["key"] == "01-identity"
        assert persona["sections"][1]["key"] == "02-profile"

    async def test_delete_section(self) -> None:
        from app.tools.persona import delete_persona_section, get_persona, update_persona

        await update_persona("temp", "Temporary content")
        result = await delete_persona_section("temp")
        assert result["action"] == "deleted"

        persona = await get_persona()
        assert len(persona["sections"]) == 0

    async def test_delete_nonexistent(self) -> None:
        from app.tools.persona import delete_persona_section

        result = await delete_persona_section("nope")
        assert result["action"] == "not_found"

    async def test_update_empty_key_rejected(self) -> None:
        from app.tools.persona import update_persona

        result = await update_persona("", "content")
        assert "error" in result

    async def test_update_empty_content_rejected(self) -> None:
        from app.tools.persona import update_persona

        result = await update_persona("key", "")
        assert "error" in result
