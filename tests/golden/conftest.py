"""Golden test conftest — uses the local dev DB, not testcontainers.

Overrides the autouse _setup_db and _run_migrations fixtures from the
parent conftest so golden tests run against pre-populated data without
truncation or testcontainer setup.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from app.config import get_settings, reset_settings
from app.db import close_pool, create_pool


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "golden: mark test as golden dataset (requires --run-golden)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption("--run-golden", default=False):
        skip = pytest.mark.skip(reason="need --run-golden option to run")
        for item in items:
            if "golden" in item.keywords:
                item.add_marker(skip)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-golden", action="store_true", default=False, help="run golden dataset tests")


@pytest.fixture(autouse=True)
async def _setup_db() -> AsyncIterator[None]:
    """Connect to the dev DB configured via environment. No truncation."""
    reset_settings()
    get_settings()
    await create_pool()
    try:
        yield
    finally:
        await close_pool()
        reset_settings()


@pytest.fixture(scope="session")
async def _run_migrations() -> AsyncIterator[None]:
    """No-op — golden tests use a pre-populated DB."""
    return
