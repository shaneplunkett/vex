"""Database connection pool and migration management.

Connection pattern: acquire per tool call, never hold across calls.

    async with get_pool().acquire() as conn:
        result = await conn.fetch("SELECT ...")
"""

import json
from pathlib import Path

import asyncpg
import asyncpg.exceptions
import structlog

from app.config import get_settings

logger = structlog.get_logger()

_pool: asyncpg.Pool | None = None

MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations"


async def create_pool() -> asyncpg.Pool:
    """Create the asyncpg connection pool. Called from lifespan startup."""
    global _pool  # noqa: PLW0603
    settings = get_settings()

    async def _init_connection(conn: asyncpg.Connection) -> None:
        """Register JSON codec so JSONB columns return dicts, not strings."""
        await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")
        await conn.set_type_codec("json", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")

    _pool = await asyncpg.create_pool(
        dsn=settings.database_url,
        min_size=settings.db_pool_min,
        max_size=settings.db_pool_max,
        init=_init_connection,
    )
    logger.info(
        "db.pool_created",
        min_size=settings.db_pool_min,
        max_size=settings.db_pool_max,
    )
    return _pool


async def close_pool() -> None:
    """Close the asyncpg connection pool. Called from lifespan shutdown."""
    global _pool  # noqa: PLW0603
    if _pool is not None:
        await _pool.close()
        logger.info("db.pool_closed")
        _pool = None


def get_pool() -> asyncpg.Pool:
    """Get the current connection pool. Raises if not initialised."""
    if _pool is None:
        msg = "Database pool not initialised — call create_pool() first"
        raise RuntimeError(msg)
    return _pool


async def run_migrations() -> int:
    """Apply any unapplied migrations from the migrations/ directory.

    Migrations run in explicit transactions — partial applies roll back cleanly.
    If a migration was previously applied outside the runner (e.g. via psql),
    it's detected via DuplicateObject errors and marked as applied.

    Returns the number of migrations applied.
    """
    pool = get_pool()
    applied = 0

    async with pool.acquire() as conn:
        # Bootstrap: ensure applied_migrations table exists.
        # This is also in 001_initial_schema.sql (kept in sync) but we need it
        # before we can check what's been applied.
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS applied_migrations (
                id serial PRIMARY KEY,
                filename text NOT NULL UNIQUE,
                applied_at timestamptz DEFAULT now()
            )
        """)

        # Get already-applied migrations
        rows = await conn.fetch("SELECT filename FROM applied_migrations")
        already_applied = {row["filename"] for row in rows}

        # Find and sort migration files
        migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))

        for migration_file in migration_files:
            if migration_file.name in already_applied:
                continue

            logger.info("db.applying_migration", filename=migration_file.name)
            sql = migration_file.read_text()

            try:
                # Run the migration in a transaction — partial failure rolls back
                async with conn.transaction():
                    await conn.execute(sql)
                # Track as applied (outside the migration transaction)
                await conn.execute(
                    "INSERT INTO applied_migrations (filename) VALUES ($1)",
                    migration_file.name,
                )
                logger.info("db.migration_applied", filename=migration_file.name)
                applied += 1
            except (
                asyncpg.exceptions.DuplicateTableError,
                asyncpg.exceptions.DuplicateObjectError,
            ):
                # Migration was previously applied outside the runner (e.g. via psql).
                # Mark it as applied so we don't retry.
                await conn.execute(
                    "INSERT INTO applied_migrations (filename) VALUES ($1) ON CONFLICT DO NOTHING",
                    migration_file.name,
                )
                logger.info("db.migration_already_applied", filename=migration_file.name)

    if applied > 0:
        logger.info("db.migrations_complete", applied=applied)
    else:
        logger.info("db.migrations_up_to_date")

    return applied
