"""Tests for topic boundary detection via embedding similarity."""

from __future__ import annotations

import pytest

from app.db import get_pool
from app.pipeline.topic_boundary import _cosine_similarity, detect_and_split_boundaries


class TestCosineSimilarity:
    """Tests for the cosine similarity helper."""

    def test_identical_vectors(self) -> None:
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_similar_vectors(self) -> None:
        a = [1.0, 1.0, 0.0]
        b = [1.0, 0.9, 0.1]
        sim = _cosine_similarity(a, b)
        assert 0.9 < sim < 1.0

    def test_string_format(self) -> None:
        """pgvector returns embeddings as strings like '[0.1,0.2,0.3]'."""
        a = "[1.0,0.0,0.0]"
        b = "[0.0,1.0,0.0]"
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector(self) -> None:
        a = [0.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert _cosine_similarity(a, b) == 0.0


class TestDetectBoundaries:
    """Tests for boundary detection with real DB chunks."""

    @pytest.fixture(autouse=True)
    async def _setup(self, _setup_db: None) -> None:
        """Ensure DB is ready."""

    async def test_no_chunks_returns_empty(self) -> None:
        pool = get_pool()
        async with pool.acquire() as conn:
            conv_id = await conn.fetchval(
                "INSERT INTO conversations (source, message_count) VALUES ('cc', 0) RETURNING id",
            )
        result = await detect_and_split_boundaries(conv_id)
        assert result["total_pairs"] == 0

    async def test_single_chunk_returns_empty(self) -> None:
        pool = get_pool()
        async with pool.acquire() as conn:
            conv_id = await conn.fetchval(
                "INSERT INTO conversations (source, message_count) VALUES ('cc', 1) RETURNING id",
            )
            # Insert one chunk with a small embedding
            embedding = "[" + ",".join(["0.1"] * 1536) + "]"
            await conn.execute(
                """
                INSERT INTO chunks (conversation_id, content, raw_content, start_ordinal, end_ordinal, embedding)
                VALUES ($1, 'test', 'test', 0, 0, $2::vector)
                """,
                conv_id,
                embedding,
            )
        result = await detect_and_split_boundaries(conv_id)
        assert result["total_pairs"] == 0

    async def test_similar_chunks_no_boundary(self) -> None:
        pool = get_pool()
        async with pool.acquire() as conn:
            conv_id = await conn.fetchval(
                "INSERT INTO conversations (source, message_count) VALUES ('cc', 2) RETURNING id",
            )
            # Two very similar embeddings
            emb = [0.1] * 1536
            emb_str = "[" + ",".join(str(x) for x in emb) + "]"
            for i in range(2):
                await conn.execute(
                    """
                    INSERT INTO chunks (conversation_id, content, raw_content, start_ordinal, end_ordinal, embedding)
                    VALUES ($1, $2, $2, $3, $3, $4::vector)
                    """,
                    conv_id,
                    f"chunk {i}",
                    i,
                    emb_str,
                )

        result = await detect_and_split_boundaries(conv_id)
        assert result["total_pairs"] == 1
        assert result["boundaries_found"] == 0
        assert result["avg_similarity"] > 0.9

    async def test_dissimilar_chunks_boundary_detected(self) -> None:
        pool = get_pool()
        async with pool.acquire() as conn:
            conv_id = await conn.fetchval(
                "INSERT INTO conversations (source, message_count) VALUES ('cc', 2) RETURNING id",
            )
            # Two orthogonal embeddings — clear topic shift
            emb_a = [1.0] + [0.0] * 1535
            emb_b = [0.0, 1.0] + [0.0] * 1534
            for i, emb in enumerate([emb_a, emb_b]):
                emb_str = "[" + ",".join(str(x) for x in emb) + "]"
                await conn.execute(
                    """
                    INSERT INTO chunks (conversation_id, content, raw_content,
                                       start_ordinal, end_ordinal, embedding, significance)
                    VALUES ($1, $2, $2, $3, $3, $4::vector, 3)
                    """,
                    conv_id,
                    f"chunk {i}",
                    i,
                    emb_str,
                )

        result = await detect_and_split_boundaries(conv_id)
        assert result["total_pairs"] == 1
        assert result["boundaries_found"] == 1

        # Verify significance was boosted on the boundary chunk
        async with pool.acquire() as conn:
            sig = await conn.fetchval(
                """
                SELECT significance FROM chunks
                WHERE conversation_id = $1 ORDER BY start_ordinal LIMIT 1 OFFSET 1
                """,
                conv_id,
            )
        assert sig == 4  # boosted from 3 to 4

    async def test_significance_capped_at_5(self) -> None:
        pool = get_pool()
        async with pool.acquire() as conn:
            conv_id = await conn.fetchval(
                "INSERT INTO conversations (source, message_count) VALUES ('cc', 2) RETURNING id",
            )
            emb_a = [1.0] + [0.0] * 1535
            emb_b = [0.0, 1.0] + [0.0] * 1534
            for i, emb in enumerate([emb_a, emb_b]):
                emb_str = "[" + ",".join(str(x) for x in emb) + "]"
                await conn.execute(
                    """
                    INSERT INTO chunks (conversation_id, content, raw_content,
                                       start_ordinal, end_ordinal, embedding, significance)
                    VALUES ($1, $2, $2, $3, $3, $4::vector, 5)
                    """,
                    conv_id,
                    f"chunk {i}",
                    i,
                    emb_str,
                )

        await detect_and_split_boundaries(conv_id)

        # Significance should stay at 5, not exceed it
        async with pool.acquire() as conn:
            sig = await conn.fetchval(
                """
                SELECT significance FROM chunks
                WHERE conversation_id = $1 ORDER BY start_ordinal LIMIT 1 OFFSET 1
                """,
                conv_id,
            )
        assert sig == 5
