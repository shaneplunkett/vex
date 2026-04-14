"""Embedding storage — embeds chunks and stores vectors in pgvector."""

from __future__ import annotations

import structlog

from app.config import get_settings
from app.db import get_pool
from app.pipeline.embedder import embed_texts

logger = structlog.get_logger()


async def embed_conversation_chunks(conversation_id: int) -> int:
    """Embed all chunks for a conversation and store vectors.

    Chunks that fail embedding are skipped — they remain queryable
    via tsvector but won't appear in semantic search.

    Updates pipeline_status to 'embedded' on success.
    Returns the number of chunks successfully embedded.
    """
    pool = get_pool()

    async with pool.acquire() as conn:
        chunks = await conn.fetch(
            """
            SELECT id, raw_content FROM chunks
            WHERE conversation_id = $1
            ORDER BY start_ordinal
            """,
            conversation_id,
        )

        if not chunks:
            logger.warning("embed_store.no_chunks", conversation_id=conversation_id)
            await conn.execute(
                """
                UPDATE conversations
                SET pipeline_status = 'failed', pipeline_error = 'no chunks to embed'
                WHERE id = $1
                """,
                conversation_id,
            )
            return 0

        texts = [chunk["raw_content"] for chunk in chunks]
        chunk_ids = [chunk["id"] for chunk in chunks]

        vectors = await embed_texts(texts)

        embedded = 0
        model_version = get_settings().embedding_model
        async with conn.transaction():
            for chunk_id, vector in zip(chunk_ids, vectors):
                if vector is not None:
                    await conn.execute(
                        """
                        UPDATE chunks
                        SET embedding = $1::vector, extraction_model_version = $2
                        WHERE id = $3
                        """,
                        str(vector),
                        model_version,
                        chunk_id,
                    )
                    embedded += 1
                else:
                    logger.warning(
                        "embed_store.chunk_failed",
                        conversation_id=conversation_id,
                        chunk_id=chunk_id,
                    )

            if embedded > 0:
                await conn.execute(
                    "UPDATE conversations SET pipeline_status = 'embedded' WHERE id = $1",
                    conversation_id,
                )
            else:
                await conn.execute(
                    """
                    UPDATE conversations
                    SET pipeline_status = 'failed', pipeline_error = 'all embeddings failed'
                    WHERE id = $1
                    """,
                    conversation_id,
                )

        logger.info(
            "embed_store.complete",
            conversation_id=conversation_id,
            chunks_total=len(chunks),
            chunks_embedded=embedded,
            chunks_failed=len(chunks) - embedded,
        )
        return embedded
