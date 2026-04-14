"""Entity/relation extraction — extracts structured knowledge from chunks via Claude Sonnet.

Uses forced tool_use for structured output (no extended thinking — testing
showed comparable quality at ~12x lower cost). Runs per-chunk for failure
isolation. Adjacent chunks provided for disambiguation context.

The extractor proposes entities and relations. The validator (#16) handles
matching against existing entities, deduplication, and confidence routing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from typing import Any

import structlog

from app.config import get_settings

logger = structlog.get_logger()

_MAX_RETRIES = 3
_BASE_DELAY = 2.0  # seconds

# ---------------------------------------------------------------------------
# Entity & Relation Type Schemas
# ---------------------------------------------------------------------------

ENTITY_TYPES = [
    "Person",
    "HealthCondition",
    "Medication",
    "PsychologicalPattern",
    "Preference",
    "Tool",
    "Skill",
    "Infrastructure",
    "Project",
    "Organisation",
    "Place",
    "Identity",
    "Routine",
    "Media",
    "InteractionPattern",
    "Milestone",
]

RELATION_TYPES = [
    "has_condition",
    "treated_by",
    "prescribed_by",
    "triggers",
    "caused_by",
    "connected_to",
    "works_at",
    "member_of",
    "uses",
    "built_on",
    "part_of",
    "prefers",
    "avoids",
    "lives_in",
    "knows",
    "treats",
    "identifies_as",
    "follows",
    "manages",
    "relates_to",
    "supersedes",
]

# ---------------------------------------------------------------------------
# Extraction Prompt
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """You are extracting structured knowledge from a conversation between a User and an Assistant.

## Your Task

Extract entities and relations from the CURRENT CHUNK below.
Use the adjacent chunks for disambiguation context only — do not extract from them.

## Entity Types (use EXACTLY these)

| Type | Description | NOT this |
|---|---|---|
| Person | Named individual with ongoing relevance | Job titles, abstract roles |
| HealthCondition | Diagnosed medical condition | Symptoms, medications |
| Medication | Prescribed or taken medication | Conditions, dosages |
| PsychologicalPattern | Recurring cognitive/emotional pattern | One-off emotional events |
| Preference | Enduring preference or taste | Transient opinions |
| Tool | Software tool or technology actively used | Languages (use Skill) |
| Skill | Language, framework, or technical competency | Tools (the software itself) |
| Infrastructure | System, service, or architectural component | Abstract concepts |
| Project | Active initiative with defined scope | Books, games, routines |
| Organisation | Company, team, or institution | Departments |
| Place | Physical or virtual location with significance | Generic locations |
| Identity | Aspect of self-concept or role | Personality traits |
| Routine | Recurring routine or protocol | One-off events |
| Media | Book, game, show, podcast with significance | Passing mentions |
| InteractionPattern | Relationship procedural knowledge | General preferences |
| Milestone | Named significant event referenced across conversations | Generic one-off events |

## Relation Types (use EXACTLY these, with directionality)

| Relation | Direction | Example |
|---|---|---|
| has_condition | Person → Condition | Alice → Diabetes |
| treated_by | Condition → Medication | Diabetes → Metformin |
| prescribed_by | Medication → Person | Metformin → Dr Smith |
| triggers | Cause → Effect | Stress → insomnia |
| caused_by | Effect → Cause | Bug → race condition |
| connected_to | Either direction | Pattern → outcome |
| works_at | Person → Organisation (exclusive) | Alice → Acme Corp |
| member_of | Person → Organisation (exclusive) | Alice → Platform Team |
| uses | Person/Project → Tool | Alice → Neovim |
| built_on | Project → Tool/Infra | Vex → Postgres |
| part_of | Component → System | API Gateway → platform |
| prefers | Person → Preference | Alice → dark mode |
| avoids | Person → Thing avoided | Alice → IDEs |
| lives_in | Person → Place (exclusive) | Alice → Melbourne |
| knows | Person → Person | Alice → Bob |
| treats | Provider → Patient | Dr Smith → Alice |
| identifies_as | Person → Identity | Alice → engineer |
| follows | Person → Routine | Alice → daily standup |
| manages | Condition → Routine/Treatment | Diabetes → diet plan |
| relates_to | Pattern → Condition/Pattern | Burnout → overwork |
| supersedes | New → Old | (temporal replacement) |

"Exclusive" means only one current relation of this type per source entity.

## Rules

1. The configured primary speakers are singletons — always use their exact names. "I"/"me" in human turns = human speaker. "I"/"me" in assistant turns = assistant speaker.
2. Only extract entities with ONGOING relevance — not passing mentions.
3. Financial items (credit cards, bank accounts, insurance) are NOT entities.
4. For each entity, indicate whether you believe it matches an EXISTING well-known entity or is NEW.
5. Include a confidence score (0.0–1.0) and brief reasoning for each entity.
6. If a relation type doesn't fit any of the 21 types, add it to the flags array instead of inventing a type.
7. Extract relations between entities mentioned in THIS chunk only.
8. Use proper noun casing for entity names. Use full official names for conditions
   and medications — use whichever form is most commonly referenced.
9. If something was true in the past but isn't now (e.g. "I used to use X"),
   note this in the relation description or add a flag.

## Chunk Significance

The chunk is rated {significance}/5 for significance.
Higher significance (4-5) = decisions, corrections, emotional moments, or milestones.
Lower significance (1-2) = routine — only extract genuinely meaningful entities.

## Context

{context}

## CURRENT CHUNK (extract from this)

Type: {chunk_type}

{chunk_content}
"""

# ---------------------------------------------------------------------------
# Tool Schema for Structured Output
# ---------------------------------------------------------------------------

_EXTRACTION_TOOL = {
    "name": "record_extraction",
    "description": "Record extracted entities and relations from a conversation chunk.",
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Entity name as it should be stored (canonical form).",
                        },
                        "entity_type": {
                            "type": "string",
                            "enum": ENTITY_TYPES,
                            "description": "One of the 16 allowed entity types.",
                        },
                        "summary": {
                            "type": "string",
                            "description": "Brief description of the entity based on this chunk's context.",
                        },
                        "match": {
                            "type": "string",
                            "enum": ["existing", "new", "uncertain"],
                            "description": "Whether this likely matches an existing entity or is new.",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence in the extraction (0.0–1.0).",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": (
                                "Brief explanation of why this entity was extracted and the match classification."
                            ),
                        },
                    },
                    "required": ["name", "entity_type", "summary", "match", "confidence", "reasoning"],
                },
            },
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Source entity name (must match an extracted entity or a singleton speaker).",
                        },
                        "target": {
                            "type": "string",
                            "description": "Target entity name (must match an extracted entity or a singleton speaker).",
                        },
                        "relation_type": {
                            "type": "string",
                            "enum": RELATION_TYPES,
                            "description": "One of the 21 allowed relation types.",
                        },
                        "description": {
                            "type": "string",
                            "description": "Brief description of this specific relationship.",
                        },
                    },
                    "required": ["source", "target", "relation_type", "description"],
                },
            },
            "flags": {
                "type": "array",
                "items": {
                    "type": "string",
                },
                "description": "Edge cases, unmatched relations, or things that need human review.",
            },
        },
        "required": ["entities", "relations", "flags"],
    },
}


def _get_prompt_hash() -> str:
    """Hash the extraction prompt + tool schema for versioning."""
    content = _EXTRACTION_PROMPT + json.dumps(_EXTRACTION_TOOL, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def get_model_version() -> str:
    """Return a version string combining model ID and prompt hash."""
    settings = get_settings()
    return f"{settings.extraction_model}:{_get_prompt_hash()}"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


_client: Any = None


def _get_client() -> Any:
    """Get or create a singleton async Anthropic client."""
    global _client  # noqa: PLW0603
    if _client is None:
        import anthropic

        settings = get_settings()
        api_key = settings.anthropic_api_key
        if api_key is None:
            msg = "VEX_BRAIN_ANTHROPIC_API_KEY is required for entity extraction"
            raise RuntimeError(msg)

        _client = anthropic.AsyncAnthropic(api_key=api_key.get_secret_value(), timeout=120.0)
    return _client


# ---------------------------------------------------------------------------
# Context Building
# ---------------------------------------------------------------------------


def _build_context(
    prev_chunk: dict[str, Any] | None,
    next_chunk: dict[str, Any] | None,
) -> str:
    """Build the disambiguation context from adjacent chunks."""
    parts = []

    if prev_chunk:
        parts.append(f"### Previous chunk\n{prev_chunk['raw_content']}")

    if next_chunk:
        parts.append(f"### Next chunk\n{next_chunk['raw_content']}")

    if not parts:
        return "(No adjacent chunks available — this may be a single-chunk conversation.)"

    return "\n\n".join(parts)


def _build_prompt(
    chunk: dict[str, Any],
    prev_chunk: dict[str, Any] | None,
    next_chunk: dict[str, Any] | None,
) -> str:
    """Build the full extraction prompt for a chunk."""
    context = _build_context(prev_chunk, next_chunk)

    return _EXTRACTION_PROMPT.format(
        context=context,
        chunk_type=chunk.get("chunk_type", "topic"),
        significance=chunk.get("significance", 3),
        chunk_content=chunk["raw_content"],
    )


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


async def extract_chunk(
    chunk: dict[str, Any],
    prev_chunk: dict[str, Any] | None = None,
    next_chunk: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Extract entities and relations from a single chunk.

    Uses Claude Sonnet with forced tool_use for structured output.

    Args:
        chunk: Dict with at least 'raw_content', 'chunk_type', 'significance'.
        prev_chunk: Adjacent chunk before this one (for context).
        next_chunk: Adjacent chunk after this one (for context).

    Returns:
        Dict with 'entities', 'relations', 'flags', and 'model_version',
        or None if extraction failed after retries.
    """
    prompt = _build_prompt(chunk, prev_chunk, next_chunk)
    client = _get_client()
    settings = get_settings()
    model_version = get_model_version()

    last_error: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            start = time.monotonic()

            response = await client.messages.create(
                model=settings.extraction_model,
                max_tokens=8000,
                temperature=0.0,
                tools=[_EXTRACTION_TOOL],
                tool_choice={"type": "tool", "name": "record_extraction"},
                messages=[{"role": "user", "content": prompt}],
            )

            elapsed = time.monotonic() - start

            # Extract the tool_use result from the response
            result = _parse_response(response)
            if result is None:
                logger.warning(
                    "extractor.no_tool_use_in_response",
                    chunk_id=chunk.get("id"),
                    attempt=attempt + 1,
                )
                continue

            result["model_version"] = model_version

            logger.info(
                "extractor.chunk_complete",
                chunk_id=chunk.get("id"),
                entities=len(result.get("entities", [])),
                relations=len(result.get("relations", [])),
                flags=len(result.get("flags", [])),
                latency_ms=round(elapsed * 1000),
            )

            return result

        except Exception as e:
            last_error = e
            if _is_retryable(e) and attempt < _MAX_RETRIES - 1:
                delay = _BASE_DELAY * (2**attempt) + random.uniform(0, 1.0)
                logger.warning(
                    "extractor.retry",
                    chunk_id=chunk.get("id"),
                    attempt=attempt + 1,
                    delay_s=delay,
                    error=str(e),
                )
                await asyncio.sleep(delay)
            else:
                break

    logger.error(
        "extractor.chunk_failed",
        chunk_id=chunk.get("id"),
        retries=_MAX_RETRIES,
        error=str(last_error),
    )
    return None


def _parse_response(response: Any) -> dict[str, Any] | None:
    """Extract the tool_use input from the Anthropic response.

    Returns the extraction result dict or None if no tool_use block found.
    """
    for block in response.content:
        if block.type == "tool_use" and block.name == "record_extraction":
            result: dict[str, Any] = block.input
            return result
    return None


def _is_retryable(error: Exception) -> bool:
    """Check if an error is transient and worth retrying."""
    import anthropic

    if isinstance(error, anthropic.RateLimitError):
        return True
    if isinstance(error, anthropic.APIStatusError) and error.status_code >= 500:
        return True
    return isinstance(error, (anthropic.APITimeoutError, anthropic.APIConnectionError))


# ---------------------------------------------------------------------------
# Batch Extraction (per-conversation)
# ---------------------------------------------------------------------------


async def extract_conversation_chunks(conversation_id: int) -> dict[str, Any]:
    """Extract entities and relations from all chunks in a conversation.

    Runs extractions concurrently up to the configured extraction_concurrency limit.
    Per-chunk failure isolation — failed chunks are logged but don't block others.

    Args:
        conversation_id: ID of the conversation whose chunks to extract.

    Returns:
        Dict with counts: 'total', 'extracted', 'failed', and 'results' list.
    """
    from app.db import get_pool

    settings = get_settings()
    semaphore = asyncio.Semaphore(settings.extraction_concurrency)
    pool = get_pool()

    # Fetch all chunks for this conversation, ordered by ordinal
    async with pool.acquire() as conn:
        chunks = await conn.fetch(
            """
            SELECT id, raw_content, chunk_type, significance,
                   start_ordinal, end_ordinal
            FROM chunks
            WHERE conversation_id = $1
            ORDER BY start_ordinal
            """,
            conversation_id,
        )

    if not chunks:
        logger.warning("extractor.no_chunks", conversation_id=conversation_id)
        return {"total": 0, "extracted": 0, "failed": 0, "results": []}

    chunk_dicts = [dict(c) for c in chunks]

    async def _extract_with_semaphore(
        idx: int,
        chunk: dict[str, Any],
    ) -> tuple[int, dict[str, Any] | None]:
        prev_chunk = chunk_dicts[idx - 1] if idx > 0 else None
        next_chunk = chunk_dicts[idx + 1] if idx < len(chunk_dicts) - 1 else None

        async with semaphore:
            result = await extract_chunk(chunk, prev_chunk, next_chunk)
            return chunk["id"], result

    # Run all extractions concurrently (bounded by semaphore)
    tasks = [_extract_with_semaphore(i, c) for i, c in enumerate(chunk_dicts)]
    completed = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    extracted = 0
    failed = 0

    for item in completed:
        if isinstance(item, BaseException):
            failed += 1
            logger.error("extractor.task_exception", error=str(item))
            continue

        chunk_id, result = item  # type: ignore[misc]
        if result is not None:
            extracted += 1
            results.append({"chunk_id": chunk_id, **result})
        else:
            failed += 1

    # Update extraction_model_version on successfully extracted chunks
    if results:
        model_version = get_model_version()
        extracted_chunk_ids = [r["chunk_id"] for r in results]
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE chunks
                SET extraction_model_version = $1
                WHERE id = ANY($2)
                """,
                model_version,
                extracted_chunk_ids,
            )

    logger.info(
        "extractor.conversation_complete",
        conversation_id=conversation_id,
        total=len(chunk_dicts),
        extracted=extracted,
        failed=failed,
    )

    return {
        "total": len(chunk_dicts),
        "extracted": extracted,
        "failed": failed,
        "results": results,
    }
