# Vex

Knowledge management MCP server — ingest conversations, extract entities and relationships, and search across your organisational memory using hybrid semantic + keyword search.

## What It Does

Vex ingests conversation data (AI chat logs, meeting transcripts, Slack threads) and builds a searchable knowledge graph on top of it:

- **Conversation ingestion** — parse and store conversations from multiple sources
- **Intelligent chunking** — break conversations into meaningful segments with topic boundary detection
- **Entity extraction** — automatically extract people, tools, decisions, and relationships
- **Hybrid search** — combine semantic (embedding) and keyword (full-text) search via reciprocal rank fusion
- **Knowledge graph** — entities, relations, and temporal tracking with supersession support
- **Coreference resolution** — resolve pronouns to named entities for better search quality

## Architecture

- **Python 3.12** with FastMCP for the MCP server interface
- **PostgreSQL + pgvector** for storage, embeddings, and full-text search
- **OpenAI** for embedding generation (text-embedding-3-small)
- **Anthropic Claude** for entity extraction
- **Docker Compose** for deployment

## Quick Start

```bash
# Local development (requires Nix)
nix develop
uv sync
pytest

# Docker deployment
docker compose up -d
```

## Configuration

All settings via environment variables with `VEX_BRAIN_` prefix. See `app/config.py` for full list.

Key settings:
- `VEX_BRAIN_DATABASE_URL` — Postgres connection string
- `VEX_BRAIN_OPENAI_API_KEY` — for embedding generation
- `VEX_BRAIN_ANTHROPIC_API_KEY` — for entity extraction
- `VEX_BRAIN_HUMAN_SPEAKER` — name for the human speaker (default: "User")
- `VEX_BRAIN_ASSISTANT_SPEAKER` — name for the assistant speaker (default: "Assistant")
