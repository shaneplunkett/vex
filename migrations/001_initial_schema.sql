-- Vex — Initial Schema
-- All three layers (conversations, chunks, knowledge) + operations tables
-- All DDL uses IF NOT EXISTS for idempotent re-runs.

-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Migration tracking (also bootstrapped in app/db.py — keep in sync)
CREATE TABLE IF NOT EXISTS applied_migrations (
    id serial PRIMARY KEY,
    filename text NOT NULL UNIQUE,
    applied_at timestamptz DEFAULT now()
);

-----------------------------------------------------------
-- LAYER 1: Conversations
-----------------------------------------------------------

CREATE TABLE IF NOT EXISTS conversations (
    id serial PRIMARY KEY,
    source text NOT NULL CHECK (source IN ('cc', 'claude_ai')),
    session_id text UNIQUE,
    name text,
    started_at timestamptz,
    ended_at timestamptz,
    message_count int DEFAULT 0,
    pipeline_status text DEFAULT 'pending'
        CHECK (pipeline_status IN ('pending', 'chunked', 'embedded', 'extracted', 'complete', 'failed')),
    pipeline_error text,
    created_at timestamptz DEFAULT now()
);

-- Note: messages.timestamp defaults to now() for convenience, but importers
-- MUST provide the actual message timestamp from source data. Using the default
-- for historical imports would break recall() recency scoring.
CREATE TABLE IF NOT EXISTS messages (
    id serial PRIMARY KEY,
    conversation_id int NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role text NOT NULL CHECK (role IN ('human', 'assistant')),
    content text NOT NULL,
    timestamp timestamptz NOT NULL DEFAULT now(),
    ordinal int NOT NULL,
    UNIQUE(conversation_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_fts ON messages USING gin(to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_conversations_source ON conversations(source);
CREATE INDEX IF NOT EXISTS idx_conversations_started ON conversations(started_at);
CREATE INDEX IF NOT EXISTS idx_conversations_pipeline_pending ON conversations(pipeline_status)
    WHERE pipeline_status NOT IN ('complete');

-----------------------------------------------------------
-- LAYER 2: Chunks
-----------------------------------------------------------

CREATE TABLE IF NOT EXISTS chunks (
    id serial PRIMARY KEY,
    conversation_id int NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    content text NOT NULL,
    raw_content text NOT NULL,
    start_message_id int REFERENCES messages(id) ON DELETE SET NULL,
    end_message_id int REFERENCES messages(id) ON DELETE SET NULL,
    start_ordinal int NOT NULL,
    end_ordinal int NOT NULL,
    embedding vector(1536),
    chunk_type text DEFAULT 'topic'
        CHECK (chunk_type IN ('topic', 'moment', 'decision', 'correction', 'emotional')),
    significance int DEFAULT 3
        CHECK (significance BETWEEN 1 AND 5),
    extraction_model_version text,
    last_accessed_at timestamptz,
    access_count int DEFAULT 0,
    created_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_conversation ON chunks(conversation_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_fts ON chunks USING gin(to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_chunks_significance ON chunks(significance);

-----------------------------------------------------------
-- LAYER 3: Knowledge Graph
-----------------------------------------------------------

CREATE TABLE IF NOT EXISTS entities (
    id serial PRIMARY KEY,
    name text NOT NULL,
    entity_type text NOT NULL,
    summary text,
    aliases text[] DEFAULT '{}',
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),
    last_accessed_at timestamptz,
    access_count int DEFAULT 0,
    UNIQUE(name, entity_type)
);

-- Relations use ON DELETE RESTRICT intentionally — entities involved in the
-- knowledge graph must not be silently removed. Delete relations first, then
-- the entity, to force explicit cleanup of graph edges.
CREATE TABLE IF NOT EXISTS relations (
    id serial PRIMARY KEY,
    source_id int NOT NULL REFERENCES entities(id) ON DELETE RESTRICT,
    target_id int NOT NULL REFERENCES entities(id) ON DELETE RESTRICT,
    relation_type text NOT NULL,
    description text,
    exclusive boolean DEFAULT false,
    valid_from timestamptz,
    created_at timestamptz DEFAULT now(),
    superseded_at timestamptz,
    superseded_by int REFERENCES relations(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS entity_chunks (
    entity_id int REFERENCES entities(id) ON DELETE CASCADE,
    chunk_id int REFERENCES chunks(id) ON DELETE CASCADE,
    PRIMARY KEY (entity_id, chunk_id)
);

CREATE TABLE IF NOT EXISTS relation_chunks (
    relation_id int REFERENCES relations(id) ON DELETE CASCADE,
    chunk_id int REFERENCES chunks(id) ON DELETE CASCADE,
    PRIMARY KEY (relation_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_name_trgm ON entities USING gin(name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_entities_fts ON entities
    USING gin(to_tsvector('english', coalesce(name, '') || ' ' || coalesce(summary, '')));
CREATE INDEX IF NOT EXISTS idx_entities_recent ON entities(last_accessed_at DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id);
CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type);
CREATE INDEX IF NOT EXISTS idx_relations_active ON relations(superseded_at) WHERE superseded_at IS NULL;

-- Junction table reverse lookups (needed for ON DELETE CASCADE performance)
CREATE INDEX IF NOT EXISTS idx_entity_chunks_chunk ON entity_chunks(chunk_id);
CREATE INDEX IF NOT EXISTS idx_relation_chunks_chunk ON relation_chunks(chunk_id);

-----------------------------------------------------------
-- OPERATIONS
-----------------------------------------------------------

CREATE TABLE IF NOT EXISTS review_queue (
    id serial PRIMARY KEY,
    chunk_id int NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    proposed jsonb NOT NULL,
    candidates jsonb,
    reason text,
    status text DEFAULT 'pending'
        CHECK (status IN ('pending', 'resolved', 'dismissed')),
    resolved_entity_id int REFERENCES entities(id),
    rejected_candidates jsonb,
    resolution_notes text,
    created_at timestamptz DEFAULT now(),
    resolved_at timestamptz
);

CREATE TABLE IF NOT EXISTS audit_log (
    id serial PRIMARY KEY,
    audit_type text NOT NULL,
    findings jsonb NOT NULL,
    actions_taken jsonb,
    created_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS extraction_examples (
    id serial PRIMARY KEY,
    entity_name text NOT NULL,
    resolved_to_id int NOT NULL REFERENCES entities(id),
    context_snippet text,
    embedding vector(1536),
    created_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_review_queue_status ON review_queue(status);
CREATE INDEX IF NOT EXISTS idx_review_queue_pending ON review_queue(created_at) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_extraction_examples_embedding
    ON extraction_examples USING hnsw (embedding vector_cosine_ops);

-----------------------------------------------------------
-- SINGLETON ENTITIES
-----------------------------------------------------------

-- Singleton entities — primary speakers. Override names via VEX_BRAIN_HUMAN_SPEAKER
-- and VEX_BRAIN_ASSISTANT_SPEAKER environment variables. These defaults match the
-- config defaults. If you change speaker names, update these seeds accordingly.
INSERT INTO entities (id, name, entity_type, summary)
VALUES
    (1, 'User', 'Person', 'Primary human user'),
    (2, 'Assistant', 'Person', 'AI assistant')
ON CONFLICT (name, entity_type) DO NOTHING;

-- Ensure the sequence is past the singletons
SELECT setval('entities_id_seq', GREATEST((SELECT MAX(id) FROM entities), 2));
