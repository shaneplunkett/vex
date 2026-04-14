-- Persona storage — serves the assistant persona definition via MCP tool.

CREATE TABLE IF NOT EXISTS persona (
    key text PRIMARY KEY,
    content text NOT NULL,
    updated_at timestamptz DEFAULT now()
);
