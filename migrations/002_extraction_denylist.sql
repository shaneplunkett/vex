-- Extraction denylist — prevents dismissed entities from re-entering review queue.
-- Checked by the validator before creating review items.

CREATE TABLE IF NOT EXISTS extraction_denylist (
    name text NOT NULL,
    entity_type text NOT NULL,
    reason text,
    created_at timestamptz DEFAULT now(),
    PRIMARY KEY (name, entity_type)
);

-- Backfill from existing dismissed review items
INSERT INTO extraction_denylist (name, entity_type, reason)
SELECT DISTINCT
    lower(trim(proposed->>'name')),
    proposed->>'entity_type',
    'Backfilled from dismissed review queue items'
FROM review_queue
WHERE status = 'dismissed'
  AND proposed->>'name' IS NOT NULL
  AND proposed->>'entity_type' IS NOT NULL
ON CONFLICT DO NOTHING;
