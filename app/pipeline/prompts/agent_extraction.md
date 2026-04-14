# Agent Extraction Prompt

You are extracting structured knowledge from conversation chunks between a User and an Assistant.

## Your Task

Read the chunks file provided (one JSON object per line, each with id, raw_content, chunk_type, significance). For each chunk, extract entities and relations.

## Entity Types (use EXACTLY these)

Person, HealthCondition, Medication, PsychologicalPattern, Preference, Tool, Skill, Infrastructure, Project, Organisation, Place, Identity, Routine, Media, InteractionPattern, Milestone

## Relation Types (use EXACTLY these, source → target)

has_condition (Person→HealthCondition), treated_by (HealthCondition→Medication), prescribed_by (Medication→Person), triggers (Cause→Effect), caused_by (Effect→Cause), connected_to (Either), works_at (Person→Org, exclusive), member_of (Person→Org, exclusive), uses (Person/Project→Tool), built_on (Project→Tool/Infra), part_of (Component→System), prefers (Person→Preference), avoids (Person→Thing), lives_in (Person→Place, exclusive), knows (Person→Person), treats (Provider→Patient), identifies_as (Person→Identity), follows (Person→Routine), manages (HealthCondition→Routine/Treatment), relates_to (Pattern→HealthCondition), supersedes (New→Old)

## Rules

1. "I"/"me" in human turns = the configured human speaker name. "I"/"me" in assistant turns = the configured assistant speaker name.
2. Only extract entities with ONGOING relevance — not passing mentions.
3. Financial items are NOT entities.
4. For each entity: name, entity_type, match (existing/new/uncertain), confidence (0.0-1.0), brief reasoning.
5. For each relation: source, target, relation_type, description.
6. Use proper noun casing. Use whichever name form is most commonly referenced.
7. Singleton entities (configured primary speakers) — always use their exact names, never extract them as new entities.
8. Higher significance chunks (4-5) deserve more careful extraction. Low significance (1-2) — only genuinely meaningful entities.

## Output

Write results as a single JSON file with this structure:

```json
{
  "conversation_id": <ID>,
  "chunks": [
    {
      "chunk_id": <id from the chunk>,
      "entities": [
        {"name": "...", "entity_type": "...", "match": "existing|new|uncertain", "confidence": 0.9, "reasoning": "..."}
      ],
      "relations": [
        {"source": "...", "target": "...", "relation_type": "...", "description": "..."}
      ],
      "flags": []
    }
  ]
}
```

Do NOT use any external APIs. You ARE the extraction model. Read the chunks, process them, write the output file.
