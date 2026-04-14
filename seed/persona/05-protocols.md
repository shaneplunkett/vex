# Operational Protocols

## Data Sources

Vex ingests conversations from multiple sources:
- Claude Code session logs (CC JSONL format)
- Claude.ai conversation exports
- Additional sources can be added via custom importers

## Pipeline

Conversations flow through: import → chunking → embedding → extraction → complete.
Each stage is tracked via pipeline_status on the conversation record.

## Entity Management

- Entities are extracted automatically during pipeline processing
- Uncertain extractions go to the review queue for human approval
- The denylist prevents known false positives from being re-extracted
- Duplicate entities can be merged via the merge_entities tool
