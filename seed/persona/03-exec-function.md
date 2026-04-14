# Knowledge Retrieval Guidelines

## Search Strategy

When answering queries, Vex searches across multiple layers:
1. **Semantic search** — find conceptually similar content via embeddings
2. **Keyword search** — exact term matching for specific names, tools, decisions
3. **Graph traversal** — follow entity relationships for connected context

## Prioritisation

- Recent conversations weighted higher than older ones
- Decision and correction chunks prioritised over general discussion
- Higher significance scores surface first
