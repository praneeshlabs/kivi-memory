-- Keyword index over memory content.
--
-- Embeddings are weak on names. "Pixel", "Meera" and "Zoho" carry little
-- distributional meaning, so a vector search can rank the wrong pet first.
-- BM25 keys on the literal token. Both are used and the rankings are fused.
--
-- The index is rebuilt from the memories table at startup, so it is safe if
-- this table is ever dropped or falls behind.

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
USING fts5(content, memory_id UNINDEXED, tokenize='unicode61');
