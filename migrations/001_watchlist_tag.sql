-- Persist the tag on watchlist rows.
--
-- Promotion used to read the tag from whichever LLM response happened to be in
-- flight. If a watched item was promoted by a later, differently worded
-- mention, the tag could change or go missing. Storing it at first sight makes
-- promotion independent of a fresh model call.

ALTER TABLE watchlist ADD COLUMN tag TEXT;
