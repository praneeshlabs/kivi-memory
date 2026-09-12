-- Record where each answer came from.
--
-- Kivi answers from memory, from general knowledge, or from the web. Without
-- this column the answers log could not tell them apart, so there was no way
-- to audit whether a reply was grounded in the user's own data.

ALTER TABLE answers ADD COLUMN answer_source TEXT;
ALTER TABLE answers ADD COLUMN follow_ups TEXT;
