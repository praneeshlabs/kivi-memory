-- ============================================================
-- TABLE: takes
-- Diagram: "Input — a finished take"
-- Raw dictation before any memory processing happens
-- ============================================================
CREATE TABLE takes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_text TEXT NOT NULL,
    formatted_text TEXT,
    app_context TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================
-- TABLE: watchlist
-- Diagram: Step 1 Filter -> "Watch" branch
-- Holds observations that are not yet worth saving.
-- If the same content is seen again, it gets promoted to memories.
-- ============================================================
CREATE TABLE watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    source_take_id INTEGER NOT NULL REFERENCES takes(id),
    embedding BLOB,                      -- used to detect repeats
    tag TEXT CHECK(tag IN ('fact', 'episode', 'preference')), -- persisted so promotion doesn't depend on a fresh LLM call
    times_seen INTEGER DEFAULT 1,
    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    promote_threshold INTEGER DEFAULT 2, -- times_seen needed to promote to memories
    status TEXT DEFAULT 'watching'
        CHECK(status IN ('watching', 'promoted', 'expired'))
);


-- ============================================================
-- TABLE: ignored
-- Diagram: Step 1 Filter -> "Drop" branch -> "Log to ignored ledger"
-- Records what Kivi deliberately chose not to remember, and why.
-- ============================================================
CREATE TABLE ignored (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_take_id INTEGER NOT NULL REFERENCES takes(id),
    content TEXT NOT NULL,
    reason TEXT NOT NULL,   -- e.g. "not about user", "unlikely to matter", "unsafe to surface"
    failed_test TEXT,       -- 'about_user' | 'likely_to_matter' | 'safe_to_surface'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================
-- TABLE: memories
-- Diagram: Step 2 Classify -> Step 3 Link -> Step 4 Save -> Step 5 Show
-- The core memory store. Every row is one remembered thing.
-- ============================================================
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Step 4: Save
    content TEXT NOT NULL,
    source_take_id INTEGER NOT NULL REFERENCES takes(id),  -- provenance
    promoted_from_watchlist_id INTEGER REFERENCES watchlist(id), -- nullable, set if it came from repetition

    -- Step 2: Classify
    tag TEXT NOT NULL CHECK(tag IN ('fact', 'episode', 'preference')),
    confidence REAL NOT NULL DEFAULT 1.0,   -- 0.0 to 1.0

    -- Step 3: Link
    related_memory_ids TEXT DEFAULT '[]',   -- JSON array of memory ids

    -- retrieval
    embedding BLOB NOT NULL,

    -- Step 4: timestamp
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Step 9: Retire (lifecycle tracking)
    last_used_at TIMESTAMP,
    usage_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'active'
        CHECK(status IN ('active', 'archived')),
    retired_reason TEXT,          -- 'contradicted' | 'refined' | 'unused' | 'user_deleted'
    retired_at TIMESTAMP,
    contradicted_by_memory_id INTEGER REFERENCES memories(id), -- if replaced by a newer fact

    -- Step 5: Show to user (edit/delete tracking)
    user_edited BOOLEAN DEFAULT 0,
    user_edited_at TIMESTAMP,
    original_content TEXT         -- preserved if user_edited = 1, for audit
);


-- ============================================================
-- TABLE: answers
-- Diagram: Step 6 Retrieve -> Step 7 Decide -> Step 8 Answer
-- Every Hey Kivi response, with full traceability.
-- ============================================================
CREATE TABLE answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    query TEXT NOT NULL,

    -- Step 6: Retrieve
    retrieved_memory_ids TEXT DEFAULT '[]',   -- JSON array, all candidates returned
    retrieval_score REAL,                      -- top similarity score

    -- Step 7: Decide
    decision TEXT NOT NULL CHECK(decision IN ('answered', 'declined_unknown')),
    decision_reason TEXT,   -- e.g. "retrieval below threshold 0.6"

    -- Step 8: Answer
    answer_text TEXT,
    cited_memory_ids TEXT DEFAULT '[]',   -- JSON array, memories actually used in the answer
    cited_take_ids TEXT DEFAULT '[]',     -- JSON array, source takes behind those memories

    -- operational (evaluation point: latency, cost)
    retrieval_latency_ms INTEGER,
    generation_latency_ms INTEGER,
    total_latency_ms INTEGER,
    model_used TEXT,
    tokens_used INTEGER,

    -- where the answer actually came from, so provenance is never ambiguous
    answer_source TEXT,      -- 'memory' | 'general' | 'mixed' | 'none'
    follow_ups TEXT,         -- JSON array of questions Kivi asked back

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================
-- TABLE: memory_archive
-- Diagram: Step 9 Retire -> "move to archive"
-- Retired memories are moved here, not deleted, for auditability.
-- ============================================================
CREATE TABLE memory_archive (
    id INTEGER PRIMARY KEY,               -- same id as original memories.id
    content TEXT NOT NULL,
    source_take_id INTEGER NOT NULL,
    tag TEXT NOT NULL,
    confidence REAL,
    related_memory_ids TEXT,
    created_at TIMESTAMP,
    archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    retired_reason TEXT NOT NULL,
    contradicted_by_memory_id INTEGER
);


-- ============================================================
-- INDEXES for retrieval performance
-- ============================================================
CREATE INDEX idx_memories_status ON memories(status);
CREATE INDEX idx_memories_tag ON memories(tag);
CREATE INDEX idx_memories_source_take ON memories(source_take_id);
CREATE INDEX idx_watchlist_status ON watchlist(status);
CREATE INDEX idx_answers_decision ON answers(decision);


-- ============================================================
-- TABLE: oauth_tokens
-- Connected accounts, authorised by the user through each provider's own
-- consent screen. Kivi never sees or stores an account password: it holds a
-- scoped, revocable token, and the user can withdraw it at the provider.
-- ============================================================
CREATE TABLE IF NOT EXISTS oauth_tokens (
    provider TEXT PRIMARY KEY,            -- 'google' | 'notion' | 'slack'
    access_token TEXT NOT NULL,
    refresh_token TEXT,                   -- null where the provider issues no refresh token
    token_type TEXT,
    scopes TEXT,
    expires_at TIMESTAMP,                 -- null means it does not expire
    account_label TEXT,                   -- email / workspace name, for display
    connected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
