-- Store connected accounts.
--
-- Replaces app passwords in .env. Kivi holds a scoped token the user granted
-- on the provider's own consent screen, and the user can revoke it there.

CREATE TABLE IF NOT EXISTS oauth_tokens (
    provider TEXT PRIMARY KEY,            -- 'google' | 'notion' | 'slack'
    access_token TEXT NOT NULL,
    refresh_token TEXT,                   -- null where the provider issues none
    token_type TEXT,
    scopes TEXT,
    expires_at TIMESTAMP,                 -- null means it does not expire
    account_label TEXT,                   -- email or workspace, for display
    connected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
