"""SQLite connection handling and schema bootstrap for Kivi."""
import os
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = os.getenv("DATABASE_PATH", "kivi.db")
SCHEMA_PATH = BASE_DIR / "schema.sql"


def get_db_path() -> Path:
    db_path = Path(DATABASE_PATH)
    if not db_path.is_absolute():
        db_path = BASE_DIR / db_path
    return db_path


def get_connection() -> sqlite3.Connection:
    """Create a new SQLite connection with row access by column name."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# Columns added after the original schema shipped. Kept here rather than in a
# migration framework: SQLite can add a nullable column in place, and doing it
# on startup means an existing kivi.db keeps working without being rebuilt.
ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("watchlist", "tag", "TEXT"),
    ("answers", "answer_source", "TEXT"),
    ("answers", "follow_ups", "TEXT"),
)


# Tables added after the original schema shipped. Created on startup so an
# existing kivi.db gains them without being rebuilt.
ADDED_TABLES: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS oauth_tokens (
        provider TEXT PRIMARY KEY,
        access_token TEXT NOT NULL,
        refresh_token TEXT,
        token_type TEXT,
        scopes TEXT,
        expires_at TIMESTAMP,
        account_label TEXT,
        connected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
)


def _apply_added_tables(conn: sqlite3.Connection) -> None:
    for statement in ADDED_TABLES:
        conn.execute(statement)


def _apply_added_columns(conn: sqlite3.Connection) -> None:
    for table, column, column_type in ADDED_COLUMNS:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


def init_db() -> None:
    """Run schema.sql if the schema is absent, then top up any newer columns."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='takes'"
        )
        already_initialized = cursor.fetchone() is not None
        if not already_initialized:
            schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
            conn.executescript(schema_sql)
        _apply_added_tables(conn)
        _apply_added_columns(conn)
        conn.commit()
    finally:
        conn.close()
