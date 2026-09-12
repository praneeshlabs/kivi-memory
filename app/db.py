"""SQLite connection handling, schema bootstrap and migrations."""
import os
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = os.getenv("DATABASE_PATH", "kivi.db")
SCHEMA_PATH = BASE_DIR / "schema.sql"
MIGRATIONS_DIR = BASE_DIR / "migrations"


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


def _ensure_migrations_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def applied_migrations(conn: sqlite3.Connection) -> set[str]:
    _ensure_migrations_table(conn)
    return {row["version"] for row in conn.execute("SELECT version FROM schema_migrations")}


def pending_migrations(conn: sqlite3.Connection) -> list[Path]:
    if not MIGRATIONS_DIR.exists():
        return []
    done = applied_migrations(conn)
    return [p for p in sorted(MIGRATIONS_DIR.glob("*.sql")) if p.stem not in done]


def _is_already_applied(conn: sqlite3.Connection, statement: str) -> bool:
    """True if this statement's change is already present.

    A fresh database is created from schema.sql, which already contains
    everything the migrations add. Re-running an ALTER there would fail, so
    each migration is checked against the live schema before it runs. This
    keeps one code path for both a new install and an upgrade.
    """
    lowered = " ".join(statement.lower().split())
    if not lowered.startswith("alter table"):
        return False
    parts = lowered.split()
    try:
        table = parts[2]
        column = parts[parts.index("column") + 1]
    except (IndexError, ValueError):
        return False
    existing = {row["name"].lower() for row in conn.execute(f"PRAGMA table_info({table})")}
    return column in existing


def _statements(sql: str) -> list[str]:
    """Split a migration file into executable statements.

    Comment lines are stripped before splitting, not after. Every migration
    here opens with an explanatory comment block, and a naive "skip chunks
    starting with --" check silently discarded the statement attached to it.
    """
    cleaned_lines = [
        line for line in sql.splitlines() if not line.strip().startswith("--")
    ]
    cleaned = "\n".join(cleaned_lines)
    return [chunk.strip() for chunk in cleaned.split(";") if chunk.strip()]


def run_migrations(conn: sqlite3.Connection) -> list[str]:
    """Apply every migration that has not run yet. Returns the versions applied."""
    _ensure_migrations_table(conn)
    applied: list[str] = []

    for path in pending_migrations(conn):
        for statement in _statements(path.read_text(encoding="utf-8")):
            if _is_already_applied(conn, statement):
                continue
            conn.execute(statement)
        conn.execute("INSERT INTO schema_migrations (version) VALUES (?)", (path.stem,))
        applied.append(path.stem)

    conn.commit()
    return applied


def init_db() -> None:
    """Create the schema if it is absent, then bring it up to date."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='takes'")
        fresh = cursor.fetchone() is None
        if fresh:
            conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
            conn.commit()
        run_migrations(conn)
    finally:
        conn.close()


def migration_status() -> dict:
    """What has run and what is waiting. Used by the health endpoint."""
    conn = get_connection()
    try:
        return {
            "applied": sorted(applied_migrations(conn)),
            "pending": [p.stem for p in pending_migrations(conn)],
        }
    finally:
        conn.close()
