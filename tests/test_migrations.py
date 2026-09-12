"""Migrations must work on a fresh database and on an old one.

Both paths are tested because they fail differently. A fresh database already
has everything from schema.sql, so a migration must notice and skip. An old
database is missing the columns, so the same migration must actually run. An
earlier version of the runner recorded every migration as applied while
executing none of them, because each file opens with a comment block and the
statement splitter threw the chunk away.
"""
import sqlite3
from pathlib import Path

import pytest


def _columns(db_path, table):
    conn = sqlite3.connect(db_path)
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


def _tables(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()


def test_statement_splitter_keeps_sql_attached_to_comments():
    from app.db import _statements

    sql = "-- explanation\n--\n-- more explanation\n\nALTER TABLE t ADD COLUMN c TEXT;"
    assert _statements(sql) == ["ALTER TABLE t ADD COLUMN c TEXT"]


def test_fresh_database_records_every_migration(db):
    from app.db import migration_status

    status = migration_status()
    assert status["pending"] == []
    assert "001_watchlist_tag" in status["applied"]
    assert "004_memory_search_index" in status["applied"]


def test_fresh_database_has_the_migrated_shape(db):
    assert "tag" in _columns(db, "watchlist")
    assert {"answer_source", "follow_ups"} <= _columns(db, "answers")
    assert "oauth_tokens" in _tables(db)


def test_old_database_is_upgraded_in_place(tmp_path, monkeypatch):
    """A database created before these migrations must gain the new shape."""
    import app.db as db_module

    root = Path(db_module.__file__).resolve().parent.parent
    base = (root / "schema.sql").read_text(encoding="utf-8")

    # Strip out everything the migrations are responsible for adding.
    base = "\n".join(
        line for line in base.splitlines()
        if "tag TEXT CHECK(tag IN" not in line
        and "answer_source TEXT" not in line
        and "follow_ups TEXT" not in line
    )
    base = base.split("-- TABLE: oauth_tokens")[0]

    old_db = tmp_path / "old.db"
    conn = sqlite3.connect(old_db)
    conn.executescript(base)
    conn.commit()
    conn.close()

    assert "tag" not in _columns(old_db, "watchlist")
    assert "oauth_tokens" not in _tables(old_db)

    monkeypatch.setattr(db_module, "DATABASE_PATH", str(old_db))
    monkeypatch.setattr(db_module, "get_db_path", lambda: old_db)
    db_module.init_db()

    assert "tag" in _columns(old_db, "watchlist")
    assert {"answer_source", "follow_ups"} <= _columns(old_db, "answers")
    assert "oauth_tokens" in _tables(old_db)
    assert db_module.migration_status()["pending"] == []


def test_migrations_are_idempotent(db):
    """Running twice must not fail or double-record."""
    from app.db import get_connection, migration_status, run_migrations

    before = migration_status()["applied"]
    conn = get_connection()
    try:
        assert run_migrations(conn) == []
    finally:
        conn.close()
    assert migration_status()["applied"] == before
