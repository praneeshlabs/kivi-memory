"""BM25 keyword search over memories, via SQLite's built-in FTS5.

Dense embeddings are weak exactly where names live: "Pixel", "Meera",
"Infosys" carry almost no distributional meaning, so a vector search for
"what colour is Pixel" can rank a memory about a different pet above the right
one. BM25 keys on the literal token and gets those right. Fusing the two is
what makes retrieval robust rather than merely plausible.

FTS5 ships with SQLite, so this costs no dependency and the index is
maintained in the same file as the data.
"""
import re
import sqlite3
from typing import Iterable

FTS_TABLE = "memories_fts"

# Strip anything FTS5 treats as syntax; user text is data, not a query language.
_FTS_SYNTAX = re.compile(r'[^\w\s-￿]', re.UNICODE)


def ensure_index(conn: sqlite3.Connection) -> None:
    """Create the index and keep it in step with memories via triggers."""
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE}
        USING fts5(content, memory_id UNINDEXED, tokenize='unicode61')
        """
    )


def reindex(conn: sqlite3.Connection) -> int:
    """Rebuild from the memories table. Idempotent; safe to run on startup."""
    ensure_index(conn)
    conn.execute(f"DELETE FROM {FTS_TABLE}")
    rows = conn.execute(
        "SELECT id, content FROM memories WHERE status = 'active'"
    ).fetchall()
    conn.executemany(
        f"INSERT INTO {FTS_TABLE} (content, memory_id) VALUES (?, ?)",
        [(row["content"], row["id"]) for row in rows],
    )
    return len(rows)


def add(conn: sqlite3.Connection, memory_id: int, content: str) -> None:
    ensure_index(conn)
    conn.execute(
        f"INSERT INTO {FTS_TABLE} (content, memory_id) VALUES (?, ?)", (content, memory_id)
    )


def remove(conn: sqlite3.Connection, memory_id: int) -> None:
    conn.execute(f"DELETE FROM {FTS_TABLE} WHERE memory_id = ?", (memory_id,))


def update(conn: sqlite3.Connection, memory_id: int, content: str) -> None:
    remove(conn, memory_id)
    add(conn, memory_id, content)


def _sanitize(query: str) -> str:
    """A bag of OR'd terms. Quoted so FTS5 never parses user text as operators."""
    cleaned = _FTS_SYNTAX.sub(" ", query or "")
    terms = [term for term in cleaned.split() if len(term) > 1]
    return " OR ".join(f'"{term}"' for term in terms[:20])


def search(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[int]:
    """Memory ids ranked by BM25, best first."""
    match = _sanitize(query)
    if not match:
        return []
    try:
        rows = conn.execute(
            f"SELECT memory_id FROM {FTS_TABLE} WHERE {FTS_TABLE} MATCH ? "
            f"ORDER BY rank LIMIT ?",
            (match, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        # Index missing or query still unparseable: dense search alone is a
        # valid answer, so degrade rather than fail the request.
        return []
    return [int(row["memory_id"]) for row in rows]
