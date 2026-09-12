"""Memory lifecycle writes: archive (Step 9 Retire), user delete, user edit."""
import sqlite3
from typing import Any, Optional

from app.services import lexical
from app.services.retrieval import embed, invalidate_cache, serialize_embedding
from app.services.text_utils import normalize_text


def archive_memory(
    conn: sqlite3.Connection,
    memory_id: int,
    retired_reason: str,
    contradicted_by_memory_id: Optional[int] = None,
) -> None:
    """Step 9 Retire: copy the row into memory_archive and mark the original archived."""
    conn.execute(
        """
        INSERT INTO memory_archive (
            id, content, source_take_id, tag, confidence,
            related_memory_ids, created_at, retired_reason, contradicted_by_memory_id
        )
        SELECT id, content, source_take_id, tag, confidence,
               related_memory_ids, created_at, ?, ?
        FROM memories WHERE id = ?
        """,
        (retired_reason, contradicted_by_memory_id, memory_id),
    )
    conn.execute(
        """
        UPDATE memories
           SET status = 'archived',
               retired_reason = ?,
               retired_at = CURRENT_TIMESTAMP,
               contradicted_by_memory_id = ?
         WHERE id = ?
        """,
        (retired_reason, contradicted_by_memory_id, memory_id),
    )
    # An archived memory must leave both search paths, or it keeps being found.
    lexical.remove(conn, memory_id)
    invalidate_cache()


def delete_memory(conn: sqlite3.Connection, memory_id: int) -> bool:
    """User-initiated delete. Archives rather than removing, for auditability."""
    row = conn.execute(
        "SELECT id FROM memories WHERE id = ? AND status = 'active'", (memory_id,)
    ).fetchone()
    if row is None:
        return False
    archive_memory(conn, memory_id, retired_reason="user_deleted")
    return True


def update_memory_content(
    conn: sqlite3.Connection, memory_id: int, new_content: str
) -> Optional[dict[str, Any]]:
    """User-initiated edit. Preserves the first original_content for audit and
    re-embeds, so retrieval reflects the edited text."""
    row = conn.execute(
        "SELECT id, content, user_edited, original_content "
        "FROM memories WHERE id = ? AND status = 'active'",
        (memory_id,),
    ).fetchone()
    if row is None:
        return None

    content = normalize_text(new_content)
    # Keep the very first version, not the previous edit.
    original_content = row["original_content"] if row["user_edited"] else row["content"]

    conn.execute(
        """
        UPDATE memories
           SET content = ?,
               embedding = ?,
               user_edited = 1,
               user_edited_at = CURRENT_TIMESTAMP,
               original_content = ?
         WHERE id = ?
        """,
        (content, serialize_embedding(embed(content)), original_content, memory_id),
    )
    lexical.update(conn, memory_id, content)
    invalidate_cache()
    updated = conn.execute(
        "SELECT id, content, tag, confidence, created_at, source_take_id, "
        "usage_count, last_used_at, user_edited, user_edited_at, original_content "
        "FROM memories WHERE id = ?",
        (memory_id,),
    ).fetchone()
    return dict(updated)
