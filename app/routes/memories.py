"""Memory view endpoints: list, user edit (PATCH), user delete (DELETE)."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.db import get_connection
from app.services.consolidate import consolidate
from app.services.memory_store import delete_memory, update_memory_content

router = APIRouter(prefix="/api", tags=["memories"])


class MemoryPatch(BaseModel):
    content: str


@router.get("/memories")
def list_memories():
    """Active memories with their source take, for the memory view."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT m.id, m.content, m.tag, m.confidence, m.created_at,
                   m.usage_count, m.last_used_at, m.user_edited, m.user_edited_at,
                   m.original_content, m.source_take_id,
                   t.raw_text AS source_take_raw_text,
                   t.created_at AS source_take_created_at
              FROM memories m
              JOIN takes t ON t.id = m.source_take_id
             WHERE m.status = 'active'
             ORDER BY m.created_at DESC
            """
        ).fetchall()
        return {"memories": [dict(row) for row in rows]}
    finally:
        conn.close()


@router.delete("/memories/{memory_id}")
def delete_memory_endpoint(memory_id: int):
    """Step 9 Retire: archive the memory rather than removing it."""
    conn = get_connection()
    try:
        if not delete_memory(conn, memory_id):
            raise HTTPException(status_code=404, detail="memory not found or already archived")
        conn.commit()
        return {"id": memory_id, "status": "archived", "retired_reason": "user_deleted"}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@router.patch("/memories/{memory_id}")
def patch_memory_endpoint(memory_id: int, payload: MemoryPatch):
    """Step 5 Show: user correction, preserving the original for audit."""
    if not payload.content.strip():
        raise HTTPException(status_code=400, detail="content must not be empty")
    conn = get_connection()
    try:
        updated = update_memory_content(conn, memory_id, payload.content)
        if updated is None:
            raise HTTPException(status_code=404, detail="memory not found or archived")
        conn.commit()
        return updated
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@router.post("/memories/consolidate")
def consolidate_memories(dry_run: bool = False) -> dict:
    """Collapse memories that restate one another.

    Ingestion judges redundancy against what existed at that moment, so
    restatements arriving over days can each look novel. This is the sweep.
    Pass ?dry_run=true to see what it would do without changing anything.
    """
    return consolidate(dry_run=dry_run)
