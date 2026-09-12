"""Connected apps: status, outbound publishing, inbound learning."""
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.db import get_connection
from app.integrations import gmail, notion, oauth, slack
from app.integrations.base import IntegrationError
from app.services.memory_pipeline import process_take

router = APIRouter(prefix="/api/integrations", tags=["integrations"])


class GmailSyncRequest(BaseModel):
    limit: int = 5


def _active_memories() -> list[dict[str, Any]]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, content, tag, created_at FROM memories "
            "WHERE status = 'active' ORDER BY tag, created_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


@router.get("")
def list_integrations() -> dict[str, Any]:
    """What's connected, and what each one still needs."""
    return {"integrations": [oauth.status(key) for key in oauth.PROVIDERS]}


@router.post("/notion/export")
def notion_export() -> dict[str, Any]:
    memories = _active_memories()
    try:
        result = notion.export_memories(memories, "Kivi memory snapshot")
    except IntegrationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"exported": len(memories), **result}


@router.post("/slack/digest")
def slack_digest() -> dict[str, Any]:
    memories = _active_memories()
    try:
        result = slack.post_message(slack.digest_text(memories))
    except IntegrationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"sent": len(memories), **result}


@router.post("/google/sync")
def gmail_sync(payload: GmailSyncRequest) -> dict[str, Any]:
    """Read recent sent mail and run it through the normal memory filter."""
    limit = max(1, min(payload.limit, 25))
    try:
        messages = gmail.fetch_sent(limit)
    except IntegrationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    results = []
    for message in messages:
        outcome = process_take(gmail.as_take_text(message), app_context="gmail")
        results.append({
            "subject": message["subject"],
            "saved": [item["content"] for item in outcome.get("saved", [])],
            "watched": [item["content"] for item in outcome.get("watched", [])],
            "dropped": len(outcome.get("dropped", [])),
        })
    return {"scanned": len(messages), "results": results}


@router.get("/preflight")
def provider_preflight() -> dict[str, Any]:
    """Live check of whether Sarvam is actually serving. Run before a demo."""
    from app.services.llm_client import preflight

    return preflight()
