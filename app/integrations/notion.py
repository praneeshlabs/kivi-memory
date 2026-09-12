"""Notion: publish Kivi's memory to a page you can read and share.

Setup (no OAuth app needed for personal use):
  1. https://www.notion.so/my-integrations -> New integration -> copy the
     "Internal Integration Secret".
  2. Open the Notion page you want Kivi to write to, "..." menu ->
     Connections -> add your integration. Without this step Notion returns
     404 for the page even with a valid token.
  3. Copy the page id from its URL (the 32-char hex after the title).
  4. In .env:  NOTION_TOKEN=ntn_...   NOTION_PAGE_ID=...
"""
from typing import Any

import httpx

from app.integrations.base import IntegrationError
from app.integrations.oauth import access_token, load_token

NOTION_VERSION = "2022-06-28"
API_ROOT = "https://api.notion.com/v1"
TIMEOUT = 15.0

# Notion rejects blocks over 2000 characters of rich text.
MAX_BLOCK_CHARS = 1900


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def _target_page(token: str) -> str:
    """The page the user granted access to during the OAuth consent step."""
    try:
        response = httpx.post(
            f"{API_ROOT}/search",
            headers=_headers(token),
            json={"filter": {"property": "object", "value": "page"}, "page_size": 1},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
    except Exception as exc:
        raise IntegrationError(f"Could not reach Notion: {exc}") from exc

    if not results:
        raise IntegrationError(
            "No Notion page was shared with Kivi. Reconnect and tick a page on "
            "the Notion consent screen."
        )
    return results[0]["id"]


def _paragraph(text: str) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [{"type": "text", "text": {"content": text[:MAX_BLOCK_CHARS]}}]
        },
    }


def _heading(text: str) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "heading_2",
        "heading_2": {
            "rich_text": [{"type": "text", "text": {"content": text[:MAX_BLOCK_CHARS]}}]
        },
    }


def export_memories(memories: list[dict[str, Any]], title: str) -> dict[str, Any]:
    """Append a snapshot of memory to the page shared with Kivi."""
    token = access_token("notion")
    if not token:
        raise IntegrationError("Notion is not connected")
    page_id = _target_page(token)

    blocks: list[dict[str, Any]] = [_heading(title)]
    for tag in ("fact", "episode", "preference"):
        group = [m for m in memories if m["tag"] == tag]
        if not group:
            continue
        blocks.append(_paragraph(f"{tag.title()}s ({len(group)})"))
        for memory in group:
            blocks.append(_paragraph(f"• {memory['content']}  —  learned {memory['created_at']}"))

    try:
        response = httpx.patch(
            f"{API_ROOT}/blocks/{page_id}/children",
            headers=_headers(token),
            json={"children": blocks[:100]},  # Notion caps children per request
            timeout=TIMEOUT,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:200]
        raise IntegrationError(f"Notion rejected the write ({exc.response.status_code}): {detail}") from exc
    except Exception as exc:
        raise IntegrationError(f"Could not reach Notion: {exc}") from exc

    return {"blocks_written": len(blocks[:100])}
