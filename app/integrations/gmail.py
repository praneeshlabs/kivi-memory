"""Gmail: let Kivi learn from mail you've sent.

Authorised over OAuth with the read-only Gmail scope, so Kivi holds a scoped,
revocable token rather than a password. It cannot send, delete, or modify
anything, and you can withdraw access at myaccount.google.com/permissions.

Sent mail is the highest-signal source for a memory system — it is the user
writing about their own life in their own words, rather than marketing they
happened to receive. Each message becomes a take and goes through the normal
filter, so Kivi still decides what is worth keeping.
"""
import base64
from typing import Any, Optional

import httpx

from app.integrations.base import IntegrationError
from app.integrations.oauth import access_token

API_ROOT = "https://gmail.googleapis.com/gmail/v1/users/me"
TIMEOUT = 20.0
MAX_BODY_CHARS = 1500


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _header_value(payload: dict[str, Any], name: str) -> str:
    for header in payload.get("headers", []) or []:
        if header.get("name", "").lower() == name.lower():
            return header.get("value", "")
    return ""


def _decode_b64(data: str) -> str:
    # Gmail uses URL-safe base64 without padding.
    padded = data + "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode(padded).decode("utf-8", "replace")
    except Exception:
        return ""


def _plain_body(payload: dict[str, Any]) -> str:
    """Depth-first search for the first text/plain part."""
    if payload.get("mimeType") == "text/plain":
        data = (payload.get("body") or {}).get("data")
        if data:
            return _decode_b64(data)
    for part in payload.get("parts", []) or []:
        found = _plain_body(part)
        if found:
            return found
    return ""


def fetch_sent(limit: int = 10) -> list[dict[str, str]]:
    """Most recent sent messages as {subject, to, date, body}."""
    token = access_token("google")
    if not token:
        raise IntegrationError("Google is not connected")

    try:
        listing = httpx.get(
            f"{API_ROOT}/messages",
            headers=_headers(token),
            params={"labelIds": "SENT", "maxResults": limit},
            timeout=TIMEOUT,
        )
        listing.raise_for_status()
        ids = [item["id"] for item in listing.json().get("messages", [])]
    except httpx.HTTPStatusError as exc:
        raise IntegrationError(
            f"Gmail refused the request ({exc.response.status_code}). "
            "Try disconnecting and reconnecting Google."
        ) from exc
    except Exception as exc:
        raise IntegrationError(f"Could not reach Gmail: {exc}") from exc

    messages: list[dict[str, str]] = []
    for message_id in ids:
        try:
            detail = httpx.get(
                f"{API_ROOT}/messages/{message_id}",
                headers=_headers(token),
                params={"format": "full"},
                timeout=TIMEOUT,
            )
            detail.raise_for_status()
            payload = detail.json().get("payload", {})
        except Exception:
            continue

        body = " ".join(_plain_body(payload).split())[:MAX_BODY_CHARS]
        if not body:
            continue
        messages.append({
            "subject": _header_value(payload, "Subject"),
            "to": _header_value(payload, "To"),
            "date": _header_value(payload, "Date"),
            "body": body,
        })

    return messages


def as_take_text(message: dict[str, str]) -> str:
    """Frame a sent email so the filter reads it as the user's own words."""
    subject = message.get("subject") or "(no subject)"
    return (
        f"From an email I sent to {message.get('to') or 'someone'} "
        f"about \"{subject}\": {message.get('body', '')}"
    )
