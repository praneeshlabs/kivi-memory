"""Slack: let Kivi send you things — a memory digest, a reminder.

Setup (bot token, no OAuth redirect needed for your own workspace):
  1. https://api.slack.com/apps -> Create New App -> From scratch.
  2. OAuth & Permissions -> Bot Token Scopes -> add "chat:write".
  3. Install to Workspace, copy the "Bot User OAuth Token" (xoxb-...).
  4. In Slack, invite the bot to the channel: /invite @YourApp
  5. In .env:  SLACK_BOT_TOKEN=xoxb-...   SLACK_CHANNEL=#kivi
"""
from typing import Any

import os

import httpx

from app.integrations.base import IntegrationError
from app.integrations.oauth import access_token

API_ROOT = "https://slack.com/api"
TIMEOUT = 12.0


def post_message(text: str) -> dict[str, Any]:
    """Post to the configured channel."""
    token = access_token("slack")
    if not token:
        raise IntegrationError("Slack is not connected")
    channel = os.getenv("SLACK_CHANNEL", "").strip() or "#general"

    try:
        response = httpx.post(
            f"{API_ROOT}/chat.postMessage",
            headers={"Authorization": f"Bearer {token}"},
            json={"channel": channel, "text": text},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise IntegrationError(f"Could not reach Slack: {exc}") from exc

    # Slack returns HTTP 200 with ok:false for application errors.
    if not payload.get("ok"):
        raise IntegrationError(f"Slack refused the message: {payload.get('error', 'unknown error')}")
    return {"channel": payload.get("channel"), "ts": payload.get("ts")}


def digest_text(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return "Kivi doesn't have any memories yet."
    lines = [f"*Kivi memory digest* — {len(memories)} active"]
    for tag in ("fact", "episode", "preference"):
        group = [m for m in memories if m["tag"] == tag]
        if group:
            lines.append(f"\n*{tag.title()}s*")
            lines.extend(f"• {m['content']}" for m in group)
    return "\n".join(lines)
