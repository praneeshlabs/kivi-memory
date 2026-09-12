"""OAuth2 authorisation-code flow for Kivi's connected accounts.

The user authorises Kivi on the provider's own consent screen and Kivi receives
a scoped token. It never sees an account password, the grant is limited to the
scopes listed here, and the user can revoke it at the provider at any time —
none of which is true of an app password pasted into a config file.

What still lives in .env is the *application's* identity (client id and secret)
issued when you register Kivi with each provider. That is not a personal
credential and cannot be used to sign in as you.
"""
import json
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx

from app.db import get_connection

CALLBACK_BASE = os.getenv("KIVI_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
HTTP_TIMEOUT = 20.0

# Pending authorisations, keyed by the opaque state we hand the provider.
# In-process is right here: these live for the seconds between redirecting the
# user out and them coming back, and must not survive a restart.
_pending_states: dict[str, float] = {}
STATE_TTL_SECONDS = 600


@dataclass
class Provider:
    key: str
    name: str
    authorize_url: str
    token_url: str
    scopes: list[str]
    client_id_env: str
    client_secret_env: str
    setup_url: str
    summary: str
    direction: str
    extra_auth_params: dict[str, str] = field(default_factory=dict)
    # Providers differ on how the token response carries the account label.
    uses_basic_auth: bool = False

    @property
    def client_id(self) -> Optional[str]:
        return os.getenv(self.client_id_env, "").strip() or None

    @property
    def client_secret(self) -> Optional[str]:
        return os.getenv(self.client_secret_env, "").strip() or None

    @property
    def redirect_uri(self) -> str:
        return f"{CALLBACK_BASE}/api/auth/{self.key}/callback"

    @property
    def registered(self) -> bool:
        return bool(self.client_id and self.client_secret)


PROVIDERS: dict[str, Provider] = {
    "google": Provider(
        key="google",
        name="Google",
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        # Identity for signing in, plus read-only Gmail. Nothing that can send,
        # delete, or touch anything else in the account.
        scopes=[
            "openid",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/gmail.readonly",
        ],
        client_id_env="GOOGLE_CLIENT_ID",
        client_secret_env="GOOGLE_CLIENT_SECRET",
        setup_url="https://console.cloud.google.com/apis/credentials",
        summary="Sign in, and learn from mail you've sent.",
        direction="read",
        # access_type=offline + prompt=consent is what makes Google return a
        # refresh token; without it the connection dies in an hour.
        extra_auth_params={"access_type": "offline", "prompt": "consent"},
    ),
    "notion": Provider(
        key="notion",
        name="Notion",
        authorize_url="https://api.notion.com/v1/oauth/authorize",
        token_url="https://api.notion.com/v1/oauth/token",
        scopes=[],  # Notion scopes are chosen by the user when picking pages
        client_id_env="NOTION_CLIENT_ID",
        client_secret_env="NOTION_CLIENT_SECRET",
        setup_url="https://www.notion.so/my-integrations",
        summary="Publish your memory to a Notion page.",
        direction="write",
        extra_auth_params={"owner": "user"},
        uses_basic_auth=True,
    ),
    "slack": Provider(
        key="slack",
        name="Slack",
        authorize_url="https://slack.com/oauth/v2/authorize",
        token_url="https://slack.com/api/oauth.v2.access",
        scopes=["chat:write", "channels:read"],
        client_id_env="SLACK_CLIENT_ID",
        client_secret_env="SLACK_CLIENT_SECRET",
        setup_url="https://api.slack.com/apps",
        summary="Send yourself memory digests and reminders.",
        direction="write",
    ),
}


class OAuthError(RuntimeError):
    """Raised when an authorisation or token exchange fails."""


# ------------------------------------------------------------------ state

def _prune_states() -> None:
    cutoff = time.time() - STATE_TTL_SECONDS
    for value in [s for s, created in _pending_states.items() if created < cutoff]:
        _pending_states.pop(value, None)


def issue_state() -> str:
    _prune_states()
    state = secrets.token_urlsafe(32)
    _pending_states[state] = time.time()
    return state


def consume_state(state: str) -> bool:
    """A state may be redeemed once; this is the CSRF guard on the callback."""
    _prune_states()
    return _pending_states.pop(state, None) is not None


# ------------------------------------------------------------- token store

def save_token(provider_key: str, payload: dict[str, Any], account_label: Optional[str]) -> None:
    expires_in = payload.get("expires_in")
    expires_at = (
        (datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))).isoformat()
        if expires_in else None
    )
    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT refresh_token FROM oauth_tokens WHERE provider = ?", (provider_key,)
        ).fetchone()
        # Google only returns a refresh token on first consent; keep the old one
        # rather than nulling a working connection on re-auth.
        refresh_token = payload.get("refresh_token") or (
            existing["refresh_token"] if existing else None
        )
        conn.execute(
            """
            INSERT INTO oauth_tokens (
                provider, access_token, refresh_token, token_type,
                scopes, expires_at, account_label, connected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(provider) DO UPDATE SET
                access_token = excluded.access_token,
                refresh_token = excluded.refresh_token,
                token_type = excluded.token_type,
                scopes = excluded.scopes,
                expires_at = excluded.expires_at,
                account_label = excluded.account_label,
                connected_at = CURRENT_TIMESTAMP
            """,
            (
                provider_key,
                payload.get("access_token", ""),
                refresh_token,
                payload.get("token_type"),
                payload.get("scope") or json.dumps(payload.get("scopes", [])),
                expires_at,
                account_label,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_token(provider_key: str) -> Optional[sqlite3.Row]:
    conn = get_connection()
    try:
        return conn.execute(
            "SELECT * FROM oauth_tokens WHERE provider = ?", (provider_key,)
        ).fetchone()
    finally:
        conn.close()


def disconnect(provider_key: str) -> bool:
    conn = get_connection()
    try:
        cursor = conn.execute("DELETE FROM oauth_tokens WHERE provider = ?", (provider_key,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# ------------------------------------------------------------------- flow

def authorize_url(provider: Provider) -> str:
    if not provider.registered:
        raise OAuthError(
            f"{provider.name} is not registered yet. Add {provider.client_id_env} and "
            f"{provider.client_secret_env} to .env — see {provider.setup_url}"
        )
    params = {
        "client_id": provider.client_id,
        "redirect_uri": provider.redirect_uri,
        "response_type": "code",
        "state": issue_state(),
        **provider.extra_auth_params,
    }
    if provider.scopes:
        # Slack separates bot scopes with commas; everyone else uses spaces.
        separator = "," if provider.key == "slack" else " "
        params["scope"] = separator.join(provider.scopes)

    return f"{provider.authorize_url}?{httpx.QueryParams(params)}"


def exchange_code(provider: Provider, code: str) -> dict[str, Any]:
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": provider.redirect_uri,
    }
    headers = {"Accept": "application/json"}
    auth = None

    if provider.uses_basic_auth:
        auth = (provider.client_id, provider.client_secret)
    else:
        data["client_id"] = provider.client_id
        data["client_secret"] = provider.client_secret

    try:
        response = httpx.post(
            provider.token_url, data=data, headers=headers, auth=auth, timeout=HTTP_TIMEOUT
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        raise OAuthError(
            f"{provider.name} rejected the token exchange "
            f"({exc.response.status_code}): {exc.response.text[:200]}"
        ) from exc
    except Exception as exc:
        raise OAuthError(f"Could not reach {provider.name}: {exc}") from exc

    # Slack returns HTTP 200 with ok:false on failure.
    if provider.key == "slack" and not payload.get("ok", True):
        raise OAuthError(f"Slack refused the connection: {payload.get('error')}")
    return payload


def refresh_if_needed(provider: Provider, token: sqlite3.Row) -> Optional[str]:
    """Return a usable access token, refreshing first if it has expired."""
    expires_at = token["expires_at"]
    if not expires_at:
        return token["access_token"]

    try:
        expiry = datetime.fromisoformat(expires_at)
    except ValueError:
        return token["access_token"]

    # Refresh a minute early so a slow request doesn't straddle expiry.
    if datetime.now(timezone.utc) < expiry - timedelta(seconds=60):
        return token["access_token"]

    if not token["refresh_token"]:
        return None

    try:
        response = httpx.post(
            provider.token_url,
            data={
                "grant_type": "refresh_token",
                "refresh_token": token["refresh_token"],
                "client_id": provider.client_id,
                "client_secret": provider.client_secret,
            },
            timeout=HTTP_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None

    save_token(provider.key, payload, token["account_label"])
    return payload.get("access_token")


def access_token(provider_key: str) -> Optional[str]:
    """A valid access token for this provider, or None if not connected."""
    provider = PROVIDERS.get(provider_key)
    token = load_token(provider_key)
    if not provider or token is None:
        return None
    return refresh_if_needed(provider, token)


def account_label(provider_key: str, payload: dict[str, Any]) -> Optional[str]:
    """A human-readable name for what just got connected."""
    if provider_key == "slack":
        team = payload.get("team") or {}
        return team.get("name")
    if provider_key == "notion":
        return payload.get("workspace_name") or (payload.get("owner") or {}).get("type")
    if provider_key == "google":
        token = payload.get("access_token")
        if not token:
            return None
        try:
            response = httpx.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {token}"},
                timeout=HTTP_TIMEOUT,
            )
            response.raise_for_status()
            return response.json().get("email")
        except Exception:
            return None
    return None


def status(provider_key: str) -> dict[str, Any]:
    provider = PROVIDERS[provider_key]
    token = load_token(provider_key)
    return {
        "key": provider.key,
        "name": provider.name,
        "direction": provider.direction,
        "summary": provider.summary,
        "registered": provider.registered,
        "connected": token is not None,
        "account_label": token["account_label"] if token else None,
        "connected_at": token["connected_at"] if token else None,
        "setup_url": provider.setup_url,
        "redirect_uri": provider.redirect_uri,
        "needs_env": [] if provider.registered
                     else [provider.client_id_env, provider.client_secret_env],
    }
