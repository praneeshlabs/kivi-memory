"""Sign-in and app connections, both over OAuth.

Kivi never handles an account password. The user authorises on the provider's
own consent screen; Kivi receives a scoped token it can use and the user can
revoke. Signing in with Google both identifies the owner and connects Gmail,
because it is the same grant.
"""
import secrets
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse

from app.integrations import oauth

router = APIRouter(prefix="/api/auth", tags=["auth"])

SESSION_COOKIE = "kivi_session"

# Single-user app: one owner, one session secret per process. Kivi is not
# multi-tenant — memories are not scoped per user — so this marks "the owner is
# signed in", not "which of many users this is".
_sessions: dict[str, str] = {}


def current_account(request: Request) -> str | None:
    token = request.cookies.get(SESSION_COOKIE)
    return _sessions.get(token) if token else None


@router.get("/status")
def auth_status(request: Request) -> dict[str, Any]:
    """Who is signed in, and what is connected."""
    return {
        "signed_in_as": current_account(request),
        "providers": [oauth.status(key) for key in oauth.PROVIDERS],
    }


@router.get("/{provider_key}/start")
def start(provider_key: str):
    """Send the user to the provider's consent screen."""
    provider = oauth.PROVIDERS.get(provider_key)
    if provider is None:
        raise HTTPException(status_code=404, detail=f"unknown provider: {provider_key}")
    try:
        return RedirectResponse(oauth.authorize_url(provider), status_code=302)
    except oauth.OAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _closing_page(message: str, ok: bool) -> HTMLResponse:
    """The popup reports back to the opener and closes itself."""
    colour = "#7ed957" if ok else "#d9705a"
    return HTMLResponse(
        f"""<!doctype html><meta charset="utf-8">
<title>Kivi</title>
<body style="background:#0d0f0d;color:{colour};font-family:system-ui;
             display:flex;align-items:center;justify-content:center;height:100vh;margin:0">
<p>{message}</p>
<script>
  try {{ window.opener && window.opener.postMessage("kivi-auth-changed", "*"); }} catch (e) {{}}
  setTimeout(function () {{ window.close(); }}, 1200);
</script>
</body>"""
    )


@router.get("/{provider_key}/callback")
def callback(provider_key: str, response: Response, code: str = "", state: str = "", error: str = ""):
    """Provider redirects here with an authorisation code."""
    provider = oauth.PROVIDERS.get(provider_key)
    if provider is None:
        raise HTTPException(status_code=404, detail=f"unknown provider: {provider_key}")
    if error:
        return _closing_page(f"{provider.name} sign-in was cancelled.", ok=False)
    if not code:
        return _closing_page("No authorisation code came back.", ok=False)
    # The state is single-use and proves this callback follows our own redirect.
    if not oauth.consume_state(state):
        return _closing_page("That sign-in link has expired. Try again.", ok=False)

    try:
        payload = oauth.exchange_code(provider, code)
    except oauth.OAuthError as exc:
        return _closing_page(str(exc), ok=False)

    label = oauth.account_label(provider_key, payload)
    oauth.save_token(provider_key, payload, label)

    page = _closing_page(f"{provider.name} connected. You can close this window.", ok=True)
    if provider_key == "google" and label:
        # Signing in with Google is also how the owner signs in to Kivi.
        session_token = secrets.token_urlsafe(32)
        _sessions[session_token] = label
        page.set_cookie(
            SESSION_COOKIE, session_token,
            httponly=True, samesite="lax", max_age=60 * 60 * 24 * 30,
        )
    return page


@router.post("/{provider_key}/disconnect")
def disconnect(provider_key: str) -> dict[str, Any]:
    if provider_key not in oauth.PROVIDERS:
        raise HTTPException(status_code=404, detail=f"unknown provider: {provider_key}")
    removed = oauth.disconnect(provider_key)
    return {"disconnected": removed, "provider": provider_key}


@router.post("/sign-out")
def sign_out(request: Request) -> dict[str, Any]:
    """Ends the local session. Connections stay until explicitly disconnected."""
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        _sessions.pop(token, None)
    response = {"signed_out": True}
    return response
