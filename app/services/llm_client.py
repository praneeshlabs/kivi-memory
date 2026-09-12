"""Model access, behind one interface.

Every model id lives here and nowhere else. This is deliberate: Groq
deprecated llama-3.3 mid-build and Sarvam deprecated Sarvam-M, both without
warning. When that happens again the fix should be one line in this file, not
a hunt through prompt code.

Providers are tried in order. Sarvam first — it is the Indic-native stack and
the reason this app can handle Tamil, Hindi and code-mixed speech at all —
with Groq as fallback so the app still runs before a Sarvam key is added.
"""
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import httpx

# v2 is gated behind a closed beta ("endpoint is currently in beta"); v1 is the
# generally available one. Verified against a live key.
SARVAM_CHAT_URL = "https://api.sarvam.ai/v1/chat/completions"
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

# Sarvam-M was retired and sarvam-30b is not exposed on v1. The API itself
# reports the live set: sarvam-105b and sarvam-105b-conversations. The
# conversations variant is documented for real-time/voice workloads, which is
# exactly what the router and conflict checks are.
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "sarvam-105b")
SARVAM_FAST_MODEL = os.getenv("SARVAM_FAST_MODEL", "sarvam-105b-conversations")
# Saarika was folded into the Saaras line on the /speech-to-text endpoint.
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v4")
SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
SARVAM_TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "shubh")

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")
GROQ_TRANSCRIBE_MODEL = os.getenv("GROQ_TRANSCRIBE_MODEL", "whisper-large-v3-turbo")

HTTP_TIMEOUT = 45.0

_groq_client = None
_http: Optional[httpx.Client] = None

# A fallback that hides the reason is worse than no fallback: the app keeps
# working while the stack you meant to run is quietly unused.
_last_sarvam_error: Optional[str] = None


def last_sarvam_error() -> Optional[str]:
    return _last_sarvam_error


def _note_sarvam_error(message: str) -> None:
    global _last_sarvam_error
    _last_sarvam_error = message[:300]


def sarvam_key() -> Optional[str]:
    return os.getenv("SARVAM_API_KEY", "").strip() or None


def groq_key() -> Optional[str]:
    return os.getenv("GROQ_API_KEY", "").strip() or None


def http() -> httpx.Client:
    """One pooled client: TLS handshakes per call were costing real latency."""
    global _http
    if _http is None:
        _http = httpx.Client(
            timeout=HTTP_TIMEOUT,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        )
    return _http


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq

        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client


@dataclass
class LLMResult:
    data: dict[str, Any]
    provider: str
    model: str
    tokens: Optional[int]
    latency_ms: int


def _extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object out of a model response.

    Not every model honours response_format, and a stray prose preamble should
    degrade to a usable object rather than an exception that loses the turn.
    """
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return {}
    return {}


def _sarvam_chat(model: str, system: str, user: str, temperature: float) -> Optional[LLMResult]:
    key = sarvam_key()
    if not key:
        return None
    started = time.perf_counter()
    try:
        response = http().post(
            SARVAM_CHAT_URL,
            headers={"api-subscription-key": key, "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "response_format": {"type": "json_object"},
            },
        )
        if response.status_code >= 400:
            _note_sarvam_error(f"HTTP {response.status_code}: {response.text[:200]}")
            return None
        payload = response.json()
    except Exception as exc:
        _note_sarvam_error(f"{type(exc).__name__}: {exc}")
        return None

    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None

    usage = payload.get("usage") or {}
    return LLMResult(
        data=_extract_json(content),
        provider="sarvam",
        model=model,
        tokens=usage.get("total_tokens"),
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


def _groq_chat(model: str, system: str, user: str, temperature: float) -> Optional[LLMResult]:
    if not groq_key():
        return None
    started = time.perf_counter()
    try:
        response = get_groq_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        content = response.choices[0].message.content
    except Exception:
        return None

    usage = getattr(response, "usage", None)
    return LLMResult(
        data=_extract_json(content),
        provider="groq",
        model=model,
        tokens=int(usage.total_tokens) if usage and usage.total_tokens else None,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


def chat_json(system: str, user: str, temperature: float = 0.2, fast: bool = False) -> LLMResult:
    """One structured-JSON turn, Sarvam first then Groq.

    Returns an empty result rather than raising: a failed classification should
    degrade the answer, never drop the user's message.
    """
    attempts: list[Callable[[], Optional[LLMResult]]] = [
        lambda: _sarvam_chat(
            SARVAM_FAST_MODEL if fast else SARVAM_MODEL, system, user, temperature
        ),
        lambda: _groq_chat(
            GROQ_FAST_MODEL if fast else GROQ_MODEL, system, user, temperature
        ),
    ]
    for attempt in attempts:
        result = attempt()
        if result is not None and result.data:
            return result
    return LLMResult(data={}, provider="none", model="none", tokens=None, latency_ms=0)


def active_provider() -> str:
    """Which provider is actually serving, not merely which key is present.

    Reporting "sarvam" just because a key exists is how a dead Sarvam path
    stayed invisible behind a working Groq fallback.
    """
    if sarvam_key() and _last_sarvam_error is None:
        return "sarvam"
    if groq_key():
        return "groq (sarvam fallback)" if sarvam_key() else "groq"
    return "none"


def preflight() -> dict[str, Any]:
    """Actually call Sarvam once and report what happens.

    Presence of a key proves nothing — a key with no credits authenticates
    fine and fails every request. This is the check to run before a demo.
    """
    if not sarvam_key():
        return {"ok": False, "reason": "SARVAM_API_KEY is not set", "detail": None}

    result = _sarvam_chat(SARVAM_MODEL, "Reply with JSON.", 'Return {"ok": true}', 0.0)
    if result is not None:
        return {"ok": True, "reason": f"{SARVAM_MODEL} responded", "detail": None}
    return {
        "ok": False,
        "reason": "Sarvam is configured but not serving; Groq is answering instead",
        "detail": last_sarvam_error(),
    }


def require_sarvam() -> bool:
    """KIVI_REQUIRE_SARVAM=true refuses to start on the fallback.

    Existing to stop exactly one failure: demoing an Indic-stack project while
    every request is silently served by a US model because billing lapsed.
    """
    return os.getenv("KIVI_REQUIRE_SARVAM", "").strip().lower() in ("1", "true", "yes")


def provider_status() -> dict[str, Any]:
    return {
        "active": active_provider(),
        "sarvam_last_error": last_sarvam_error(),
        "sarvam_configured": bool(sarvam_key()),
        "groq_configured": bool(groq_key()),
        "sarvam_healthy": bool(sarvam_key()) and _last_sarvam_error is None,
        "chat_model": SARVAM_MODEL if sarvam_key() else GROQ_MODEL,
        "stt_model": SARVAM_STT_MODEL if sarvam_key() else GROQ_TRANSCRIBE_MODEL,
        "tts_model": SARVAM_TTS_MODEL if sarvam_key() else None,
        "tts_available": bool(sarvam_key()) and _last_sarvam_error is None,
    }
