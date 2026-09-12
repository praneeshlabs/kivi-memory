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

SARVAM_CHAT_URL = "https://api.sarvam.ai/v2/chat/completions"
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

# Sarvam-M (24B) was retired from the chat API; 30B is its documented
# successor and is plenty for classification and short answers.
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "sarvam-30b")
SARVAM_FAST_MODEL = os.getenv("SARVAM_FAST_MODEL", "sarvam-30b")
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
        response.raise_for_status()
        payload = response.json()
    except Exception:
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
    if sarvam_key():
        return "sarvam"
    return "groq" if groq_key() else "none"


def provider_status() -> dict[str, Any]:
    return {
        "active": active_provider(),
        "sarvam_configured": bool(sarvam_key()),
        "groq_configured": bool(groq_key()),
        "chat_model": SARVAM_MODEL if sarvam_key() else GROQ_MODEL,
        "stt_model": SARVAM_STT_MODEL if sarvam_key() else GROQ_TRANSCRIBE_MODEL,
        "tts_model": SARVAM_TTS_MODEL if sarvam_key() else None,
        "tts_available": bool(sarvam_key()),
    }
