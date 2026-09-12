"""Text to speech with Sarvam Bulbul.

Kivi is a voice product, and until now the voice only went one way: you spoke,
it typed back. Bulbul closes the loop in the same Indian languages Saaras
transcribes, so an answer can be heard rather than read.

Requires a Sarvam key. Without one, speaking is simply unavailable — there is
no English-only fallback worth shipping for an Indic voice assistant.
"""
from typing import Optional

from app.services.llm_client import (
    SARVAM_TTS_MODEL,
    SARVAM_TTS_SPEAKER,
    SARVAM_TTS_URL,
    http,
    sarvam_key,
)

# Bulbul v3 caps at 2500 characters; Kivi's answers are far shorter, but a web
# result summary can run long.
MAX_TEXT_CHARS = 2400

SUPPORTED_LANGUAGES = {
    "bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN",
    "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN",
}
DEFAULT_LANGUAGE = "en-IN"


def is_available() -> bool:
    return bool(sarvam_key())


def _language_for(code: Optional[str]) -> str:
    """Map a detected code onto something Bulbul accepts."""
    if not code or code == "unknown":
        return DEFAULT_LANGUAGE
    if code in SUPPORTED_LANGUAGES:
        return code
    # Saaras may return a bare tag like "ta"; Bulbul wants the regional form.
    base = code.split("-")[0].lower()
    for supported in SUPPORTED_LANGUAGES:
        if supported.startswith(f"{base}-"):
            return supported
    return DEFAULT_LANGUAGE


def synthesize(text: str, language_code: Optional[str] = None) -> Optional[str]:
    """Return base64-encoded WAV for the text, or None if unavailable."""
    key = sarvam_key()
    if not key or not (text or "").strip():
        return None

    try:
        response = http().post(
            SARVAM_TTS_URL,
            headers={"api-subscription-key": key, "Content-Type": "application/json"},
            json={
                "text": text[:MAX_TEXT_CHARS],
                "target_language_code": _language_for(language_code),
                "model": SARVAM_TTS_MODEL,
                "speaker": SARVAM_TTS_SPEAKER,
            },
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        # Speech is an enhancement; failing it must never fail the answer.
        return None

    audios = payload.get("audios") or []
    return audios[0] if audios else None
