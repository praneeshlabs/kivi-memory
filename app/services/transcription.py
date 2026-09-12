"""Speech to text.

Sarvam's Saaras line is the primary path: it is trained on Indian languages
and handles code-mixed speech ("nalaiku meeting iruku", "kal dentist hai"),
which is how people in India actually talk and which an English-first model
transcribes badly. Groq Whisper stays as fallback so the app still works
before a Sarvam key is added.

Saarika was the earlier ASR name; Sarvam folded it into the Saaras models on
the /speech-to-text endpoint, so that is what this calls.
"""
from typing import Optional

from app.services.llm_client import (
    GROQ_TRANSCRIBE_MODEL,
    SARVAM_STT_MODEL,
    SARVAM_STT_URL,
    get_groq_client,
    groq_key,
    http,
    sarvam_key,
)
from app.services.text_utils import normalize_text

MAX_AUDIO_BYTES = 25 * 1024 * 1024

# Unset by default: Saaras detects the language, which is what you want when
# the speaker switches mid-sentence. Pin it only for a known-monolingual user.
DEFAULT_LANGUAGE_CODE = "unknown"


def _transcribe_sarvam(filename: str, audio_bytes: bytes) -> Optional[tuple[str, str]]:
    key = sarvam_key()
    if not key:
        return None
    try:
        response = http().post(
            SARVAM_STT_URL,
            headers={"api-subscription-key": key},
            files={"file": (filename, audio_bytes, "audio/webm")},
            data={"model": SARVAM_STT_MODEL, "language_code": DEFAULT_LANGUAGE_CODE},
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None

    transcript = (payload.get("transcript") or "").strip()
    if not transcript:
        return None
    return transcript, payload.get("language_code") or "unknown"


def _transcribe_groq(filename: str, audio_bytes: bytes) -> Optional[tuple[str, str]]:
    if not groq_key():
        return None
    try:
        response = get_groq_client().audio.transcriptions.create(
            file=(filename, audio_bytes),
            model=GROQ_TRANSCRIBE_MODEL,
            response_format="text",
        )
    except Exception:
        return None
    text = response if isinstance(response, str) else getattr(response, "text", "")
    text = (text or "").strip()
    return (text, "unknown") if text else None


def transcribe_audio(filename: str, audio_bytes: bytes) -> Optional[str]:
    """Transcript of a recorded dictation, or None if nothing was said."""
    result = transcribe_with_language(filename, audio_bytes)
    return result[0] if result else None


def transcribe_with_language(
    filename: str, audio_bytes: bytes
) -> Optional[tuple[str, str]]:
    """(transcript, detected language code), or None."""
    for attempt in (_transcribe_sarvam, _transcribe_groq):
        result = attempt(filename, audio_bytes)
        if result:
            transcript, language = result
            cleaned = (normalize_text(transcript) or "").strip()
            if cleaned:
                return cleaned, language
    return None
