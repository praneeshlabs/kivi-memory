"""POST /api/hey-kivi - answer a question from stored memories, or decline honestly."""
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import tts
from app.services.qa_pipeline import process_hey_kivi

router = APIRouter(prefix="/api", tags=["hey-kivi"])


class Turn(BaseModel):
    role: Literal["user", "kivi"]
    text: str


class HeyKiviRequest(BaseModel):
    query: str
    # Ask Kivi to speak the answer back (Bulbul; needs a Sarvam key).
    speak: bool = False
    language_code: Optional[str] = None
    # Set when the user clicked a follow-up chip: this message ANSWERS that
    # question rather than asking a new one.
    answering: Optional[str] = None
    # Recent turns, oldest first. Sent by the client so a follow-up like
    # "tell me more about it" can be resolved against what came before.
    history: Optional[list[Turn]] = None


@router.post("/hey-kivi")
def hey_kivi(payload: HeyKiviRequest) -> dict[str, Any]:
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    history = [turn.model_dump() for turn in payload.history or []]
    result = process_hey_kivi(payload.query, history, answering=payload.answering)

    if payload.speak and result.get("answer"):
        audio = tts.synthesize(result["answer"], payload.language_code)
        result["audio_base64"] = audio
        result["spoken"] = audio is not None
    return result
