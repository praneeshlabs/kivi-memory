"""POST /api/takes - ingest a dictation and run it through the memory pipeline."""
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.services.memory_pipeline import process_take
from app.services.transcription import MAX_AUDIO_BYTES, transcribe_audio

router = APIRouter(prefix="/api", tags=["takes"])


class TakeRequest(BaseModel):
    raw_text: str
    app_context: Optional[str] = None


@router.post("/takes")
def create_take(payload: TakeRequest):
    if not payload.raw_text.strip():
        raise HTTPException(status_code=400, detail="raw_text must not be empty")
    return process_take(payload.raw_text, payload.app_context)


@router.post("/transcribe")
async def transcribe_only(audio: UploadFile = File(...)):
    """Speech to text with no side effects, for the Hey Kivi box: a spoken
    question must not be recorded as a take before it has been routed."""
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty audio upload")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="audio file too large (25MB max)")

    try:
        transcript = transcribe_audio(audio.filename or "dictation.webm", audio_bytes)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"transcription failed: {exc}") from exc

    if not transcript:
        raise HTTPException(status_code=422, detail="no speech detected in the recording")
    return {"transcript": transcript}


@router.post("/takes/audio")
async def create_take_from_audio(
    audio: UploadFile = File(...),
    app_context: Optional[str] = Form(None),
):
    """Voice dictation: transcribe, then run the same pipeline as a typed take."""
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty audio upload")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="audio file too large (25MB max)")

    try:
        transcript = transcribe_audio(audio.filename or "dictation.webm", audio_bytes)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"transcription failed: {exc}") from exc

    if not transcript:
        raise HTTPException(status_code=422, detail="no speech detected in the recording")

    result = process_take(transcript, app_context)
    result["transcript"] = transcript
    return result
