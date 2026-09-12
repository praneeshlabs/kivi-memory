"""Kivi semantic memory system - FastAPI entrypoint."""
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.db import get_connection, init_db
from app.routes import auth, hey_kivi, integrations, memories, takes
from app.services import cache as cache_service
from app.services.llm_client import provider_status
from app.services.retrieval import EMBEDDING_MODEL_NAME, cache_size, warm_up

load_dotenv()


def _build_search_index() -> None:
    """Keep the BM25 index in step with the memories table on boot."""
    from app.db import get_connection
    from app.services import lexical

    conn = get_connection()
    try:
        lexical.reindex(conn)
        conn.commit()
    finally:
        conn.close()


def require_groq_api_key() -> None:
    """Fail fast on startup if no model provider is configured at all."""
    if not os.getenv("SARVAM_API_KEY", "").strip() and not os.getenv("GROQ_API_KEY", "").strip():
        raise RuntimeError(
            "No model provider configured. Copy .env.example to .env and set "
            "SARVAM_API_KEY (https://dashboard.sarvam.ai) for the Indic stack, "
            "or GROQ_API_KEY as a fallback, then restart the server."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: confirm config, then run schema.sql and confirm the DB connects.
    require_groq_api_key()
    init_db()
    _build_search_index()
    # Load the embedding model now so the first real request doesn't pay for it.
    warm_up()
    yield


app = FastAPI(title="Kivi Memory", lifespan=lifespan)

app.include_router(takes.router)
app.include_router(hey_kivi.router)
app.include_router(memories.router)
app.include_router(integrations.router)
app.include_router(auth.router)


@app.get("/api/health")
def health():
    """Hello-world endpoint that confirms the DB connection and schema are working."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row["name"] for row in cursor.fetchall()]
        return {
            "status": "ok",
            "message": "Kivi backend is running",
            "db_connected": True,
            "tables": tables,
            "models": provider_status(),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "memories_cached": cache_size(),
            "cache": cache_service.stats(),
        }
    finally:
        conn.close()


app.mount("/", StaticFiles(directory="static", html=True), name="static")
