"""Test fixtures: a throwaway DB and a scripted LLM.

The LLM is stubbed on purpose. These tests pin the *pipeline's* behaviour —
what it does with a given classification — so a model swap or a provider
deprecation shows up as a failing assertion rather than as a user noticing
something odd in a screenshot three weeks later.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """A fresh SQLite file per test, with the real schema applied."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_PATH", str(db_path))

    import app.db as db_module

    monkeypatch.setattr(db_module, "DATABASE_PATH", str(db_path))
    monkeypatch.setattr(db_module, "get_db_path", lambda: db_path)
    db_module.init_db()

    from app.services import cache, retrieval

    cache.clear()
    retrieval.invalidate_cache()
    yield db_path
    cache.clear()
    retrieval.invalidate_cache()


class ScriptedLLM:
    """Serves queued payloads per prompt kind, recording what it was asked.

    Keyed by kind rather than strict call order on purpose: the pipeline
    legitimately skips the conflict check when there are no candidates, and a
    positional queue would then hand the next call the wrong payload — a test
    harness artefact that looks exactly like a product bug.
    """

    FILTER = "filter"
    CONFLICT = "conflict"
    OTHER = "other"

    def __init__(self):
        self.queues: dict[str, list[dict]] = {}
        self.calls: list[tuple[str, str]] = []

    @staticmethod
    def kind_of(system: str) -> str:
        if "memory filter" in system:
            return ScriptedLLM.FILTER
        if "relation to the new one" in system:
            return ScriptedLLM.CONFLICT
        return ScriptedLLM.OTHER

    def push(self, payload: dict, kind: str = FILTER):
        self.queues.setdefault(kind, []).append(payload)
        return self

    def push_conflict(self, payload: dict):
        return self.push(payload, ScriptedLLM.CONFLICT)

    def __call__(self, system, user, temperature=0.2, fast=False):
        from app.services.llm_client import LLMResult

        self.calls.append((system, user))
        queue = self.queues.get(self.kind_of(system), [])
        payload = queue.pop(0) if queue else {}
        return LLMResult(data=payload, provider="test", model="test", tokens=7, latency_ms=1)


@pytest.fixture()
def llm(monkeypatch):
    """Replace the single LLM entry point everywhere it is imported."""
    scripted = ScriptedLLM()
    import app.services.llm_client as client
    import app.services.memory_pipeline as memory_pipeline

    monkeypatch.setattr(client, "chat_json", scripted)
    monkeypatch.setattr(memory_pipeline, "chat_json", scripted)
    return scripted


@pytest.fixture()
def save_item():
    """A filter verdict that saves one fact."""
    def build(content, tag="fact", decision="save", confidence=0.95, **extra):
        return {"items": [{
            "content": content, "decision": decision, "tag": tag,
            "reason": extra.get("reason"), "failed_test": extra.get("failed_test"),
            "confidence": confidence,
        }]}
    return build
