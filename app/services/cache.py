"""Small in-process caches for the two things Kivi repeats most.

Embeddings are deterministic for a given string and model, so re-encoding the
same text is pure waste — and the ingestion path encodes the same content
several times across linking, conflict detection and storage.

Answers are cached only for repeated identical questions inside a short window,
which is a real pattern (a user rephrasing, or re-asking after a refresh) and
is bounded so a changed memory store cannot serve a stale answer for long.
"""
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

import numpy as np

EMBEDDING_CACHE_SIZE = 512
ANSWER_CACHE_SIZE = 64
# Short by design: memories change, and a minute-old answer is defensible
# while an hour-old one is a bug.
ANSWER_TTL_SECONDS = 60

_lock = threading.Lock()
_embeddings: "OrderedDict[str, np.ndarray]" = OrderedDict()
_answers: "OrderedDict[str, tuple[float, int, dict[str, Any]]]" = OrderedDict()

_stats = {"embedding_hits": 0, "embedding_misses": 0, "answer_hits": 0, "answer_misses": 0}


def get_embedding(key: str) -> Optional[np.ndarray]:
    with _lock:
        vector = _embeddings.get(key)
        if vector is None:
            _stats["embedding_misses"] += 1
            return None
        _embeddings.move_to_end(key)
        _stats["embedding_hits"] += 1
        return vector


def put_embedding(key: str, vector: np.ndarray) -> None:
    with _lock:
        _embeddings[key] = vector
        _embeddings.move_to_end(key)
        while len(_embeddings) > EMBEDDING_CACHE_SIZE:
            _embeddings.popitem(last=False)


def get_answer(key: str, memory_version: int) -> Optional[dict[str, Any]]:
    """Only a hit if it is fresh AND the memory store hasn't changed since."""
    with _lock:
        entry = _answers.get(key)
        if entry is None:
            _stats["answer_misses"] += 1
            return None
        stored_at, version, payload = entry
        if version != memory_version or time.time() - stored_at > ANSWER_TTL_SECONDS:
            _answers.pop(key, None)
            _stats["answer_misses"] += 1
            return None
        _answers.move_to_end(key)
        _stats["answer_hits"] += 1
        return payload


def put_answer(key: str, memory_version: int, payload: dict[str, Any]) -> None:
    with _lock:
        _answers[key] = (time.time(), memory_version, payload)
        _answers.move_to_end(key)
        while len(_answers) > ANSWER_CACHE_SIZE:
            _answers.popitem(last=False)


def clear() -> None:
    with _lock:
        _embeddings.clear()
        _answers.clear()


def stats() -> dict[str, Any]:
    with _lock:
        total_embed = _stats["embedding_hits"] + _stats["embedding_misses"]
        total_answer = _stats["answer_hits"] + _stats["answer_misses"]
        return {
            **_stats,
            "embedding_hit_rate": round(_stats["embedding_hits"] / total_embed, 3) if total_embed else 0.0,
            "answer_hit_rate": round(_stats["answer_hits"] / total_answer, 3) if total_answer else 0.0,
            "embeddings_cached": len(_embeddings),
            "answers_cached": len(_answers),
        }
