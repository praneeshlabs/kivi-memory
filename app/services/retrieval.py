"""Retrieval: hybrid dense + lexical search over the memory store.

Three problems are solved here that a plain vector scan does not solve.

1. Quality. Dense similarity alone scored ~0.36 on queries it answered
   correctly — weak absolute signal, and embeddings are specifically bad at
   proper nouns ("Pixel", "Meera", "Infosys") because a name carries little
   distributional meaning. BM25 is excellent at exactly those. The two are
   fused by Reciprocal Rank Fusion, which needs no score calibration between
   them.

2. Scale. Loading every embedding from SQLite on every query is a table scan
   per message. The matrix is cached in process and invalidated on write, so
   a query is one matmul against memory-resident floats.

3. Language. A multilingual encoder means Tamil, Hindi and code-mixed text
   retrieve against English memories and vice versa, which an English-only
   model cannot do at all.
"""
import sqlite3
import threading
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.services import lexical

# Multilingual by design: Kivi is built for Indian users, who do not speak one
# language at a time. Same 384 dimensions as the English-only model it replaced.
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# float16 halves storage; the ~0.001 precision loss is far below anything that
# reorders a ranking.
EMBEDDING_DTYPE = np.float16

# Reciprocal Rank Fusion constant. 60 is the value from the original paper and
# is deliberately large so no single engine dominates on rank alone.
RRF_K = 60

_embedding_model: Optional[SentenceTransformer] = None
_model_lock = threading.Lock()

# In-process cache of the active memory matrix.
_cache_lock = threading.Lock()
_cached_ids: Optional[list[int]] = None
_cached_matrix: Optional[np.ndarray] = None
_cached_rows: Optional[list[sqlite3.Row]] = None
_cache_dirty = True


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        with _model_lock:
            if _embedding_model is None:
                _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def warm_up() -> None:
    """Load the model at startup so the first real request isn't paying for it."""
    get_embedding_model().encode("warm up", normalize_embeddings=True)


def embed(text: str) -> np.ndarray:
    vector = get_embedding_model().encode(text, normalize_embeddings=True)
    return np.asarray(vector, dtype=np.float32)


def cached_embed_safe(text: str) -> np.ndarray:
    """embed() behind the shared cache, importable without a circular import."""
    from app.services import cache

    hit = cache.get_embedding(text)
    if hit is not None:
        return hit
    vector = embed(text)
    cache.put_embedding(text, vector)
    return vector


def embed_batch(texts: list[str]) -> np.ndarray:
    """One forward pass for many texts — far cheaper than a loop."""
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    vectors = get_embedding_model().encode(texts, normalize_embeddings=True, batch_size=32)
    return np.asarray(vectors, dtype=np.float32)


def serialize_embedding(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=EMBEDDING_DTYPE).tobytes()


def deserialize_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=EMBEDDING_DTYPE).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator == 0.0 else float(np.dot(a, b) / denominator)


# ----------------------------------------------------------------- the cache

def invalidate_cache() -> None:
    """Called by every write path. Cheap, and the only correctness requirement."""
    global _cache_dirty
    with _cache_lock:
        _cache_dirty = True


def _load_cache(conn: sqlite3.Connection) -> tuple[list[sqlite3.Row], np.ndarray]:
    global _cached_ids, _cached_matrix, _cached_rows, _cache_dirty
    with _cache_lock:
        if not _cache_dirty and _cached_rows is not None and _cached_matrix is not None:
            return _cached_rows, _cached_matrix

        rows = conn.execute(
            """
            SELECT m.id, m.content, m.tag, m.confidence, m.embedding, m.source_take_id,
                   t.created_at AS source_take_created_at
              FROM memories m
              JOIN takes t ON t.id = m.source_take_id
             WHERE m.status = 'active'
             ORDER BY m.id
            """
        ).fetchall()

        if rows:
            matrix = np.vstack([deserialize_embedding(row["embedding"]) for row in rows])
            norms = np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
            matrix = matrix / norms
        else:
            matrix = np.empty((0, 384), dtype=np.float32)

        _cached_rows, _cached_matrix = rows, matrix
        _cached_ids = [int(row["id"]) for row in rows]
        _cache_dirty = False
        return rows, matrix


def cache_size() -> int:
    return 0 if _cached_rows is None else len(_cached_rows)


# ------------------------------------------------------------------ searching

def load_active_memories(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    rows, _ = _load_cache(conn)
    return list(rows)


def load_active_memories_with_takes(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return load_active_memories(conn)


def score_against(rows: list[sqlite3.Row], query_embedding: np.ndarray) -> np.ndarray:
    if not rows:
        return np.empty(0, dtype=np.float32)
    matrix = np.vstack([deserialize_embedding(row["embedding"]) for row in rows])
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    query = query_embedding / max(float(np.linalg.norm(query_embedding)), 1e-12)
    return matrix @ query


def rank_rows(
    rows: list[sqlite3.Row], query_embedding: np.ndarray, top_k: Optional[int] = None
) -> list[tuple[float, sqlite3.Row]]:
    scores = score_against(rows, query_embedding)
    order = np.argsort(-scores)
    if top_k is not None:
        order = order[:top_k]
    return [(float(scores[i]), rows[i]) for i in order]


def dense_search(
    conn: sqlite3.Connection, query_embedding: np.ndarray, top_k: int
) -> list[tuple[float, sqlite3.Row]]:
    """Cosine ranking against the cached matrix — one matmul, no DB read."""
    rows, matrix = _load_cache(conn)
    if not rows:
        return []
    query = query_embedding / max(float(np.linalg.norm(query_embedding)), 1e-12)
    scores = matrix @ query
    order = np.argsort(-scores)[:top_k]
    return [(float(scores[i]), rows[i]) for i in order]


def hybrid_search(
    conn: sqlite3.Connection, query: str, query_embedding: np.ndarray, top_k: int = 5
) -> list[tuple[float, sqlite3.Row]]:
    """Dense and BM25 rankings fused by Reciprocal Rank Fusion.

    RRF scores by position rather than raw score, so the two engines' wildly
    different scales never need calibrating: a memory ranked well by either
    method surfaces, and one ranked well by both surfaces higher.
    """
    rows, _ = _load_cache(conn)
    if not rows:
        return []

    by_id = {int(row["id"]): row for row in rows}
    # Over-fetch from each engine so fusion has something to work with.
    depth = max(top_k * 4, 20)

    dense_ranked = dense_search(conn, query_embedding, depth)
    dense_scores = {int(row["id"]): score for score, row in dense_ranked}

    fused: dict[int, float] = {}
    for rank, (_, row) in enumerate(dense_ranked):
        fused[int(row["id"])] = fused.get(int(row["id"]), 0.0) + 1.0 / (RRF_K + rank + 1)

    for rank, memory_id in enumerate(lexical.search(conn, query, depth)):
        if memory_id in by_id:
            fused[memory_id] = fused.get(memory_id, 0.0) + 1.0 / (RRF_K + rank + 1)

    ordered = sorted(fused.items(), key=lambda item: -item[1])[:top_k]
    # Report the dense cosine as the surfaced score: it is the interpretable
    # one, and the decision layer is already calibrated against it.
    return [(dense_scores.get(mid, 0.0), by_id[mid]) for mid, _ in ordered if mid in by_id]


def rank_memories_by_similarity(
    conn: sqlite3.Connection, query_embedding: np.ndarray, top_k: int = 5
) -> list[tuple[float, sqlite3.Row]]:
    return dense_search(conn, query_embedding, top_k)
