"""Retrieval, caching and the no-LLM fast paths."""
import numpy as np
import pytest


# ------------------------------------------------------------- fast paths

@pytest.mark.parametrize("text,kind", [
    ("ok", "ack"), ("thanks", "ack"), ("Thanks!", "ack"),
    ("hi", "greeting"), ("hey kivi", "greeting"), ("vanakkam", "greeting"),
    ("namaste", "greeting"), ("bye", "farewell"), ("what can you do?", "capability"),
    ("", "empty"), ("   ", "empty"),
])
def test_trivial_input_never_reaches_a_model(text, kind):
    from app.services import fast_paths

    result = fast_paths.check(text)
    assert result is not None and result["kind"] == kind


@pytest.mark.parametrize("text", [
    "I work at Google",
    "ok so I moved to Bangalore last week",
    "what colour is Pixel?",
    "thanks for remembering my dentist appointment on Tuesday",
])
def test_real_content_falls_through_to_the_pipeline(text):
    """The fast path must never swallow something worth remembering."""
    from app.services import fast_paths

    assert fast_paths.check(text) is None


# --------------------------------------------------------------- lexical

def test_bm25_finds_names_that_embeddings_blur(db):
    """Proper nouns are exactly where dense retrieval is weakest."""
    from app.db import get_connection
    from app.services import lexical

    conn = get_connection()
    try:
        lexical.ensure_index(conn)
        lexical.add(conn, 1, "The user adopted a cat named Pixel")
        lexical.add(conn, 2, "The user adopted a rabbit named Momo")
        conn.commit()

        assert lexical.search(conn, "What colour is Momo?") == [2]
        assert lexical.search(conn, "Tell me about Pixel") == [1]
    finally:
        conn.close()


def test_lexical_query_cannot_be_broken_by_punctuation(db):
    """User text is data; FTS5 operators in it must not blow up the query."""
    from app.db import get_connection
    from app.services import lexical

    conn = get_connection()
    try:
        lexical.ensure_index(conn)
        lexical.add(conn, 1, "The user works at Zoho")
        conn.commit()
        for hostile in ['Zoho AND (', 'NEAR("', '"unbalanced', "* OR *", "()"]:
            assert isinstance(lexical.search(conn, hostile), list)
    finally:
        conn.close()


def test_removing_a_memory_drops_it_from_lexical_search(db):
    from app.db import get_connection
    from app.services import lexical

    conn = get_connection()
    try:
        lexical.ensure_index(conn)
        lexical.add(conn, 1, "The user works at Acme")
        conn.commit()
        assert lexical.search(conn, "Acme") == [1]
        lexical.remove(conn, 1)
        conn.commit()
        assert lexical.search(conn, "Acme") == []
    finally:
        conn.close()


# ----------------------------------------------------------------- cache

def test_answer_cache_is_invalidated_when_memories_change():
    from app.services import cache

    cache.clear()
    payload = {"answer": "You work at Zoho"}
    cache.put_answer("where do i work", 111, payload)

    assert cache.get_answer("where do i work", 111) == payload
    # A different memory version means the store changed underneath it.
    assert cache.get_answer("where do i work", 222) is None


def test_embedding_cache_round_trips():
    from app.services import cache

    cache.clear()
    vector = np.ones(384, dtype=np.float32)
    assert cache.get_embedding("hello") is None
    cache.put_embedding("hello", vector)
    assert np.array_equal(cache.get_embedding("hello"), vector)


def test_embedding_cache_evicts_and_does_not_grow_without_bound():
    from app.services import cache

    cache.clear()
    for i in range(cache.EMBEDDING_CACHE_SIZE + 50):
        cache.put_embedding(f"text-{i}", np.zeros(4, dtype=np.float32))
    assert cache.stats()["embeddings_cached"] <= cache.EMBEDDING_CACHE_SIZE


# ------------------------------------------------------------ text utils

def test_normalisation_strips_model_whitespace_artifacts():
    from app.services.text_utils import normalize_text

    assert normalize_text("5 pm") == "5 pm"
    assert normalize_text("a b") == "a b"
    assert normalize_text(None) is None
