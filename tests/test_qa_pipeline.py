"""Recall behaviour, pinned.

This file exists because the guarantee Kivi is sold on — that nothing is ever
presented as your memory unless a memory backs it — lived entirely in a prompt
and a branch, with no test. A prompt is not a guarantee until something fails
when it stops holding.
"""
import pytest


def _seed(db, contents):
    """Insert memories directly, bypassing the LLM filter."""
    from app.db import get_connection
    from app.services import lexical
    from app.services.retrieval import (
        embed, invalidate_cache, serialize_embedding,
    )

    conn = get_connection()
    try:
        conn.execute("INSERT INTO takes (raw_text) VALUES ('seed')")
        take_id = conn.execute("SELECT MAX(id) m FROM takes").fetchone()["m"]
        lexical.ensure_index(conn)
        for content, tag in contents:
            cursor = conn.execute(
                "INSERT INTO memories (content, source_take_id, tag, confidence, embedding) "
                "VALUES (?, ?, ?, 1.0, ?)",
                (content, take_id, tag, serialize_embedding(embed(content))),
            )
            lexical.add(conn, int(cursor.lastrowid), content)
        conn.commit()
        invalidate_cache()
    finally:
        conn.close()


@pytest.fixture()
def qa_llm(monkeypatch, llm):
    """The scripted LLM, also patched into qa_pipeline."""
    import app.services.qa_pipeline as qa

    monkeypatch.setattr(qa, "chat_json", llm)
    return llm


def _route(intent="question", scope="personal", resolved=None, reaction="", live=False):
    return {
        "intent": intent, "scope": scope, "needs_live_data": live,
        "resolved_query": resolved or "q", "reaction": reaction,
    }


# ------------------------------------------------------- the citation gate

def test_ungrounded_prose_is_discarded_not_shown(db, qa_llm):
    """THE core guarantee. The model writes something plausible but cites no
    memory; the user must see the decline, never the prose."""
    from app.services import qa_pipeline

    _seed(db, [("The user works at Zoho", "fact")])
    qa_llm.push({"answer": "You probably work in tech somewhere.",
                 "used_memory_ids": [], "gap": None}, qa_llm.OTHER)

    result = qa_pipeline.answer_question("Where do I work?", _route(resolved="Where do I work?"))

    assert result["decision"] == "declined_unknown"
    assert result["answer_source"] == "none"
    assert "probably work in tech" not in result["answer"]
    assert result["answer"] in qa_pipeline.DECLINE_SENTENCES
    assert result["cited_memories"] == []


def test_a_cited_answer_is_shown_with_its_citation(db, qa_llm):
    from app.services import qa_pipeline

    _seed(db, [("The user works at Zoho", "fact")])
    qa_llm.push({"answer": "You work at Zoho.", "used_memory_ids": [1], "gap": None},
                qa_llm.OTHER)

    result = qa_pipeline.answer_question("Where do I work?", _route(resolved="Where do I work?"))

    assert result["decision"] == "answered"
    assert result["answer_source"] == "memory"
    assert result["answer"] == "You work at Zoho."
    assert [m["id"] for m in result["cited_memories"]] == [1]


def test_model_cannot_cite_a_memory_it_was_never_shown(db, qa_llm):
    """A hallucinated id must not become a citation."""
    from app.services import qa_pipeline

    _seed(db, [("The user works at Zoho", "fact")])
    qa_llm.push({"answer": "You work at Zoho.", "used_memory_ids": [1, 999], "gap": None},
                qa_llm.OTHER)

    result = qa_pipeline.answer_question("Where do I work?", _route(resolved="Where do I work?"))
    assert [m["id"] for m in result["cited_memories"]] == [1]


def test_empty_store_declines_without_generating_an_answer(db, qa_llm):
    """No answer-generation call when there is nothing to ground one in.

    It does still ask what it should learn — for a new user that question IS
    the onboarding — but it never spends the expensive call trying to answer
    from an empty store.
    """
    from app.services import qa_pipeline

    result = qa_pipeline.answer_question("Where do I work?", _route(resolved="Where do I work?"))

    assert result["decision"] == "declined_unknown"
    assert result["cited_memories"] == []
    answer_calls = [
        system for system, _ in qa_llm.calls
        if "Answer using ONLY the memories" in system or "your own knowledge" in system
    ]
    assert answer_calls == []


# -------------------------------------------------------------- follow-ups

def test_a_named_gap_triggers_follow_up_questions(db, qa_llm):
    """Answering "I don't know their name" is exactly when to ask."""
    from app.services import qa_pipeline

    _seed(db, [("The user is in love with someone", "fact")])
    qa_llm.push({"answer": "I don't know their name.", "used_memory_ids": [1],
                 "gap": "the name of the person they love"}, qa_llm.OTHER)
    qa_llm.push({"questions": ["What is their name?", "How did you meet?"]}, qa_llm.OTHER)

    result = qa_pipeline.answer_question("What is my partner's name?",
                                         _route(resolved="What is my partner's name?"))

    assert result["decision"] == "answered"
    assert result["follow_ups"] == ["What is their name?", "How did you meet?"]


# ------------------------------------------------------------------ routing

def test_router_falls_back_to_question_when_the_model_fails(qa_llm):
    """Reading memory is the safe default; writing something unintended is not."""
    from app.services import qa_pipeline

    route = qa_pipeline.route_input("something ambiguous", "")
    assert route["intent"] == "question"
    assert route["resolved_query"] == "something ambiguous"


def test_router_carries_the_resolved_query(qa_llm):
    from app.services import qa_pipeline

    qa_llm.push({"intent": "question", "scope": "personal", "needs_live_data": False,
                 "resolved_query": "What colour is Momo the rabbit?"}, qa_llm.OTHER)
    route = qa_pipeline.route_input("what colour is she?", "User: I adopted Momo")
    assert route["resolved_query"] == "What colour is Momo the rabbit?"


def test_retrieval_uses_the_resolved_query_not_the_pronoun(db, qa_llm):
    """Searching "what colour is she?" ranked the wrong pet top. Retrieval and
    generation must both see the resolved form."""
    from app.services import qa_pipeline

    _seed(db, [("Pixel the cat is black and white", "fact"),
               ("Momo the rabbit is grey", "fact")])
    qa_llm.push({"answer": "Momo is grey.", "used_memory_ids": [2], "gap": None},
                qa_llm.OTHER)

    qa_pipeline.answer_question("what colour is she?",
                                _route(resolved="What colour is Momo the rabbit?"))

    _, user_prompt = qa_llm.calls[0]
    assert "Momo" in user_prompt


# --------------------------------------------------------------- fast paths

def test_trivial_input_short_circuits_before_any_model_call(db, qa_llm):
    from app.services import qa_pipeline

    result = qa_pipeline.process_hey_kivi("thanks")
    assert result["fast_path"] == "ack"
    assert qa_llm.calls == []


def test_an_answer_to_a_question_is_never_treated_as_trivial(db, qa_llm):
    """"Meera" is a real fact when Kivi asked for a name."""
    from app.services import qa_pipeline

    qa_llm.push({"intent": "statement", "scope": "personal", "needs_live_data": False,
                 "resolved_query": "The user's partner is called Meera"}, qa_llm.OTHER)
    qa_llm.push({"items": [{"content": "The user's partner is called Meera",
                            "decision": "save", "tag": "fact", "confidence": 0.9}]},
                qa_llm.FILTER)

    result = qa_pipeline.process_hey_kivi("Meera", answering="What is their name?")
    assert result["mode"] == "remember"
    assert result["ingested"]["saved"]


# ------------------------------------------------------------------ persona

def test_persona_reaction_replaces_the_canned_acknowledgement(db, qa_llm):
    from app.services import qa_pipeline

    qa_llm.push({"items": [{"content": "The user got promoted", "decision": "save",
                            "tag": "episode", "confidence": 0.9}]}, qa_llm.FILTER)
    result = qa_pipeline.remember_statement("I got promoted", reaction="Aiyo, congratulations!")
    assert result["answer"] == "Aiyo, congratulations!"


def test_neutral_persona_suppresses_the_reaction(db, qa_llm, monkeypatch):
    """KIVI_PERSONA=neutral must actually turn personality off."""
    from app.services import qa_pipeline

    monkeypatch.setenv("KIVI_PERSONA", "neutral")
    qa_llm.push({"items": [{"content": "The user got promoted", "decision": "save",
                            "tag": "episode", "confidence": 0.9}]}, qa_llm.FILTER)
    result = qa_pipeline.remember_statement("I got promoted", reaction="Aiyo, congratulations!")
    assert result["answer"] != "Aiyo, congratulations!"


# ------------------------------------------------------------- consolidation

def test_consolidation_collapses_restatements_of_one_fact(db, qa_llm):
    """The three-way case that shipped: one fact stored three ways."""
    from app.services import consolidate as consolidate_module
    import app.services.consolidate as c

    _seed(db, [
        ("User will not tell the name of the person they love until committed", "fact"),
        ("The user is in love with someone whose identity they have not disclosed", "fact"),
        ("The user is in love with someone, but it is not yet confirmed", "fact"),
    ])
    qa_llm.push({"groups": [{"keep": 1, "remove": [2, 3], "reason": "same fact restated"}]},
                qa_llm.OTHER)
    import app.services.consolidate as consolidate_mod
    consolidate_mod.chat_json = qa_llm

    result = consolidate_module.consolidate()

    assert result["removed"] == 2
    assert result["active_after"] == 1


def test_consolidation_dry_run_changes_nothing(db, qa_llm):
    from app.services import consolidate as consolidate_module
    import app.services.consolidate as consolidate_mod

    _seed(db, [("The user works at Zoho", "fact"), ("The user is employed by Zoho", "fact")])
    qa_llm.push({"groups": [{"keep": 1, "remove": [2], "reason": "same"}]}, qa_llm.OTHER)
    consolidate_mod.chat_json = qa_llm

    result = consolidate_module.consolidate(dry_run=True)

    assert result["dry_run"] is True
    assert result["active_after"] == result["active_before"] == 2
