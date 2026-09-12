"""Ingestion behaviour, pinned.

Every case here is a bug that actually shipped during development and was
caught by a screenshot rather than a test. That is exactly the class of
regression this file exists to prevent.
"""
import sqlite3

import pytest


def _rows(db_path, sql):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql)]
    finally:
        conn.close()


def test_saves_a_plain_fact(db, llm, save_item):
    from app.services.memory_pipeline import process_take

    llm.push(save_item("The user works at Zoho"))
    result = process_take("I work at Zoho")

    assert len(result["saved"]) == 1
    assert result["saved"][0]["tag"] == "fact"
    assert _rows(db, "SELECT * FROM memories WHERE status='active'")


def test_drop_is_logged_with_a_reason(db, llm, save_item):
    from app.services.memory_pipeline import process_take

    llm.push(save_item("The sky is blue", decision="drop",
                       reason="not about user", failed_test="about_user"))
    result = process_take("The sky is really blue today")

    assert result["saved"] == []
    assert result["dropped"][0]["failed_test"] == "about_user"
    assert len(_rows(db, "SELECT * FROM ignored")) == 1


def test_empty_extraction_still_leaves_a_trace(db, llm):
    """Every take must have an auditable outcome, even a boring one."""
    from app.services.memory_pipeline import process_take

    llm.push({"items": []})
    result = process_take("hmm okay then")

    ignored = _rows(db, "SELECT * FROM ignored")
    assert len(ignored) == 1
    assert ignored[0]["reason"] == "no memory-worthy content extracted"
    assert result["dropped"][0]["failed_test"] is None


def test_preference_watches_before_promoting(db, llm, save_item):
    from app.services.memory_pipeline import process_take

    llm.push(save_item("User drinks filter coffee when late",
                       tag="preference", decision="watch"))
    first = process_take("filter coffee again, running late")
    assert first["watched"][0]["times_seen"] == 1
    assert first["saved"] == []

    llm.push(save_item("User drinks filter coffee when late",
                       tag="preference", decision="watch"))
    second = process_take("filter coffee, late again")
    assert second["watched"][0]["promoted"] is True
    assert len(second["saved"]) == 1


def test_contradiction_archives_the_old_fact(db, llm, save_item):
    """A job change scores ~0.52 similarity — far below any duplicate
    threshold — so this must be decided semantically, not by distance."""
    from app.services.memory_pipeline import process_take

    llm.push(save_item("The user works at Acme"))
    process_take("I work at Acme")

    old_id = _rows(db, "SELECT id FROM memories")[0]["id"]
    llm.push(save_item("The user works at Google")).push_conflict(
        {"verdicts": [{"memory_id": old_id, "relation": "supersedes", "reason": "changed job"}]}
    )
    result = process_take("I moved to Google")

    assert result["saved"][0]["superseded_memory_ids"] == [old_id]
    active = _rows(db, "SELECT content FROM memories WHERE status='active'")
    assert len(active) == 1 and "Google" in active[0]["content"]
    assert _rows(db, "SELECT * FROM memory_archive")[0]["retired_reason"] == "contradicted"


def test_refinement_retires_the_vaguer_memory(db, llm, save_item):
    from app.services.memory_pipeline import process_take

    llm.push(save_item("The user is in love with someone"))
    process_take("I'm in love with someone")
    old_id = _rows(db, "SELECT id FROM memories")[0]["id"]

    llm.push(save_item("The user is in love with someone named Meera")).push_conflict(
        {"verdicts": [{"memory_id": old_id, "relation": "refines", "reason": "adds the name"}]}
    )
    result = process_take("she is called Meera")

    assert result["saved"][0]["refined_memory_ids"] == [old_id]
    assert len(_rows(db, "SELECT id FROM memories WHERE status='active'")) == 1
    assert _rows(db, "SELECT * FROM memory_archive")[0]["retired_reason"] == "refined"


def test_duplicate_reinforces_instead_of_adding_a_row(db, llm, save_item):
    from app.services.memory_pipeline import process_take

    llm.push(save_item("The user works at Zoho", confidence=0.8))
    process_take("I work at Zoho")
    existing = _rows(db, "SELECT id, confidence FROM memories")[0]

    llm.push(save_item("The user works at Zoho")).push_conflict(
        {"verdicts": [{"memory_id": existing["id"], "relation": "duplicate", "reason": "same"}]}
    )
    result = process_take("I'm at Zoho")

    assert result["saved"][0]["action"] == "reinforced_existing"
    rows = _rows(db, "SELECT id, confidence FROM memories")
    assert len(rows) == 1
    assert rows[0]["confidence"] > existing["confidence"]


def test_conflict_check_failure_never_loses_the_memory(db, llm, save_item):
    """If the conflict model errors or returns junk, the memory still saves."""
    from app.services.memory_pipeline import process_take

    llm.push(save_item("The user has a dentist appointment"))
    result = process_take("dentist on Tuesday")

    assert len(result["saved"]) == 1
    assert _rows(db, "SELECT * FROM memories WHERE status='active'")


def test_unicode_whitespace_is_normalised_before_storage(db, llm, save_item):
    """LLMs emit U+202F; it renders as mojibake and breaks exact matching."""
    from app.services.memory_pipeline import process_take

    llm.push(save_item("Meeting at 5 pm with Priya"))
    process_take("meeting at 5 pm")

    content = _rows(db, "SELECT content FROM memories")[0]["content"]
    assert " " not in content and " " not in content
    assert "5 pm" in content


def test_solicited_answer_is_marked_for_the_filter(db, llm, save_item):
    """When Kivi asked the question, the filter must be told so — otherwise it
    re-judges the answer generically and drops it."""
    from app.services.memory_pipeline import process_take

    llm.push(save_item("Pixel is grey"))
    process_take("she's grey", answering_question="What colour is Pixel?")

    _, user_prompt = llm.calls[0]
    assert "What colour is Pixel?" in user_prompt
    assert "solicited" in user_prompt.lower()


def test_links_are_bidirectional(db, llm, save_item):
    from app.services.memory_pipeline import process_take
    import json

    llm.push(save_item("The user lives in Chennai and works there"))
    process_take("I live in Chennai")
    llm.push(save_item("The user lives in Chennai near the office"))
    process_take("my place in Chennai is near the office")

    rows = _rows(db, "SELECT id, related_memory_ids FROM memories WHERE status='active'")
    linked = [r for r in rows if json.loads(r["related_memory_ids"])]
    # Whatever links exist must point both ways.
    for row in linked:
        for other_id in json.loads(row["related_memory_ids"]):
            other = [r for r in rows if r["id"] == other_id]
            if other:
                assert row["id"] in json.loads(other[0]["related_memory_ids"])
