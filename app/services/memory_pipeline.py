"""Kivi ingestion pipeline: filter/classify a take, then watch, drop or save each item.

Responsibilities are split into three layers:
  - LLM call        : classify_take()
  - embedding logic : get_embedding_model(), embed(), cosine_similarity(), ...
  - DB writes       : insert_take(), record_ignored(), handle_watch_item(), handle_save_item()
process_take() is the orchestrator that runs the steps in order.
"""
import json
import sqlite3
from typing import Any, Optional

import numpy as np

from app.db import get_connection
from app.services import cache, lexical
from app.services.llm_client import chat_json
from app.services.memory_store import archive_memory
from app.services.retrieval import (
    cosine_similarity,
    deserialize_embedding,
    embed,
    invalidate_cache,
    load_active_memories,
    rank_rows,
    serialize_embedding,
)
from app.services.text_utils import normalize_items, normalize_text


def cached_embed(text: str) -> np.ndarray:
    """embed() with a process-local cache — the same content gets encoded by
    the link step, the conflict shortlist and the insert."""
    hit = cache.get_embedding(text)
    if hit is not None:
        return hit
    vector = embed(text)
    cache.put_embedding(text, vector)
    return vector

WATCHLIST_MATCH_THRESHOLD = 0.85   # same observation seen again
RELATED_THRESHOLD = 0.75           # Step 3: Link

# Conflict detection is semantic, not similarity-based: two facts contradict
# precisely when they fill the same slot with DIFFERENT values, and different
# values push cosine similarity DOWN. Measured on the current encoder, a real
# job change ("works at Acme" -> "works at Google") scores 0.324 while an
# entirely unrelated pair ("works at Google" vs "has a cat named Pixel")
# scores 0.357 — the distributions overlap, so NO absolute cutoff can
# separate them.
#
# The shortlist is therefore rank-based, not threshold-based: take the few
# nearest same-tag memories by dense AND by BM25, and let the model judge.
# BM25 is what catches "works at X" -> "works at Y", since the shared frame
# is lexical even when the values differ. Removing the cutoff also removes a
# constant that silently needed re-tuning whenever the encoder changed.
CONFLICT_CANDIDATE_LIMIT = 8
CONFIDENCE_REINFORCEMENT = 0.1     # preference reinforcement bump
CONFLICT_TAGS = ("fact", "preference")  # episodes accumulate, they don't supersede

FILTER_SYSTEM_PROMPT = """You are Kivi's memory filter. Given a dictation, decide if it contains
something worth remembering about the user. Apply three tests:
1. Is it about the user (not general world knowledge)?
2. Is it likely to matter again later (not a one-time instruction)?
3. Is it safe to surface later?
   This test blocks what KIVI would be INFERRING about the user, not what the
   user chose to tell it. The distinction matters:
   - The user stating something about their own life, feelings, relationships,
     health or beliefs PASSES this test. "I'm in love with someone", "I'm
     stressed about the deadline", "I've started therapy" are self-disclosures.
     The user said them on purpose; refusing to remember them is not safety,
     it is forgetting what they told you.
   - Kivi guessing at an unstated inner state FAILS ("user seems anxious",
     "user is probably unhappy at work") — that is an inference, not a fact.
   - People in the user's own life are part of the user's life. Their
     partner's name, their manager's name, their dog's name, who they are
     meeting — these PASS, because the memory is about the user's
     relationships and commitments, not about a stranger.
   - What FAILS here is someone else's private business that does not involve
     the user: gossip they overheard, a colleague's medical news, a friend's
     secret they were told in confidence.
   Only fail this test for inferences, third-party secrets, or content that
   would genuinely harm the user if surfaced back to them.

If it passes all three clearly, decide SAVE.
If it's borderline or only mentioned once with unclear permanence, decide WATCH.
If it fails any test clearly, decide DROP and give a one-line reason
naming which test failed.

A single dictation may contain zero, one, or multiple memory-worthy items.
Extract each separately.

One sentence often carries BOTH a one-off event and a standing pattern, and you
should extract both rather than choosing between them:
- "Order me a filter coffee, I'm running late again" yields an episode (they
  asked for a filter coffee today) AND a preference (they drink filter coffee
  when running late). Emit two items.
- "Book the usual table at Ananda for Friday" yields an episode (a booking on
  Friday) AND a preference (they have a usual table at Ananda).
When the user asks Kivi to DO something, the request itself still reveals what
they want and how they like it. Capture the preference even though the action
is not yours to take. A request phrased as a command is not a reason to drop it.

If SAVE or WATCH, also classify each item as one of: fact, episode, preference.
- fact: stable info about the user, stays true until contradicted
- episode: something that happened, useful for a window of time. If it has a
  concrete date/time (e.g. "meeting at 5pm today"), its relevance window is
  already known, so decide SAVE rather than WATCH even on first sighting.
- preference: a pattern, should get stronger with repetition, so if this
  is the first time you're seeing this pattern, lean toward WATCH not SAVE

Return strict JSON:
{
  'items': [
    {
      'content': 'the extracted memory content, written as a standalone fact/episode/preference',
      'decision': 'save' | 'watch' | 'drop',
      'tag': 'fact' | 'episode' | 'preference' | null,
      'reason': 'short reason, required if decision is drop',
      'failed_test': 'about_user' | 'likely_to_matter' | 'safe_to_surface' | null,
      'confidence': 0.0 to 1.0
    }
  ]
}"""


# ----------------------------------------------------------------- LLM call

def classify_take(
    raw_text: str,
    app_context: Optional[str] = None,
    answering_question: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Step 2: one Groq call that filters and classifies every item in the dictation."""
    user_message = f"Dictation:\n{raw_text}"
    if app_context:
        user_message += f"\n\nApp context: {app_context}"
    if answering_question:
        # Kivi asked for this. Dropping the answer would be absurd — it would
        # just ask the same question again next time.
        user_message += (
            f'\n\nIMPORTANT: this is the user answering a question Kivi just '
            f'asked them: "{answering_question}"\n'
            "Kivi solicited this information, so it is memory-worthy by "
            "definition. Decide SAVE unless the message is empty or refuses to "
            "answer. Do not drop it for being small, mundane, or about a pet, "
            "object or place — Kivi asked for it, and the subject of the answer "
            "is part of the user's life."
        )

    payload = chat_json(FILTER_SYSTEM_PROMPT, user_message, temperature=0.2).data
    items = payload.get("items", [])
    return items if isinstance(items, list) else []


CONFLICT_SYSTEM_PROMPT = """You maintain a personal memory store about one user. A new memory has just been formed. For each existing memory you are shown, decide its relation to the new one.

- "supersedes": both describe the SAME attribute of the user (employer, job title, city, relationship status, phone, a specific recurring preference) and the new memory reflects a change, correction, or explicit replacement. The old memory is no longer true.
- "refines": the new memory says everything the old one said AND adds detail, so the old one is now a vaguer version of the same fact. "in love with someone" refined by "in love with someone named Meera". The old one is not wrong, just redundant.
- "duplicate": same meaning as the new memory, adding no new information.
- "coexists": about a different attribute, or both can be true at once.

Rules:
- Only supersede when the old memory would now be WRONG to tell the user.
- Different attributes always coexist (employer vs favourite food).
- Two memories about the same attribute with the SAME value are duplicates, not supersessions.
- Prefer "refines" over "coexists" when the new memory strictly contains the old one's information. Keeping both would show the user the same fact twice, once with less detail.
- Be conservative otherwise: when unsure, choose "coexists".

Return strict JSON:
{
  "verdicts": [
    {"memory_id": 1, "relation": "supersedes" | "refines" | "duplicate" | "coexists", "reason": "short"}
  ]
}"""


def detect_conflicts(
    new_content: str, new_tag: str, candidates: list[dict[str, Any]]
) -> dict[int, str]:
    """Ask the model which existing memories the new one supersedes or duplicates.

    Returns {memory_id: relation} for non-coexisting verdicts only.
    """
    if not candidates:
        return {}

    listing = "\n".join(
        f"[memory_id={c['id']}] ({c['tag']}) {c['content']}" for c in candidates
    )
    user_message = (
        f"New memory ({new_tag}): {new_content}\n\nExisting memories:\n{listing}"
    )

    # A conflict check failing must never lose the incoming memory: chat_json
    # returns an empty payload rather than raising.
    payload = chat_json(CONFLICT_SYSTEM_PROMPT, user_message, temperature=0.0, fast=True).data

    offered = {c["id"] for c in candidates}
    verdicts: dict[int, str] = {}
    for verdict in payload.get("verdicts", []) or []:
        try:
            memory_id = int(verdict.get("memory_id"))
        except (TypeError, ValueError):
            continue
        relation = str(verdict.get("relation", "")).lower()
        if memory_id in offered and relation in ("supersedes", "refines", "duplicate"):
            verdicts[memory_id] = relation
    return verdicts


# ----------------------------------------------------------------- DB writes

def insert_take(conn: sqlite3.Connection, raw_text: str, app_context: Optional[str]) -> int:
    """Step 1: persist the raw take before any memory processing."""
    cursor = conn.execute(
        "INSERT INTO takes (raw_text, app_context) VALUES (?, ?)",
        (normalize_text(raw_text), normalize_text(app_context)),
    )
    return int(cursor.lastrowid)


def record_ignored(conn: sqlite3.Connection, take_id: int, item: dict[str, Any]) -> dict[str, Any]:
    """Step 1 Filter -> Drop branch: log to the ignored ledger."""
    content = item.get("content", "")
    reason = item.get("reason") or "no reason given"
    failed_test = item.get("failed_test")
    conn.execute(
        "INSERT INTO ignored (source_take_id, content, reason, failed_test) VALUES (?, ?, ?, ?)",
        (take_id, content, reason, failed_test),
    )
    return {"content": content, "reason": reason, "failed_test": failed_test}


def record_no_items_ignored(conn: sqlite3.Connection, take_id: int, raw_text: str) -> dict[str, Any]:
    """Fallback so every take has an auditable outcome: the classifier extracted
    nothing memory-worthy at all, so log the whole take to the ignored ledger
    rather than leaving it with no trace of why it produced no items."""
    content = normalize_text(raw_text)
    reason = "no memory-worthy content extracted"
    conn.execute(
        "INSERT INTO ignored (source_take_id, content, reason, failed_test) VALUES (?, ?, ?, ?)",
        (take_id, content, reason, None),
    )
    return {"content": content, "reason": reason, "failed_test": None}


def _insert_memory(
    conn: sqlite3.Connection,
    take_id: int,
    content: str,
    tag: str,
    confidence: float,
    related_memory_ids: list[int],
    embedding: np.ndarray,
    promoted_from_watchlist_id: Optional[int] = None,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO memories (
            content, source_take_id, promoted_from_watchlist_id,
            tag, confidence, related_memory_ids, embedding
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            content,
            take_id,
            promoted_from_watchlist_id,
            tag,
            confidence,
            json.dumps(related_memory_ids),
            serialize_embedding(embedding),
        ),
    )
    memory_id = int(cursor.lastrowid)
    _link_back(conn, memory_id, related_memory_ids)
    # Both search paths must learn about the new row immediately: the BM25
    # index by insert, the dense matrix by invalidation.
    lexical.add(conn, memory_id, content)
    invalidate_cache()
    return memory_id


def _link_back(conn: sqlite3.Connection, memory_id: int, related_memory_ids: list[int]) -> None:
    """Make the Step 3 links bidirectional.

    Writing links only on the new row leaves older memories unaware of what
    later connected to them, so the store is a list with back-pointers rather
    than a graph. Linking both ways means any memory can be walked from either
    end, which is what makes 'what else do you know about this?' answerable.
    """
    for other_id in related_memory_ids:
        row = conn.execute(
            "SELECT related_memory_ids FROM memories WHERE id = ?", (other_id,)
        ).fetchone()
        if row is None:
            continue
        try:
            existing = json.loads(row["related_memory_ids"] or "[]")
        except (TypeError, ValueError):
            existing = []
        if memory_id not in existing:
            existing.append(memory_id)
            conn.execute(
                "UPDATE memories SET related_memory_ids = ? WHERE id = ?",
                (json.dumps(sorted(existing)), other_id),
            )


def _conflict_candidates(
    conn: sqlite3.Connection,
    content: str,
    scored: list[tuple[float, sqlite3.Row]],
    tag: str,
) -> list[dict[str, Any]]:
    """The few same-tag memories most likely to be about the same thing.

    Union of the nearest by embedding and the top BM25 hits, because the two
    fail on opposite cases: embeddings miss "Acme -> Google" (different values
    look different) and BM25 misses paraphrase. Ranked, never thresholded.
    """
    same_tag = [(score, row) for score, row in scored if row["tag"] == tag]
    chosen: dict[int, dict[str, Any]] = {}

    for _, row in same_tag[:CONFLICT_CANDIDATE_LIMIT]:
        chosen[int(row["id"])] = {
            "id": int(row["id"]), "content": row["content"], "tag": row["tag"]
        }

    by_id = {int(row["id"]): row for _, row in same_tag}
    for memory_id in lexical.search(conn, content, CONFLICT_CANDIDATE_LIMIT):
        row = by_id.get(memory_id)
        if row is not None and memory_id not in chosen:
            chosen[memory_id] = {
                "id": memory_id, "content": row["content"], "tag": row["tag"]
            }

    return list(chosen.values())[:CONFLICT_CANDIDATE_LIMIT]


def handle_save_item(
    conn: sqlite3.Connection,
    take_id: int,
    item: dict[str, Any],
    promoted_from_watchlist_id: Optional[int] = None,
) -> dict[str, Any]:
    """Steps 3 + 4: link against existing memories, resolve conflicts, then save."""
    content = item.get("content", "")
    tag = item.get("tag") or "fact"
    confidence = float(item.get("confidence", 1.0))
    embedding = cached_embed(content)

    rows = load_active_memories(conn)
    scored = rank_rows(rows, embedding)

    related_memory_ids = [int(row["id"]) for score, row in scored if score > RELATED_THRESHOLD]

    candidates = _conflict_candidates(conn, content, scored, tag)

    verdicts: dict[int, str] = {}
    if tag in CONFLICT_TAGS and candidates:
        verdicts = detect_conflicts(content, tag, candidates)

    superseded_ids = [mid for mid, relation in verdicts.items() if relation == "supersedes"]
    refined_ids = [mid for mid, relation in verdicts.items() if relation == "refines"]
    duplicate_ids = [mid for mid, relation in verdicts.items() if relation == "duplicate"]

    # Saying the same thing again adds no row; it just strengthens what's there.
    if duplicate_ids and not superseded_ids and not refined_ids:
        existing_id = duplicate_ids[0]
        existing = conn.execute(
            "SELECT id, content, tag, confidence FROM memories WHERE id = ?", (existing_id,)
        ).fetchone()
        if existing is not None:
            new_confidence = min(1.0, float(existing["confidence"]) + CONFIDENCE_REINFORCEMENT)
            conn.execute(
                "UPDATE memories SET confidence = ? WHERE id = ?", (new_confidence, existing_id)
            )
            return {
                "memory_id": existing_id,
                "content": existing["content"],
                "tag": existing["tag"],
                "confidence": new_confidence,
                "related_memory_ids": related_memory_ids,
                "action": "reinforced_existing",
            }

    retired_ids = superseded_ids + refined_ids
    related_memory_ids = [i for i in related_memory_ids if i not in retired_ids]
    memory_id = _insert_memory(
        conn, take_id, content, tag, confidence,
        related_memory_ids, embedding, promoted_from_watchlist_id,
    )

    # The superseded memories are retired, not deleted: Memory shows current
    # truth, the archive keeps the history of what changed and why.
    for old_id in superseded_ids:
        archive_memory(
            conn, old_id,
            retired_reason="contradicted",
            contradicted_by_memory_id=memory_id,
        )
    # Retired for redundancy, not for being wrong — the distinction matters
    # when reading the archive back.
    for old_id in refined_ids:
        archive_memory(
            conn, old_id,
            retired_reason="refined",
            contradicted_by_memory_id=memory_id,
        )

    return {
        "memory_id": memory_id,
        "content": content,
        "tag": tag,
        "confidence": confidence,
        "related_memory_ids": related_memory_ids,
        "action": ("replaced_contradicted" if superseded_ids
                   else "refined_earlier" if refined_ids else "created"),
        "superseded_memory_ids": superseded_ids,
        "refined_memory_ids": refined_ids,
        "promoted_from_watchlist_id": promoted_from_watchlist_id,
    }


def handle_watch_item(
    conn: sqlite3.Connection, take_id: int, item: dict[str, Any]
) -> dict[str, Any]:
    """Step 1 Filter -> Watch branch, with promotion on repetition."""
    content = item.get("content", "")
    tag = item.get("tag")
    embedding = cached_embed(content)

    match: Optional[sqlite3.Row] = None
    match_score = 0.0
    watching = conn.execute(
        "SELECT id, content, embedding, tag, times_seen, promote_threshold "
        "FROM watchlist WHERE status = 'watching'"
    ).fetchall()
    for row in watching:
        if row["embedding"] is None:
            continue
        score = cosine_similarity(embedding, deserialize_embedding(row["embedding"]))
        if score > WATCHLIST_MATCH_THRESHOLD and score > match_score:
            match, match_score = row, score

    if match is None:
        cursor = conn.execute(
            "INSERT INTO watchlist (content, source_take_id, embedding, tag) VALUES (?, ?, ?, ?)",
            (content, take_id, serialize_embedding(embedding), tag),
        )
        watchlist_id = int(cursor.lastrowid)
        row = conn.execute(
            "SELECT times_seen, promote_threshold FROM watchlist WHERE id = ?", (watchlist_id,)
        ).fetchone()
        return {
            "watchlist_id": watchlist_id,
            "content": content,
            "tag": tag,
            "times_seen": int(row["times_seen"]),
            "promote_threshold": int(row["promote_threshold"]),
            "promoted": False,
            "action": "new_watch",
        }

    # Seen before: count it again.
    watchlist_id = int(match["id"])
    times_seen = int(match["times_seen"]) + 1
    promote_threshold = int(match["promote_threshold"])
    conn.execute(
        "UPDATE watchlist SET times_seen = ?, last_seen_at = CURRENT_TIMESTAMP WHERE id = ?",
        (times_seen, watchlist_id),
    )

    result = {
        "watchlist_id": watchlist_id,
        "content": match["content"],
        "tag": tag,
        "times_seen": times_seen,
        "promote_threshold": promote_threshold,
        "promoted": False,
        "action": "repeat_seen",
        "similarity": round(match_score, 4),
    }

    if times_seen >= promote_threshold:
        promotion_item = {
            "content": match["content"],
            "tag": match["tag"] or tag or "fact",
            "confidence": float(item.get("confidence", 1.0)),
        }
        saved = handle_save_item(
            conn, take_id, promotion_item, promoted_from_watchlist_id=watchlist_id
        )
        conn.execute("UPDATE watchlist SET status = 'promoted' WHERE id = ?", (watchlist_id,))
        result["promoted"] = True
        result["action"] = "promoted"
        result["memory"] = saved

    return result


# --------------------------------------------------------------- orchestrator

def process_take(
    raw_text: str,
    app_context: Optional[str] = None,
    answering_question: Optional[str] = None,
) -> dict[str, Any]:
    """Run the full ingestion pipeline for one dictation."""
    conn = get_connection()
    try:
        take_id = insert_take(conn, raw_text, app_context)
        items = normalize_items(
            classify_take(raw_text, app_context, answering_question)
        )

        saved: list[dict[str, Any]] = []
        watched: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []

        if not items:
            dropped.append(record_no_items_ignored(conn, take_id, raw_text))

        for item in items:
            decision = str(item.get("decision", "")).lower()
            if decision == "drop":
                dropped.append(record_ignored(conn, take_id, item))
            elif decision == "watch":
                result = handle_watch_item(conn, take_id, item)
                watched.append(result)
                if result.get("promoted"):
                    saved.append(result["memory"])
            elif decision == "save":
                saved.append(handle_save_item(conn, take_id, item))

        conn.commit()
        return {"take_id": take_id, "saved": saved, "watched": watched, "dropped": dropped}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
