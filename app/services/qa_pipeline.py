"""Hey Kivi: Step 6 Retrieve -> Step 7 Decide -> Step 8 Answer.

Kivi answers from three places and never blurs which one it used:
  - memory  : the user's own recorded facts, always citation-gated
  - general : the model's world knowledge, labelled as not from memory
  - live    : a web lookup, only when a search key is configured

Keeping these separate is the point. The anti-hallucination guarantee is not
"Kivi only knows memory", it is "Kivi never presents anything as your memory
unless a memory backs it".
"""
import concurrent.futures
import json
import random
import sqlite3
import time
from datetime import datetime
from typing import Any, Optional

from app.db import get_connection
from app.services import cache, fast_paths, persona, tts, web_search as _ws
from app.services.llm_client import chat_json, provider_status
from app.services.memory_pipeline import process_take
from app.services.retrieval import cache_size, embed, hybrid_search
from app.services.text_utils import normalize_text

RETRIEVAL_TOP_K = 5

# A floor, not a verdict. Measured on real data, a TRUE question-to-statement
# match runs as low as 0.19 while unrelated questions sit near 0.12 — too narrow
# a gap for a threshold to arbitrate. The floor only skips retrieval that is
# obviously empty; the real decision is citation gating.
RETRIEVAL_FLOOR = 0.15

DECLINE_SENTENCES = [
    "I don't have anything in memory about that yet.",
    "Nothing in my memory covers that yet.",
    "I haven't learned anything about that yet.",
]

ROUTER_SYSTEM_PROMPT = """You route input sent to Kivi, a personal memory assistant that also answers general questions.

You are given the recent conversation, then the newest message. Decide three things.

intent:
- "statement": the user is telling Kivi something about themselves to remember ("I work at Google", "I'm in love with someone")
- "request": the user is asking Kivi to DO something in the world — order, book, buy, send, schedule, remind ("order me a coffee", "book a table at 8")
- "question": the user is asking something

scope (only meaningful for questions):
- "personal": only the user's own recorded memories could answer it ("where do I work?", "do I have a dog?")
- "general": world knowledge answers it, the user's memories are irrelevant ("who founded Google?", "how do I boil an egg?")
- "mixed": needs both — a fact about the user AND world knowledge ("can I take my dog to my office?" needs their employer and that company's pet policy)

needs_live_data: true when a correct answer depends on information that changes day to day — weather, news, prices, sports results, "today", "right now". False for stable knowledge.

resolved_query: the newest message rewritten so it stands completely on its own, with every pronoun and reference replaced by what it actually refers to. This applies to statements as much as questions — it is what gets stored or searched, and it will be read later with no conversation around it.

How to resolve a pronoun: find the most recently mentioned thing it could refer to, and name that exact thing. Pets, objects and places take pronouns too — "she" often means a pet, not a person. If Kivi just asked a question, the newest message is almost always the answer to it, so resolve against what that question was about.

Worked examples:
- After "I'm in love with someone", "wanna know more about it?" becomes "Tell me more about the user being in love with someone."
- After Kivi asks "What is their name?", "she is called Meera" becomes "The person the user is in love with is called Meera."
- After "I adopted a cat called Pixel" and Kivi asks "What colour is Pixel?", the message "she is black and white with stripes on her face" becomes "Pixel, the user's cat, is black and white with stripes on her face." Note "she" is the cat.
- After "I work at Google", "moving to the Search team" becomes "The user is moving to the Search team at Google."

Never leave a bare "she", "he", "they", "it", or "that" in resolved_query, and never replace one with a vague stand-in like "the person" — name the actual subject. If the message already stands alone, repeat it unchanged.

reaction: what Kivi says out loud in response. Only matters for a statement or
a request — react to WHAT THEY SAID, not to the act of storing it. The
interface separately shows what was saved, so never describe the saving.

Return strict JSON:
{"intent": "question" | "statement" | "request", "scope": "personal" | "general" | "mixed", "needs_live_data": true | false, "resolved_query": "...", "reaction": "..."}"""


def _router_prompt() -> str:
    """Router instructions plus the current voice, so the reaction it drafts
    already sounds like Kivi rather than like a form."""
    return (
        f"{ROUTER_SYSTEM_PROMPT}\n\n"
        f"--- How your reaction should sound ---\n{persona.voice()}"
    )

MEMORY_ANSWER_PROMPT = """Answer using ONLY the memories provided. Do not add information not present in these memories. If they don't fully answer the question, say what you don't know.

Write the way a friend who knows them would: warm but brief, one or two sentences, no preamble, no restating the question, no corporate hedging. Address the user as "you". Use the conversation above for context on what they are referring to.

Return strict JSON:
{
  "answer": "your answer, grounded only in the memories given",
  "used_memory_ids": [3, 7],
  "gap": "the specific thing you would need to know to answer this fully, or null if the memories answered it completely"
}

used_memory_ids must list only the memory ids you actually relied on. If none of the memories answer the question, return an empty list.
Set "gap" whenever you had to say you don't know something — name the missing detail plainly, e.g. "the name of the person they are in love with".

CRITICAL — answer about the subject that was actually asked about. If the question names a specific person, pet, place or thing, only use memories about THAT one. A memory about a different pet, a different person or a different company is NOT an answer, however similarly worded it is. Retrieval returns near-misses by design; rejecting them is your job. When the memories are about something else, cite nothing and say you do not know."""

MIXED_ANSWER_PROMPT = """Answer the user's question using two sources: what Kivi remembers about them, and your own general knowledge.

Weave the two together in plain prose. Address the user as "you" for what they told you. Where you are relying on general knowledge rather than something they said, make that clear in ordinary words ("policies vary", "usually", "worth checking") — and if it varies by place or policy, say so rather than asserting.

Never annotate the prose with literal source markers like "(you)" or "(general knowledge)"; the interface labels provenance separately. Just write it the way a person would say it.

Direct and conversational, two or three sentences at most.

Return strict JSON:
{
  "answer": "the combined answer",
  "used_memory_ids": [3],
  "used_general_knowledge": true
}

used_memory_ids must list only memory ids you actually relied on."""

GENERAL_ANSWER_PROMPT = """Answer the user's question from your own knowledge, briefly and directly. Two or three sentences at most.

If you are not confident, or the answer depends on current information you may not have, say so plainly rather than guessing.

Return strict JSON: {"answer": "..."}"""

LIVE_ANSWER_PROMPT = """Answer the user's question using the web search results provided. Cite nothing that is not in the results. If the results don't answer it, say so.

These are search snippets, not a live feed, so they can be hours or days stale and sources often disagree. Handle that honestly:
- Never state a clock time as if it were now. The current time is given to you below — use only that. If a snippet carries its own timestamp, attribute it ("as of the reading published at ...").
- For numbers that move (prices, temperature), say roughly what the sources show and note it may be behind. Do not average conflicting figures into one confident number.
- Prefer the figure that carries the most recent date.

Two or three sentences, direct.

Return strict JSON: {"answer": "..."}"""

FOLLOW_UP_PROMPT = """Kivi could not answer a question because it has nothing relevant in memory. Suggest what it should learn.

You are given the question and everything Kivi currently knows about the user. Write 2-3 short questions Kivi could ask to fill the gap, so that a similar question is answerable next time.

Rules:
- Ask about the user, not the world. "Where is your office?" not "What is Google's pet policy?"
- Build on what is already known where you can — if Kivi knows their employer, ask about that employer specifically.
- Never repeat something already in memory.
- Each question under 12 words, phrased the way a person would ask.

Return strict JSON: {"questions": ["...", "..."]}"""


# Kept as a thin shim so call sites read the same; provider selection,
# JSON salvage and fallback all live in llm_client now.
def _chat_json(model: str, system: str, user: str, temperature: float = 0.2) -> dict[str, Any]:
    result = chat_json(system, user, temperature=temperature, fast=(model == "fast"))
    payload = dict(result.data)
    payload["_tokens"] = result.tokens
    payload["_model"] = result.model
    return payload


FAST = "fast"
MAIN = "main"


def _model_name() -> str:
    return provider_status()["chat_model"]


def cached_embed(text: str) -> Any:
    """embed() behind the process cache — the router path and the retrieval
    path ask for the same vector within milliseconds of each other."""
    hit = cache.get_embedding(text)
    if hit is not None:
        return hit
    vector = embed(text)
    cache.put_embedding(text, vector)
    return vector


def _prewarm_embedding(text: str) -> None:
    """Speculative encode while the router call is in flight."""
    try:
        cached_embed(text)
    except Exception:
        pass


def _memory_version(conn: sqlite3.Connection) -> int:
    """Cheap fingerprint of the memory store, so a cached answer is dropped the
    moment anything it could have been based on changes."""
    row = conn.execute(
        "SELECT COUNT(*) n, COALESCE(MAX(id), 0) mx, "
        "COALESCE(MAX(COALESCE(retired_at, created_at)), '') ts FROM memories"
    ).fetchone()
    return hash((row["n"], row["mx"], row["ts"]))


# -------------------------------------------------------------------- routing

MAX_HISTORY_TURNS = 8


def format_history(history: Optional[list[dict[str, Any]]]) -> str:
    """Recent turns as plain transcript, oldest first."""
    if not history:
        return ""
    lines = []
    for turn in history[-MAX_HISTORY_TURNS:]:
        speaker = "User" if str(turn.get("role")) == "user" else "Kivi"
        text = normalize_text(str(turn.get("text", ""))) or ""
        if text.strip():
            lines.append(f"{speaker}: {text.strip()}")
    return "\n".join(lines)


def _with_history(transcript: str, text: str, label: str = "Newest message") -> str:
    if not transcript:
        return text
    return f"Conversation so far:\n{transcript}\n\n{label}: {text}"


def route_input(text: str, transcript: str = "") -> dict[str, Any]:
    """Decide whether to remember this, and if not, where the answer lives.

    Also resolves the message against the conversation, so a follow-up like
    "tell me more about it" is searched as what it actually refers to rather
    than as a bag of pronouns that matches nothing.
    """
    payload = _chat_json(
        FAST, _router_prompt(),
        _with_history(transcript, text),
        # A flat zero makes the reaction wooden and repetitive; the routing
        # fields are constrained enough that a little warmth costs nothing.
        temperature=0.3,
    )
    intent = str(payload.get("intent", "")).lower()
    scope = str(payload.get("scope", "")).lower()
    resolved = normalize_text(str(payload.get("resolved_query", "") or "")).strip()
    return {
        # Falling back to "question" is the safe default: it reads memory
        # rather than writing something the user did not ask to store.
        "intent": intent if intent in ("statement", "request") else "question",
        "scope": scope if scope in ("personal", "general", "mixed") else "personal",
        "needs_live_data": bool(payload.get("needs_live_data", False)),
        "resolved_query": resolved or text,
        "reaction": (normalize_text(str(payload.get("reaction", "") or "")) or "").strip(),
    }


# ------------------------------------------------------------------ retrieval

def retrieve_candidates(
    conn: sqlite3.Connection, query: str, top_k: int = RETRIEVAL_TOP_K
) -> list[dict[str, Any]]:
    """Step 6: top_k active memories ranked by similarity, with their source takes."""
    return [
        {
            "id": int(row["id"]),
            "content": row["content"],
            "tag": row["tag"],
            "source_take_id": int(row["source_take_id"]),
            "source_take_created_at": row["source_take_created_at"],
            "similarity": float(score),
        }
        for score, row in hybrid_search(conn, query, cached_embed(query), top_k)
    ]


def _memory_context(candidates: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"[memory_id={c['id']}] ({c['tag']}) {c['content']} "
        f"(recorded {c['source_take_created_at']})"
        for c in candidates
    )


def _validated_ids(raw: Any, offered: set[int]) -> list[int]:
    if not isinstance(raw, list):
        return []
    return [int(i) for i in raw if isinstance(i, int) and int(i) in offered]


# ------------------------------------------------------------- answer sources

def _styled(prompt: str) -> str:
    return f"{prompt}\n\nTone: {persona.answer_style()}"


def answer_from_memory(
    query: str, candidates: list[dict[str, Any]], transcript: str = ""
) -> tuple[str, list[int], Optional[str], Optional[int]]:
    payload = _chat_json(
        MAIN, _styled(MEMORY_ANSWER_PROMPT),
        _with_history(transcript,
                      f"{query}\n\nMemories:\n{_memory_context(candidates)}", "Question"),
    )
    answer = normalize_text(payload.get("answer", "")) or ""
    used = _validated_ids(payload.get("used_memory_ids"), {c["id"] for c in candidates})
    raw_gap = payload.get("gap")
    gap = (normalize_text(str(raw_gap)) or "").strip() if raw_gap else ""
    return answer, used, gap or None, payload.get("_tokens")


def answer_mixed(
    query: str, candidates: list[dict[str, Any]], transcript: str = ""
) -> tuple[str, list[int], Optional[int]]:
    payload = _chat_json(
        MAIN, _styled(MIXED_ANSWER_PROMPT),
        _with_history(transcript,
                      f"{query}\n\nWhat Kivi remembers:\n{_memory_context(candidates)}", "Question"),
    )
    answer = normalize_text(payload.get("answer", "")) or ""
    used = _validated_ids(payload.get("used_memory_ids"), {c["id"] for c in candidates})
    return answer, used, payload.get("_tokens")


def answer_general(query: str, transcript: str = "") -> tuple[str, Optional[int]]:
    payload = _chat_json(MAIN, GENERAL_ANSWER_PROMPT,
                         _with_history(transcript, query, "Question"))
    return normalize_text(payload.get("answer", "")) or "", payload.get("_tokens")


def _now_line() -> str:
    """Kivi has no clock unless we give it one; without this it invents times."""
    now = datetime.now().astimezone()
    return now.strftime("The current date and time is %A %d %B %Y, %H:%M %Z.")


def answer_live(query: str) -> tuple[Optional[str], list[dict[str, Any]], Optional[int]]:
    """Answer from a live web lookup, or (None, [], None) if search is unavailable."""
    results = _ws.search(query)
    if not results:
        return None, [], None
    context = "\n\n".join(
        f"[{r['title']}] ({r['url']})\n{r['content']}" for r in results
    )
    payload = _chat_json(
        MAIN, LIVE_ANSWER_PROMPT,
        f"{_now_line()}\n\nQuestion: {query}\n\nSearch results:\n{context}",
    )
    return normalize_text(payload.get("answer", "")) or "", results, payload.get("_tokens")


def suggest_follow_ups(query: str, known: list[dict[str, Any]]) -> list[str]:
    """Questions Kivi should ask to close the gap it just hit."""
    listing = "\n".join(f"- ({m['tag']}) {m['content']}" for m in known) or "- (nothing yet)"
    payload = _chat_json(
        FAST, FOLLOW_UP_PROMPT,
        f"Unanswered question: {query}\n\nWhat Kivi already knows:\n{listing}",
        temperature=0.4,
    )
    questions = payload.get("questions", [])
    if not isinstance(questions, list):
        return []
    return [normalize_text(str(q)) for q in questions[:3] if str(q).strip()]


# ------------------------------------------------------------------ DB writes

def log_answer(conn: sqlite3.Connection, record: dict[str, Any]) -> int:
    """Step 8: every Hey Kivi response is logged with full traceability."""
    cursor = conn.execute(
        """
        INSERT INTO answers (
            query, retrieved_memory_ids, retrieval_score, decision, decision_reason,
            answer_text, cited_memory_ids, cited_take_ids,
            retrieval_latency_ms, generation_latency_ms, total_latency_ms,
            model_used, tokens_used, answer_source, follow_ups
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record["query"],
            json.dumps(record["retrieved_memory_ids"]),
            record["retrieval_score"],
            record["decision"],
            record["decision_reason"],
            record["answer_text"],
            json.dumps(record["cited_memory_ids"]),
            json.dumps(record["cited_take_ids"]),
            record["retrieval_latency_ms"],
            record["generation_latency_ms"],
            record["total_latency_ms"],
            record["model_used"],
            record["tokens_used"],
            record["answer_source"],
            json.dumps(record["follow_ups"]),
        ),
    )
    return int(cursor.lastrowid)


def bump_cited_memories(conn: sqlite3.Connection, memory_ids: list[int]) -> None:
    """Lifecycle tracking: a memory that got used is a memory worth keeping."""
    for memory_id in memory_ids:
        conn.execute(
            "UPDATE memories SET usage_count = usage_count + 1, "
            "last_used_at = CURRENT_TIMESTAMP WHERE id = ?",
            (memory_id,),
        )


# --------------------------------------------------------------- orchestrator

def answer_question(
    query: str, route: dict[str, Any], transcript: str = ""
) -> dict[str, Any]:
    """Retrieve, then answer from memory, general knowledge, or the web."""
    conn = get_connection()
    try:
        started = time.perf_counter()
        # Repeat questions are common (a refresh, a rephrase); serve them from
        # cache, but only while the memory store is unchanged.
        cache_key = f"{route.get('resolved_query') or query}|{transcript[-200:]}"
        version = _memory_version(conn)
        if not route["needs_live_data"]:
            hit = cache.get_answer(cache_key, version)
            if hit is not None:
                return {**hit, "cached": True,
                        "total_latency_ms": int((time.perf_counter() - started) * 1000)}
        # Search what the message actually refers to, not its pronouns.
        search_query = route.get("resolved_query") or query
        candidates = retrieve_candidates(conn, search_query)
        retrieval_latency_ms = int((time.perf_counter() - started) * 1000)

        retrieved_memory_ids = [c["id"] for c in candidates]
        retrieval_score = candidates[0]["similarity"] if candidates else None
        has_memory = bool(candidates) and retrieval_score >= RETRIEVAL_FLOOR

        scope = route["scope"]
        answer_text = ""
        cited_memory_ids: list[int] = []
        tokens_used: Optional[int] = None
        model_used: Optional[str] = None
        answer_source = "none"
        decision_reason: Optional[str] = None
        sources: list[dict[str, Any]] = []
        gap: Optional[str] = None
        # Whether answer_text is allowed to reach the user. Ungrounded prose
        # from the memory path must never be shown, however plausible it reads.
        trusted = False

        generation_started = time.perf_counter()

        if route["needs_live_data"]:
            live_answer, sources, tokens_used = answer_live(search_query)
            if live_answer:
                answer_text, answer_source, model_used = live_answer, "live", _model_name()
                trusted = True
            else:
                # Honest about the limit instead of guessing at today's facts.
                answer_text = (
                    "I can't look that up — it needs live information and web search "
                    "isn't connected yet."
                )
                answer_source = "none"
                decision_reason = "live data required, web search not configured"
                trusted = True

        elif scope == "general":
            answer_text, tokens_used = answer_general(search_query, transcript)
            answer_source, model_used, trusted = "general", _model_name(), True

        elif scope == "mixed" and has_memory:
            answer_text, cited_memory_ids, tokens_used = answer_mixed(
                search_query, candidates, transcript
            )
            answer_source, model_used, trusted = "mixed", _model_name(), True

        elif has_memory:
            answer_text, cited_memory_ids, gap, tokens_used = answer_from_memory(
                search_query, candidates, transcript
            )
            model_used = _model_name()
            if cited_memory_ids:
                answer_source, trusted = "memory", True
            else:
                # Citation gating: prose grounded in nothing is discarded, not shown.
                answer_source = "none"
                decision_reason = "retrieved memories did not answer the question"

        elif scope == "mixed":
            answer_text, tokens_used = answer_general(search_query, transcript)
            answer_source, model_used, trusted = "general", _model_name(), True

        else:
            answer_source = "none"
            decision_reason = (
                "no memories exist yet" if not candidates
                else f"nothing above retrieval floor {RETRIEVAL_FLOOR} "
                     f"(top score {retrieval_score:.3f})"
            )

        generation_latency_ms = int((time.perf_counter() - generation_started) * 1000)

        # The gate: anything not trusted is replaced, never merely supplemented.
        if not trusted:
            answer_text = random.choice(DECLINE_SENTENCES)

        decision = "answered" if answer_source != "none" else "declined_unknown"

        # Ask rather than just refusing — that is how the store grows. This
        # fires on a partial answer too: saying "I don't know their name" is a
        # named gap, and the moment to ask is while it is being discussed.
        follow_ups: list[str] = []
        should_ask = (
            gap is not None
            or answer_source == "none"
            or (answer_source == "general" and scope in ("personal", "mixed"))
        )
        if should_ask:
            follow_ups = suggest_follow_ups(
                gap or route.get("resolved_query") or query, candidates
            )

        cited_memories = [c for c in candidates if c["id"] in cited_memory_ids]
        cited_take_ids = sorted({c["source_take_id"] for c in cited_memories})
        total_latency_ms = int((time.perf_counter() - started) * 1000)

        log_answer(conn, {
            "query": query,
            "retrieved_memory_ids": retrieved_memory_ids,
            "retrieval_score": retrieval_score,
            "decision": decision,
            "decision_reason": decision_reason,
            "answer_text": answer_text,
            "cited_memory_ids": cited_memory_ids,
            "cited_take_ids": cited_take_ids,
            "retrieval_latency_ms": retrieval_latency_ms,
            "generation_latency_ms": generation_latency_ms,
            "total_latency_ms": total_latency_ms,
            "model_used": model_used,
            "tokens_used": tokens_used,
            "answer_source": answer_source,
            "follow_ups": follow_ups,
        })
        bump_cited_memories(conn, cited_memory_ids)
        conn.commit()

        payload = {
            "mode": "answer",
            "answer": answer_text,
            "decision": decision,
            "answer_source": answer_source,
            "cited_memories": [
                {
                    "id": c["id"],
                    "content": c["content"],
                    "tag": c["tag"],
                    "source_take_id": c["source_take_id"],
                    "source_take_created_at": c["source_take_created_at"],
                }
                for c in cited_memories
            ],
            "web_sources": [{"title": s["title"], "url": s["url"]} for s in sources],
            "follow_ups": follow_ups,
            "retrieval_score": retrieval_score,
            "total_latency_ms": total_latency_ms,
        }
        if not route["needs_live_data"]:
            cache.put_answer(cache_key, version, payload)
        return payload
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def last_kivi_question(history: Optional[list[dict[str, Any]]]) -> Optional[str]:
    """The question Kivi asked most recently, if its last turn asked one."""
    for turn in reversed(history or []):
        if str(turn.get("role")) == "kivi":
            text = str(turn.get("text", ""))
            return text if "?" in text else None
    return None


def remember_statement(
    text: str, answering: Optional[str] = None, reaction: str = ""
) -> dict[str, Any]:
    """The user stated something rather than asking: run it through ingestion."""
    started = time.perf_counter()
    result = process_take(text, answering_question=answering)

    saved = result.get("saved", [])
    watched = result.get("watched", [])
    dropped = result.get("dropped", [])

    # What Kivi says is the reaction to what they told it. The outcome of
    # storing it is shown separately in the interface, so it does not need
    # narrating twice.
    if reaction and persona.is_expressive() and (saved or watched):
        return {
            "mode": "remember",
            "answer": reaction,
            "decision": None,
            "answer_source": None,
            "cited_memories": [],
            "web_sources": [],
            "follow_ups": [],
            "retrieval_score": None,
            "ingested": result,
            "total_latency_ms": int((time.perf_counter() - started) * 1000),
        }

    superseded = [mid for item in saved for mid in item.get("superseded_memory_ids") or []]
    if superseded:
        acknowledgement = "Updated — I've replaced what I had with that."
    elif any(item.get("action") == "reinforced_existing" for item in saved):
        acknowledgement = "I already had that, so I've noted it again."
    elif saved:
        acknowledgement = "Got it, I'll remember that."
    elif watched:
        acknowledgement = "Noted. If you mention it again, I'll remember it properly."
    elif dropped:
        acknowledgement = "That didn't look like something worth keeping, so I let it go."
    else:
        acknowledgement = "Nothing there worth remembering."

    return {
        "mode": "remember",
        "answer": acknowledgement,
        "decision": None,
        "answer_source": None,
        "cited_memories": [],
        "web_sources": [],
        "follow_ups": [],
        "retrieval_score": None,
        "ingested": result,
        "total_latency_ms": int((time.perf_counter() - started) * 1000),
    }


def _trivial_reply(kind: str, answer: str, started: float) -> dict[str, Any]:
    return {
        "mode": "chat",
        "answer": answer,
        "decision": None,
        "answer_source": None,
        "cited_memories": [],
        "web_sources": [],
        "follow_ups": [],
        "retrieval_score": None,
        "fast_path": kind,
        "total_latency_ms": int((time.perf_counter() - started) * 1000),
    }


def process_hey_kivi(
    raw_query: str,
    history: Optional[list[dict[str, Any]]] = None,
    answering: Optional[str] = None,
) -> dict[str, Any]:
    """Route one Hey Kivi input to recall, general knowledge, or remembering."""
    started = time.perf_counter()
    text = normalize_text(raw_query)

    # "ok", "thanks", "hi" need no model. This is a large share of real chat
    # traffic and used to cost two inferences apiece. An answer to a question
    # Kivi asked is never trivial, though — "Meera" is a real fact.
    if not answering:
        trivial = fast_paths.check(text)
        if trivial is not None:
            return _trivial_reply(trivial["kind"], trivial["answer"], started)

    transcript = format_history(history)

    # The router and the query embedding are independent, so overlap them.
    # Most messages already stand alone, so the speculative embedding of the
    # raw text is usually the one retrieval needs; when the router rewrites the
    # message we simply embed again.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        route_future = pool.submit(route_input, text, transcript)
        pool.submit(_prewarm_embedding, text)
        route = route_future.result()

    # The user explicitly answered a question Kivi asked. Do not let the router
    # decide the intent: a bare "Meera" or "grey" reads as a question in
    # isolation, which would search memory, find nothing, and ask again.
    if answering:
        return remember_statement(
            route.get("resolved_query") or text, answering=answering,
            reaction=route.get("reaction", ""),
        )

    if route["intent"] == "request":
        # Kivi has no hands. Be straight about that, but still learn from what
        # the request revealed about how they like things.
        result = remember_statement(
            route.get("resolved_query") or text,
            answering=last_kivi_question(history),
        )
        saved = result.get("ingested", {}).get("saved", [])
        learned = "; ".join(item["content"] for item in saved)
        result["answer"] = (
            "I can't place orders or bookings yet — no connected app for that. "
            + (f"I have noted it though: {learned}" if learned
               else "Nothing in it for me to remember either.")
        )
        return result

    if route["intent"] == "statement":
        # Store the resolved form so "yeah her name is Meera" is saved as a
        # standalone fact rather than a fragment that means nothing later.
        # If Kivi asked for this, say so: a solicited answer is worth keeping
        # by definition — Kivi just said it wanted to know.
        return remember_statement(
            route.get("resolved_query") or text,
            answering=last_kivi_question(history),
            reaction=route.get("reaction", ""),
        )
    return answer_question(text, route, transcript)
