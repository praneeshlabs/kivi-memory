"""Consolidation: collapse memories that say the same thing.

Ingestion decides redundancy one memory at a time, against what existed at
that moment. That is not enough on its own — three restatements arriving over
three days can each look novel next to what came before, and the store ends up
holding one fact five ways. Retrieval then returns five near-identical rows and
the user reads the same thing repeatedly.

This is the sweep that fixes what ingestion let through: group the plausibly
redundant, ask the model which are genuinely the same situation, keep the most
informative, archive the rest with a pointer to the survivor.

Nothing is deleted. A consolidated memory is archived as 'redundant' and stays
in memory_archive, so the decision is auditable and reversible by hand.
"""
import json
import sqlite3
from typing import Any

from app.db import get_connection
from app.services.llm_client import chat_json
from app.services.memory_store import archive_memory
from app.services.retrieval import (
    cached_embed_safe,
    load_active_memories,
    rank_rows,
)

# How many neighbours to consider around each memory. Redundancy is local:
# a restatement is always near its original, never across the whole store.
NEIGHBOURHOOD = 6
# Only bother asking about pairs with some lexical/semantic proximity.
MIN_NEIGHBOUR_SIMILARITY = 0.45

CONSOLIDATE_PROMPT = """You are tidying a personal memory store. You are shown a small group of memories that may describe the SAME situation in different words.

Identify groups that are redundant — where keeping all of them would tell the user the same thing more than once. For each redundant group, choose the ONE memory that should survive: the most complete and specific phrasing.

Rules:
- Different wording is not different information.
- Only group memories about the SAME situation. Two genuinely different facts about the user's life are not redundant, however similar the words.
- A memory that adds any real detail the others lack should be the survivor, not a casualty.
- If nothing is redundant, return an empty list. That is a normal answer.

Return strict JSON:
{
  "groups": [
    {"keep": 7, "remove": [9, 11], "reason": "same undisclosed-partner fact restated"}
  ]
}"""


def _neighbourhoods(conn: sqlite3.Connection) -> list[list[dict[str, Any]]]:
    """Small clusters of same-tag memories that sit close together."""
    rows = load_active_memories(conn)
    if len(rows) < 2:
        return []

    seen: set[int] = set()
    clusters: list[list[dict[str, Any]]] = []

    for row in rows:
        memory_id = int(row["id"])
        if memory_id in seen:
            continue
        same_tag = [other for other in rows if other["tag"] == row["tag"]]
        if len(same_tag) < 2:
            continue

        embedding = cached_embed_safe(row["content"])
        ranked = rank_rows(same_tag, embedding, NEIGHBOURHOOD)
        group = [
            {"id": int(r["id"]), "content": r["content"], "tag": r["tag"]}
            for score, r in ranked
            if score >= MIN_NEIGHBOUR_SIMILARITY
        ]
        if len(group) < 2:
            continue

        clusters.append(group)
        seen.update(item["id"] for item in group)

    return clusters


def consolidate(dry_run: bool = False) -> dict[str, Any]:
    """Find and collapse redundant memories. Returns what changed."""
    conn = get_connection()
    try:
        active_before = len(load_active_memories(conn))
        merges: list[dict[str, Any]] = []

        for group in _neighbourhoods(conn):
            listing = "\n".join(
                f"[memory_id={item['id']}] ({item['tag']}) {item['content']}"
                for item in group
            )
            payload = chat_json(
                CONSOLIDATE_PROMPT, f"Memories:\n{listing}", temperature=0.0, fast=True
            ).data

            offered = {item["id"] for item in group}
            by_id = {item["id"]: item for item in group}

            for entry in payload.get("groups", []) or []:
                try:
                    keep = int(entry.get("keep"))
                except (TypeError, ValueError):
                    continue
                remove = [
                    int(r) for r in entry.get("remove", []) or []
                    if isinstance(r, int) and int(r) in offered and int(r) != keep
                ]
                if keep not in offered or not remove:
                    continue

                merges.append({
                    "kept": keep,
                    "kept_content": by_id[keep]["content"],
                    "removed": remove,
                    "removed_content": [by_id[r]["content"] for r in remove],
                    "reason": str(entry.get("reason", ""))[:200],
                })

                if not dry_run:
                    for memory_id in remove:
                        archive_memory(
                            conn, memory_id,
                            retired_reason="redundant",
                            contradicted_by_memory_id=keep,
                        )

        if not dry_run:
            conn.commit()

        active_after = len(load_active_memories(conn))
        return {
            "dry_run": dry_run,
            "active_before": active_before,
            "active_after": active_after,
            "removed": sum(len(m["removed"]) for m in merges),
            "merges": merges,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
