"""Retrieval scaling benchmark.

Answers the question a reviewer will ask: what happens at 50k memories?
Populates a throwaway DB with synthetic memories and times retrieval with the
cache cold and warm.

    python bench/scale.py 50000
"""
import io
import os
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("GROQ_API_KEY", "bench")

# The Windows console defaults to cp1252 and cannot print the Tamil probe.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SUBJECTS = ["Priya", "Rahul", "Meera", "Kavya", "Arjun", "Divya", "Karthik"]
PLACES = ["Chennai", "Bangalore", "Hyderabad", "Pune", "Coimbatore", "Kochi"]
THINGS = ["dentist appointment", "team offsite", "filter coffee", "gym session",
          "flight booking", "quarterly review", "cricket match"]


def synthetic(n: int) -> list[str]:
    random.seed(7)
    return [
        f"The user had a {random.choice(THINGS)} with {random.choice(SUBJECTS)} "
        f"in {random.choice(PLACES)} (note {i})"
        for i in range(n)
    ]


def main() -> None:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    db_path = Path(tempfile.mkdtemp()) / "bench.db"
    os.environ["DATABASE_PATH"] = str(db_path)

    import app.db as db_module

    db_module.DATABASE_PATH = str(db_path)
    db_module.get_db_path = lambda: db_path
    db_module.init_db()

    from app.services import lexical, retrieval

    contents = synthetic(count)
    print(f"encoding {count} memories ...")
    started = time.perf_counter()
    vectors = retrieval.embed_batch(contents)
    encode_s = time.perf_counter() - started
    print(f"  encoded in {encode_s:.1f}s ({count / encode_s:.0f}/s)")

    conn = db_module.get_connection()
    conn.execute("INSERT INTO takes (raw_text) VALUES ('bench')")
    take_id = conn.execute("SELECT MAX(id) m FROM takes").fetchone()["m"]
    conn.executemany(
        "INSERT INTO memories (content, source_take_id, tag, confidence, embedding) "
        "VALUES (?, ?, 'episode', 1.0, ?)",
        [(c, take_id, retrieval.serialize_embedding(v)) for c, v in zip(contents, vectors)],
    )
    conn.commit()
    indexed = lexical.reindex(conn)
    conn.commit()
    print(f"  stored {count}, BM25-indexed {indexed}")

    size_mb = db_path.stat().st_size / 1e6
    print(f"  db size: {size_mb:.1f} MB  ({size_mb * 1000 / count:.2f} KB per memory)")

    queries = [
        "When is my dentist appointment?",
        "Who did I meet in Bangalore?",
        "Tell me about the team offsite",
        "எனக்கு பல் மருத்துவர் சந்திப்பு எப்போது?",
    ]

    retrieval.invalidate_cache()
    started = time.perf_counter()
    retrieval.hybrid_search(conn, queries[0], retrieval.embed(queries[0]), 5)
    print(f"\n  cold (cache miss, loads {count} vectors): "
          f"{(time.perf_counter() - started) * 1000:.0f}ms")

    timings = []
    for query in queries:
        started = time.perf_counter()
        retrieval.hybrid_search(conn, query, retrieval.embed(query), 5)
        timings.append((time.perf_counter() - started) * 1000)
    for query, ms in zip(queries, timings):
        print(f"  warm: {ms:6.1f}ms  {query[:44]}")
    print(f"\n  warm average: {sum(timings) / len(timings):.1f}ms at {count} memories")
    conn.close()


if __name__ == "__main__":
    main()
