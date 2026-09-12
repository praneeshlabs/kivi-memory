"""Re-embed stored memories and watchlist rows.

Run this after changing EMBEDDING_MODEL_NAME or EMBEDDING_DTYPE in
app/services/retrieval.py, otherwise old vectors stay in the old format and
similarity scores become meaningless:

    python reembed.py
"""
from dotenv import load_dotenv

load_dotenv()

from app.db import get_connection, init_db
from app.services.retrieval import EMBEDDING_MODEL_NAME, embed, serialize_embedding


def main() -> None:
    init_db()
    conn = get_connection()
    try:
        memories = conn.execute("SELECT id, content FROM memories").fetchall()
        for row in memories:
            conn.execute(
                "UPDATE memories SET embedding = ? WHERE id = ?",
                (serialize_embedding(embed(row["content"])), row["id"]),
            )

        watchlist = conn.execute("SELECT id, content FROM watchlist").fetchall()
        for row in watchlist:
            conn.execute(
                "UPDATE watchlist SET embedding = ? WHERE id = ?",
                (serialize_embedding(embed(row["content"])), row["id"]),
            )

        conn.commit()
        print(f"re-embedded with {EMBEDDING_MODEL_NAME}: "
              f"{len(memories)} memories, {len(watchlist)} watchlist rows")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
