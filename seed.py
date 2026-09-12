"""Fill a fresh database with a realistic set of memories.

Run this so the app has something in it when you open it, and so the
evaluation has a known store to retrieve against.

    python seed.py            # add the seed data
    python seed.py --reset    # wipe the database first

The seed writes memories directly. It does not call the language model, so it
costs nothing and gives the same result every time. That matters for the
evaluation: if seeding went through the filter, the store would differ between
runs and retrieval scores would not be comparable.

The person described here is invented. The details are the kind an Indian user
would actually give a voice assistant, including a few in Tamil and Hindi.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("GROQ_API_KEY", "seed-no-model-needed")

# (raw take, memory content, tag)
# The take is what the user said. The memory is what Kivi would have stored.
SEED = [
    ("I work at Zoho as a data analyst",
     "The user works at Zoho as a data analyst", "fact"),
    ("I moved to Chennai last month, staying in Adyar",
     "The user lives in Adyar, Chennai, and moved there recently", "fact"),
    ("My manager is Rahul and my skip level is Divya",
     "The user's manager is Rahul and their skip level manager is Divya", "fact"),
    ("I am allergic to peanuts, quite badly",
     "The user has a severe peanut allergy", "fact"),
    ("My sister Kavya is getting married in December in Coimbatore",
     "The user's sister Kavya is getting married in December in Coimbatore", "fact"),
    ("I adopted a cat called Pixel, she is black and white",
     "The user has a cat named Pixel who is black and white", "fact"),
    ("My blood group is O positive",
     "The user's blood group is O positive", "fact"),

    ("I have a dentist appointment next Tuesday at 3pm",
     "The user has a dentist appointment next Tuesday at 3 PM", "episode"),
    ("Team offsite is on the 14th at Mahabalipuram",
     "The user's team offsite is on the 14th at Mahabalipuram", "episode"),
    ("I finished the quarterly report and sent it to Rahul",
     "The user finished the quarterly report and sent it to Rahul", "episode"),
    ("நாளைக்கு அம்மாவை ஹாஸ்பிடல் கூட்டிட்டு போகணும்",
     "The user needs to take their mother to the hospital tomorrow", "episode"),

    ("I always order filter coffee when I am working late",
     "The user drinks filter coffee when working late", "preference"),
    ("I sign off my work emails with Warm regards",
     "The user signs off work emails with 'Warm regards'", "preference"),
    ("I prefer morning meetings, I am useless after 4pm",
     "The user prefers morning meetings and is least productive after 4 PM", "preference"),
    ("मुझे वेज खाना ही पसंद है, बाहर जाने पर भी",
     "The user prefers vegetarian food, including when eating out", "preference"),
    ("I cycle to work most mornings, takes about 40 minutes",
     "The user cycles to work most mornings, a ride of about 40 minutes", "preference"),
]


def reset(db_path: Path) -> None:
    if db_path.exists():
        db_path.unlink()
        print(f"removed {db_path.name}")


def seed() -> dict:
    from app.db import get_connection, init_db
    from app.services import lexical
    from app.services.retrieval import embed_batch, invalidate_cache, serialize_embedding

    init_db()
    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT COUNT(*) n FROM memories WHERE status = 'active'"
        ).fetchone()["n"]
        if existing:
            print(f"database already has {existing} active memories, nothing added")
            print("use --reset to start over")
            return {"added": 0, "existing": existing}

        lexical.ensure_index(conn)
        vectors = embed_batch([memory for _, memory, _ in SEED])

        for (raw_text, content, tag), vector in zip(SEED, vectors):
            cursor = conn.execute(
                "INSERT INTO takes (raw_text, app_context) VALUES (?, 'seed')",
                (raw_text,),
            )
            take_id = int(cursor.lastrowid)
            cursor = conn.execute(
                "INSERT INTO memories (content, source_take_id, tag, confidence, embedding) "
                "VALUES (?, ?, ?, 1.0, ?)",
                (content, take_id, tag, serialize_embedding(vector)),
            )
            lexical.add(conn, int(cursor.lastrowid), content)

        conn.commit()
        invalidate_cache()

        counts = {
            row["tag"]: row["n"]
            for row in conn.execute(
                "SELECT tag, COUNT(*) n FROM memories WHERE status='active' GROUP BY tag"
            )
        }
        return {"added": len(SEED), "by_tag": counts}
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Kivi with sample memories.")
    parser.add_argument("--reset", action="store_true", help="delete the database first")
    args = parser.parse_args()

    from app.db import get_db_path

    if args.reset:
        reset(get_db_path())

    result = seed()
    if result["added"]:
        print(f"seeded {result['added']} memories")
        for tag, count in sorted(result.get("by_tag", {}).items()):
            print(f"  {tag}: {count}")


if __name__ == "__main__":
    main()
