"""Deterministic handling for input that needs no model at all.

Two bottlenecks meet here. Every message was costing 2-4 inferences, and
"prompt-as-architecture" meant even "ok" went through a router. Greetings,
acknowledgements and empty noise are a large share of real chat traffic and
none of them need a 30B model to classify.

These rules are exact-match on a normalised string, never fuzzy. A rule that
guessed would reintroduce the unpredictability this file exists to remove.
"""
import re
import unicodedata
from typing import Optional

# Acknowledgements: nothing to remember, nothing to look up.
_ACKS = {
    "ok", "okay", "k", "kk", "cool", "nice", "great", "good", "fine", "sure",
    "thanks", "thank you", "thx", "ty", "cheers", "got it", "gotcha",
    "yes", "yeah", "yep", "yup", "no", "nope", "nah",
    "hmm", "hm", "oh", "ah", "haha", "lol", "sorry",
    "sari", "seri", "theek hai", "thik hai", "acha", "accha", "ha", "haan",
}

_GREETINGS = {
    "hi", "hey", "hello", "yo", "hiya", "hey kivi", "hi kivi", "hello kivi",
    "good morning", "good afternoon", "good evening", "morning", "evening",
    "namaste", "vanakkam", "namaskara", "sat sri akal",
}

_FAREWELLS = {"bye", "goodbye", "see you", "cya", "good night", "gn", "night"}

_CAPABILITY_PATTERNS = (
    re.compile(r"^what (can|do) you do\??$", re.I),
    re.compile(r"^who are you\??$", re.I),
    re.compile(r"^help\??$", re.I),
)

GREETING_REPLY = "Hey. Tell me something to remember, or ask what I already know."
ACK_REPLY = "Anytime."
FAREWELL_REPLY = "See you."
CAPABILITY_REPLY = (
    "I remember things you tell me about yourself, answer questions from those "
    "memories with citations, and look things up when memory can't help. "
    "Speak or type — I work in English and Indian languages."
)


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").strip().lower()
    # Trailing punctuation and emoji shouldn't defeat an exact match.
    text = re.sub(r"[!.,~\s]+$", "", text)
    text = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", "", text).strip()
    return text


def check(text: str) -> Optional[dict[str, str]]:
    """Return a canned reply for trivial input, or None to run the full pipeline.

    Kept deliberately narrow: anything with substance falls through.
    """
    normalised = _normalise(text)
    if not normalised:
        return {"kind": "empty", "answer": "I didn't catch that — say it again?"}

    # More than a handful of words is never a bare greeting.
    if len(normalised.split()) > 4:
        return None

    if normalised in _GREETINGS:
        return {"kind": "greeting", "answer": GREETING_REPLY}
    if normalised in _ACKS:
        return {"kind": "ack", "answer": ACK_REPLY}
    if normalised in _FAREWELLS:
        return {"kind": "farewell", "answer": FAREWELL_REPLY}
    for pattern in _CAPABILITY_PATTERNS:
        if pattern.match(normalised):
            return {"kind": "capability", "answer": CAPABILITY_REPLY}
    return None
