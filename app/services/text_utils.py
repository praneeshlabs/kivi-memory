"""Text normalization shared by ingestion and retrieval.

LLM output sometimes contains non-standard whitespace (e.g. U+202F narrow
no-break space) that renders as garbled text (e.g. "5 pm"). Every content
string that reaches storage goes through normalize_text() first.
"""
import unicodedata
from typing import Any, Optional

_NARROW_NO_BREAK_SPACE = " "  # narrow no-break space
_NO_BREAK_SPACE = " "         # no-break space


def normalize_text(text: Optional[str]) -> Optional[str]:
    """Replace non-standard whitespace, then NFKC-normalize as general cleanup."""
    if text is None:
        return None
    text = text.replace(_NARROW_NO_BREAK_SPACE, " ").replace(_NO_BREAK_SPACE, " ")
    return unicodedata.normalize("NFKC", text)


def normalize_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize every text field an LLM item carries before it reaches storage."""
    for item in items:
        if "content" in item:
            item["content"] = normalize_text(item["content"])
        if "reason" in item:
            item["reason"] = normalize_text(item["reason"])
    return items
