"""Live web lookup for questions memory cannot answer.

Two providers, tried in order:
  1. Tavily, if SEARCH_API_KEY is set — better ranking, stable rate limits.
  2. DuckDuckGo via the `ddgs` package — no key, no signup, works out of the box.

Groq's built-in browsing models (groq/compound) are not enabled on every
account, which is why search is done here rather than by the model.
"""
import os
from typing import Any, Optional

import httpx

TAVILY_ENDPOINT = "https://api.tavily.com/search"
SEARCH_TIMEOUT_SECONDS = 6.0
# Three snippets is enough to answer and cite; each extra one costs a round
# trip on the scrape and tokens on the generation that follows.
MAX_RESULTS = 3
MAX_SNIPPET_CHARS = 450

# Named explicitly and ordered by measured speed. ddgs defaults to "auto",
# which walks every backend in turn and costs ~8s; naming them takes the same
# query to under a second, with the rest as fallbacks when one is rate limited.
FREE_BACKENDS = ("brave", "duckduckgo", "bing")


def is_configured() -> bool:
    """Search is always available: DuckDuckGo needs no credentials."""
    return True


def active_provider() -> str:
    return "tavily" if os.getenv("SEARCH_API_KEY", "").strip() else "duckduckgo"


def _search_tavily(query: str, api_key: str) -> Optional[list[dict[str, Any]]]:
    try:
        response = httpx.post(
            TAVILY_ENDPOINT,
            json={
                "api_key": api_key,
                "query": query,
                "max_results": MAX_RESULTS,
                "search_depth": "basic",
            },
            timeout=SEARCH_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None

    return [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": (item.get("content") or "")[:MAX_SNIPPET_CHARS],
        }
        for item in payload.get("results", [])[:MAX_RESULTS]
    ]


def _search_duckduckgo(query: str) -> Optional[list[dict[str, Any]]]:
    try:
        from ddgs import DDGS
    except Exception:
        return None

    results: list[dict[str, Any]] = []
    for backend in FREE_BACKENDS:
        try:
            results = list(
                DDGS(timeout=SEARCH_TIMEOUT_SECONDS).text(
                    query, max_results=MAX_RESULTS, backend=backend
                )
            )
        except Exception:
            # Rate limited or blocked: try the next backend rather than failing.
            continue
        if results:
            break

    if not results:
        # No backend answered: caller degrades to "I can't check that" rather
        # than guessing at today's facts.
        return None

    return [
        {
            "title": item.get("title", ""),
            "url": item.get("href", "") or item.get("url", ""),
            "content": (item.get("body", "") or "")[:MAX_SNIPPET_CHARS],
        }
        for item in results
    ]


def search(query: str) -> Optional[list[dict[str, Any]]]:
    """Return [{title, url, content}], or None when the lookup failed."""
    api_key = os.getenv("SEARCH_API_KEY", "").strip()
    if api_key:
        results = _search_tavily(query, api_key)
        if results:
            return results
    return _search_duckduckgo(query)
