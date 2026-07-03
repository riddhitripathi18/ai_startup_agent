"""
Search Client
=============
Shared module that provides web search via a configurable backend.

Set ``SEARCH_PROVIDER`` in your ``.env`` to choose the backend:

- ``serper`` (default) — Serper.dev Google Search API (requires ``SERPER_API_KEY``)
- ``duckduckgo`` — DuckDuckGo (free, no API key needed)
"""

import os
import logging
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "serper").lower()

# -- Serper ------------------------------------------------------------------

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SERPER_URL = "https://google.serper.dev/search"


def _serper_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Run a search via the Serper.dev Google Search API."""
    if not SERPER_API_KEY or SERPER_API_KEY == "your_api_key_here":
        logger.error("SERPER_API_KEY is not set. Add it to your .env file.")
        return []

    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": max_results}

    try:
        resp = requests.post(SERPER_URL, json=payload, headers=headers, timeout=8)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("organic", [])[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "body": item.get("snippet", ""),
                "href": item.get("link", ""),
            })
        return results
    except Exception as e:
        logger.warning(f"Serper search failed for '{query}': {e}")
        return []


# -- DuckDuckGo --------------------------------------------------------------

def _duckduckgo_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Run a search via DuckDuckGo (no API key required)."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        # Normalise to the same dict shape as Serper
        return [
            {
                "title": r.get("title", ""),
                "body": r.get("body", ""),
                "href": r.get("href", ""),
            }
            for r in raw
        ]
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed for '{query}': {e}")
        return []


# -- Public API --------------------------------------------------------------

def web_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """
    Search the web using the provider configured via ``SEARCH_PROVIDER``.

    Returns a list of dicts, each with keys ``title``, ``body``, ``href``.
    Falls back to an empty list on any error.
    """
    if SEARCH_PROVIDER == "duckduckgo":
        return _duckduckgo_search(query, max_results)
    return _serper_search(query, max_results)


# Backward-compat alias — prefer ``web_search`` in new code.
serper_search = web_search
