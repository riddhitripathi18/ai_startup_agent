"""
VC Research Tool
================
Searches for Venture Capital firms that have invested in similar startups
within a given market/domain. Uses Serper (Google Search) by default, with
optional Crunchbase Basic API integration when CRUNCHBASE_API_KEY is set.
"""

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv

from search_client import web_search

load_dotenv()

logger = logging.getLogger(__name__)

CRUNCHBASE_API_KEY = os.getenv("CRUNCHBASE_API_KEY", "")
CRUNCHBASE_API_URL = "https://api.crunchbase.com/api/v4"
REQUEST_TIMEOUT = 8

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
}

# Patterns that strongly suggest a VC firm name
_VC_NAME_PATTERNS = [
    re.compile(r'\b([A-Z][A-Za-z0-9\'&.\s]+?(?:Capital|Ventures|Partners|Equity|Group|Fund|Investments|Associates|Advisors|Global|Growth))\b'),
    re.compile(r'led\s+by\s+([A-Z][A-Za-z0-9\s&.]+?)(?:[,]|\s+invested|\s+participated|\s+and|\s+at\s+|$)'),
    re.compile(r'raised\s+\$[\d.,]+\s*(?:million|billion|M|B|Mn|Bn)?\s*(?:in\s+)?(?:funding|Series|Seed|Round|from).*?(?:from\s+|led\s+by\s+)([A-Z][A-Za-z0-9\s&.\']+?)(?:[,]|\s+and|\s+\.|$)'),
]

# Known notable VC firms to catch even without pattern matches
_KNOWN_VC = [
    "Andreessen Horowitz", "a16z", "Sequoia Capital", "Accel", "Benchmark",
    "Greylock Partners", "Kleiner Perkins", "Bessemer Venture Partners",
    "Index Ventures", "Lightspeed Venture Partners", "Insight Partners",
    "General Catalyst", "Founders Fund", "Y Combinator", "First Round Capital",
    "Union Square Ventures", "Tiger Global", "SoftBank", "Coatue",
    "Menlo Ventures", "Redpoint Ventures", "NEA", "Battery Ventures",
    "Felicis Ventures", "GV", "Khosla Ventures", "Matrix Partners",
    "Mayfield Fund", "Spark Capital", "Venrock", "8VC",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_page_text(url: str, max_chars: int = 800) -> str:
    """Download a URL and extract main text via trafilatura."""
    try:
        import trafilatura
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=_HEADERS)
        if resp.status_code != 200:
            return ""
        text = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
        return (text or "")[:max_chars]
    except Exception:
        return ""


def _search_vc(query: str, max_results: int, deep: bool = False) -> list[dict]:
    """Run a web search and optionally deep-scrape result pages."""
    results = web_search(query, max_results=max_results)
    if not results:
        return []
    if deep:
        for r in results:
            url = r.get("href", "")
            if url:
                page_text = _fetch_page_text(url)
                if page_text:
                    r["page_text"] = page_text
    return results


def _extract_vc_names(text: str) -> set[str]:
    """Extract likely VC firm names from a text string."""
    found = set()

    # Pattern-based extraction
    for pattern in _VC_NAME_PATTERNS:
        for match in pattern.finditer(text):
            name = match.group(1).strip().rstrip(",").rstrip(".")
            # Filter out false positives
            if len(name) > 3 and not name.lower().startswith(("the ", "this ", "that ")):
                found.add(name)

    # Known VC check (catch names that don't match patterns)
    text_lower = text.lower()
    for name in _KNOWN_VC:
        if name.lower() in text_lower:
            found.add(name)

    return found


def _crunchbase_search_vc_firms(market: str, limit: int = 5) -> list[dict]:
    """Search Crunchbase Basic API for investor organizations in the market."""
    if not CRUNCHBASE_API_KEY:
        return []

    try:
        headers = {
            "X-API-Key": CRUNCHBASE_API_KEY,
            "Content-Type": "application/json",
        }
        payload = {
            "field_ids": [
                "name",
                "short_description",
                "website_url",
                "location_identifiers",
            ],
            "query": [
                {
                    "type": "predicate",
                    "field_id": "facet_ids",
                    "operator_id": "includes",
                    "values": ["investor"],
                }
            ],
            "limit": limit,
        }
        resp = requests.post(
            f"{CRUNCHBASE_API_URL}/searches/organizations",
            json=payload,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            logger.warning(f"Crunchbase API returned {resp.status_code}: {resp.text[:200]}")
            return []

        data = resp.json()
        results = []
        for entity in data.get("entities", []):
            props = entity.get("properties", {})
            locs = props.get("location_identifiers") or []
            results.append({
                "name": props.get("name", "Unknown"),
                "description": props.get("short_description", ""),
                "website": props.get("website_url", ""),
                "location": locs[0].get("value", "") if locs else "",
            })
        return results

    except Exception as e:
        logger.warning(f"Crunchbase API search failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_vcs(
    market: str,
    idea_name: str,
    idea_content: str,
    pain_point: str,
    max_results: int = 5,
    max_deep_scrape: int = 2,
    max_chars: int = 3000,
) -> str:
    """
    Search for Venture Capital firms investing in the given market/domain.

    Runs multiple Serper queries in parallel. If the ``CRUNCHBASE_API_KEY``
    environment variable is set, also queries the Crunchbase Basic API for
    investor organizations.

    Args:
        market: The market/domain (e.g. 'Fitness Apps').
        idea_name: The generated startup name.
        idea_content: The startup idea description.
        pain_point: The selected pain point the idea addresses.
        max_results: Number of results per Serper query (configurable depth).
        max_deep_scrape: Number of queries whose results get deep-scraped.
        max_chars: Maximum report length (0 = no limit).

    Returns:
        A structured markdown report of relevant VC firms.
    """
    # -- Queries that surface VC firm names ---------------------------------
    vc_queries = [
        f'"top venture capital" "{market}" list firms',
        f'"{market}" startup "raised" "led by" OR "from" funding',
        f'"{market}" "series" funding investor venture capital',
    ]

    # -- General queries (original) -----------------------------------------
    general_queries = [
        f'"venture capital" "{market}" investors funding portfolio',
        f'"{market}" startups raised "series A" OR seed investors funding',
        f'crunchbase "{market}" investors funding',
        f'angellist "{market}" venture capital investors',
        f'"{idea_name}" similar startups funded investors',
        f'"{market}" "pain point" startup funding investors venture',
    ]

    all_queries = vc_queries + general_queries
    num_vc_queries = len(vc_queries)

    # -- Parallel search ----------------------------------------------------
    all_results: list[dict] = []
    seen_urls: set[str] = set()
    all_vc_names: set[str] = set()

    def _search(query: str, idx: int):
        deep = idx < max_deep_scrape
        results = _search_vc(query, max_results=max_results, deep=deep)
        for r in results:
            # Extract VC names from every result
            text = f"{r.get('title', '')} {r.get('body', '')} {r.get('page_text', '')}"
            all_vc_names.update(_extract_vc_names(text))

            url = r.get("href", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)

    with ThreadPoolExecutor(max_workers=len(all_queries)) as pool:
        futures = {pool.submit(_search, q, i): q for i, q in enumerate(all_queries)}
        for _ in as_completed(futures):
            pass

    # -- Optional Crunchbase search -----------------------------------------
    cb_firms = _crunchbase_search_vc_firms(market, limit=max_results)
    for firm in cb_firms:
        if firm["name"]:
            all_vc_names.add(firm["name"])

    # -- Deduplicate by title -----------------------------------------------
    seen_titles: set[str] = set()
    unique_results: list[dict] = []
    for r in all_results:
        title = r.get("title", "")
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_results.append(r)

    # -- Build the markdown report ------------------------------------------
    report = f"## VC Investment Landscape: {market}\n\n"

    # VC Firms section (new)
    if all_vc_names:
        _VC_SUFFIXES = ("Capital", "Ventures", "Partners", "Equity", "Fund", "Group", "Investments", "Global", "Growth", "Associates", "Advisors")

        clean_names = set()
        for name in sorted(all_vc_names, key=lambda x: -len(x)):
            lower = name.lower()
            if len(name) < 4:
                continue
            # Skip obviously generic phrases
            if any(lower.startswith(skip) for skip in [
                "top ", "list ", "best ", "series ", "how to ", "guide ",
            ]):
                continue
            if lower in ("top 13 global",):
                continue
            # Strip leading noise words
            for prefix in ("Crunchbase ", "AngelList ", "Site:"):
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break
            # Skip sentences / long phrases (VC names are 1–5 words)
            if len(name.split()) > 5:
                continue
            # Skip phrases that contain "has" / "including" / "today" etc.
            if any(w in name.lower().split() for w in ("has", "including", "today", "get", "find", "with", "their")):
                continue
            # Must look like a firm name: ends with a VC suffix OR is a known VC
            if not (name.endswith(_VC_SUFFIXES) or name in _KNOWN_VC):
                continue
            clean_names.add(name)

        if clean_names:
            report += "### VC Firms\n"
            for name in sorted(clean_names)[:15]:
                report += f"- **{name}**\n"
            report += "\n"

    # Crunchbase section (kept as-is)
    if cb_firms:
        report += "### Crunchbase — VC Firms in this Space\n"
        for firm in cb_firms:
            name = firm["name"]
            desc = firm["description"]
            website = firm["website"]
            location = firm["location"]
            report += f"- **{name}**"
            if location:
                report += f" ({location})"
            report += "\n"
            if desc:
                report += f"  - {desc}\n"
            if website:
                report += f"  - [{website}]({website})\n"
        report += "\n"

    # Web search results (kept as-is)
    if unique_results:
        report += "### Web Search — Investors & Funding Activity\n"
        for i, r in enumerate(unique_results[: max_results * 2], 1):
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            page_text = r.get("page_text", "")

            report += f"{i}. **[{title}]({href})**\n"
            if body:
                report += f"   {body}\n"
            if page_text:
                excerpt = page_text[:300]
                report += f"   > {excerpt}\n"
            report += "\n"
    else:
        report += "No VC-related results found via web search.\n"

    if max_chars and len(report) > max_chars:
        report = report[: max_chars - 60] + "\n\n… (truncated for brevity)"

    return report


# ---------------------------------------------------------------------------
# CLI test entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test VC research for a startup idea.")
    parser.add_argument("--market", default="Fitness Apps", help="Market / domain")
    parser.add_argument("--name", default="FitProgress", help="Startup idea name")
    parser.add_argument("--desc", default="A fitness tracking app that accurately logs workouts, tracks progress over time, and provides personalized training recommendations.", help="Startup idea description")
    parser.add_argument("--pain-point", default="Users cannot accurately track their long-term fitness progress across different workout types.", help="Pain point being solved")
    parser.add_argument("--max-results", type=int, default=5, help="Results per query")
    parser.add_argument("--deep", type=int, default=2, help="Number of queries to deep-scrape")
    parser.add_argument("--no-truncate", action="store_true", help="Disable output truncation")

    args = parser.parse_args()

    print(f"Searching VCs for: {args.market}\n")

    report = find_vcs(
        market=args.market,
        idea_name=args.name,
        idea_content=args.desc,
        pain_point=args.pain_point,
        max_results=args.max_results,
        max_deep_scrape=args.deep,
        max_chars=0 if args.no_truncate else 3000,
    )

    print(report)
