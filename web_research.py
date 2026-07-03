"""
Deep Web Research Tool
======================
A LangChain tool that performs targeted, multi-query web searches and
page-level content extraction to gather *real* market data, competitor
info, industry trends, and pain points for a given domain.

Performance: uses ThreadPoolExecutor for parallel page fetches.
All HTTP requests have an 8-second timeout.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.tools import tool
from search_client import web_search
import requests
import trafilatura

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 8  # seconds


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_page_text(url: str, max_chars: int = 500) -> str:
    """Download a URL and extract readable text via trafilatura."""
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            )
        })
        if resp.status_code != 200:
            return ""
        text = trafilatura.extract(
            resp.text,
            include_comments=False,
            include_tables=False,
        )
        return (text or "")[:max_chars]
    except Exception:
        return ""


def _search_and_extract(query: str, max_results: int = 3, deep: bool = True) -> str:
    """Run a Google search via Serper, optionally deep-scrape each result page."""
    results = web_search(query, max_results=max_results)

    if not results:
        return ""

    if not deep:
        # Return snippets only
        lines = []
        for r in results:
            lines.append(f"- **{r.get('title', '')}**: {r.get('body', '')}")
        return "\n".join(lines)

    # Deep-scrape pages in parallel
    page_texts: dict[int, str] = {}

    def _fetch(idx, url):
        return idx, _fetch_page_text(url)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {
            pool.submit(_fetch, i, r.get("href", "")): i
            for i, r in enumerate(results) if r.get("href")
        }
        for f in as_completed(futs):
            idx, txt = f.result()
            page_texts[idx] = txt

    lines = []
    for i, r in enumerate(results):
        title = r.get("title", "")
        body = r.get("body", "")
        page = page_texts.get(i, "")
        excerpt = ""
        if page:
            excerpt = page[:300]
            excerpt = f"\n  > {excerpt}"
        lines.append(f"- **{title}**: {body}{excerpt}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The LangChain tool
# ---------------------------------------------------------------------------

@tool
def deep_web_research(query: str) -> str:
    """Search the web for real-time domain-specific market data, competitor info,
    industry trends, and user pain points. Use this tool FIRST before any
    analysis to gather live, grounded data.

    Args:
        query: The market domain, startup idea, or topic to research
               (e.g. 'AI fitness coaching apps' or 'telemedicine startups').

    Returns:
        A structured text report with data from multiple web sources.
    """
    sections: dict[str, str] = {}

    # Define research queries — each targets a different angle.
    # Drastically reduced the max_results to speed up the LLM inference
    research_queries = [
        (
            "Market Size & Trends",
            f'"{query}" market size OR market value OR CAGR OR growth rate 2025 OR 2026',
            1, True,  # Only 1 deep result
        ),
        (
            "Key Competitors & Players",
            f'"{query}" top companies OR competitors OR market leaders OR startups',
            2, False, # Snippets only
        ),
        (
            "Recent Funding & Investments",
            f'"{query}" funding OR investment OR raised OR seed OR Series A 2025 OR 2026',
            1, False, # Snippets only
        ),
        (
            "User Complaints & Pain Points",
            f'"{query}" complaints OR problems OR frustrations OR "pain points" OR challenges',
            2, False, # Snippets only
        ),
        (
            "Industry Expert Opinions",
            f'"{query}" trends OR predictions OR expert OR analysis OR future',
            1, True,  # Only 1 deep result
        ),
    ]

    def _run_query(section_title, q, max_r, deep):
        try:
            return section_title, _search_and_extract(q, max_results=max_r, deep=deep)
        except Exception as e:
            logger.error(f"Research section '{section_title}' failed: {e}")
            return section_title, ""

    # Run all research queries in parallel
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {
            pool.submit(_run_query, title, q, max_r, deep): title
            for title, q, max_r, deep in research_queries
        }
        for f in as_completed(futures):
            title, data = f.result()
            sections[title] = data

    # Build the report in defined order
    report = f"## Web Research Report: {query}\n\n"
    for title, _, _, _ in research_queries:
        data = sections.get(title, "").strip()
        report += f"### {title}\n"
        if data:
            report += data + "\n\n"
        else:
            report += "No data found for this category.\n\n"

    # Cap total output to keep within LLM context limits
    # A smaller context (2000 chars) ensures the local LLM runs much faster
    if len(report) > 2000:
        report = report[:1900] + "\n\n… (truncated for brevity)"

    return report
