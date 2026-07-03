import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from google_play_scraper import search, reviews, Sort
from search_client import web_search
import requests
import trafilatura

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT = 8  # seconds – strict cap per HTTP request

# Mapping of user inputs to broader/related search terms so we don't
# miss results by searching for the exact phrase only.
_SECTOR_SYNONYMS = {
    "ecommerce": ["online store", "online shopping", "dropshipping", "e-commerce", "shopify", "woocommerce", "online retail"],
    "fitness": ["workout", "exercise", "gym", "health tracking", "wellness", "personal training"],
    "edtech": ["online learning", "education technology", "e-learning", "online courses", "LMS"],
    "fintech": ["digital banking", "payments", "neobank", "financial technology", "mobile banking"],
    "healthtech": ["telemedicine", "health technology", "digital health", "remote patient monitoring"],
    "foodtech": ["food delivery", "meal kit", "restaurant technology", "cloud kitchen"],
}


def _expand_sector(sector: str) -> list[str]:
    """Generate a list of related search terms from the user's input.
    Always includes the original input plus broader variations."""
    terms = [sector]
    sector_lower = sector.lower().strip()

    # Check the synonym map
    for key, synonyms in _SECTOR_SYNONYMS.items():
        if key in sector_lower or sector_lower in key:
            terms.extend(synonyms)
            break

    # Always add generic variations by splitting multi-word inputs
    words = sector_lower.split()
    if len(words) >= 2:
        # Add individual meaningful words (skip very short ones)
        for w in words:
            if len(w) > 3 and w not in terms:
                terms.append(w)

    # Keep unique, max 5 terms to avoid too many queries
    seen = set()
    unique = []
    for t in terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique.append(t)
    return unique[:5]


def _build_or_query(terms: list[str]) -> str:
    """Join multiple terms into an OR query string for Google/Serper."""
    if len(terms) == 1:
        return terms[0]
    return " OR ".join(f'"{t}"' for t in terms)


def _fetch_page_text(url: str, max_chars: int = 1500) -> str:
    """Download a URL and extract its main text content via trafilatura.
    Returns at most *max_chars* characters. Returns '' on any failure."""
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        if resp.status_code != 200:
            return ""
        text = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
        return (text or "")[:max_chars]
    except Exception:
        return ""


def _search_snippets(query: str, max_results: int = 5) -> list[dict]:
    """Run a Google search via Serper API and return result dicts."""
    return web_search(query, max_results=max_results)


def _format_results(results: list[dict], deep_scrape: bool = False) -> str:
    """Turn DDG result dicts into markdown bullet points.
    If *deep_scrape* is True, also fetch the page text for richer content."""
    lines = []
    for r in results:
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")

        extra = ""
        if deep_scrape and href:
            page_text = _fetch_page_text(href, max_chars=800)
            if page_text:
                extra = f"\n  > {page_text[:400]}…" if len(page_text) > 400 else f"\n  > {page_text}"

        lines.append(f"- **{title}**: {body}{extra}")
    return "\n".join(lines) if lines else ""

# ---------------------------------------------------------------------------
# Source 1: Google Play Store
# ---------------------------------------------------------------------------

def get_play_store_complaints(sector: str, max_apps: int = 2, max_reviews: int = 3) -> str:
    """Search for apps related to the sector and fetch negative reviews."""
    try:
        # Search using expanded terms for better coverage
        terms = _expand_sector(sector)
        all_app_ids = set()
        app_results = []
        for term in terms[:3]:
            hits = search(term, n_hits=max_apps)
            for app in hits:
                if app['appId'] not in all_app_ids:
                    all_app_ids.add(app['appId'])
                    app_results.append(app)
            if len(app_results) >= max_apps:
                break

        if not app_results:
            return "No relevant apps found on the Play Store."

        complaints = []
        for app_info in app_results[:max_apps]:
            app_id = app_info['appId']
            app_title = app_info['title']
            try:
                result, _ = reviews(
                    app_id,
                    lang='en',
                    country='us',
                    sort=Sort.MOST_RELEVANT,
                    count=max_reviews * 2
                )

                app_complaints = [r['content'] for r in result if r['score'] <= 3]

                if app_complaints:
                    complaints.append(f"#### App: {app_title}")
                    for complaint in app_complaints[:max_reviews]:
                        clean_complaint = complaint.replace('\n', ' ').strip()
                        complaints.append(f"- {clean_complaint}")
            except Exception as e:
                logger.warning(f"Failed to fetch reviews for {app_id}: {e}")
                continue

        if not complaints:
            return "No negative reviews found for apps in this sector."

        return "\n".join(complaints)
    except Exception as e:
        logger.error(f"Play Store scraping failed: {e}")
        return "Failed to fetch Play Store data."

# ---------------------------------------------------------------------------
# Source 2: Reddit (broadened queries)
# ---------------------------------------------------------------------------

def get_reddit_complaints(sector: str, max_results: int = 3) -> str:
    """Search Reddit for sector-wide complaints using broadened terms."""
    terms = _expand_sector(sector)
    or_terms = _build_or_query(terms)

    queries = [
        f'site:reddit.com ({or_terms}) problems OR complaints OR frustrating OR "pain points"',
        f'site:reddit.com ({or_terms}) worst OR terrible OR "I hate" OR "wish there was"',
        f'site:reddit.com/r/startups OR site:reddit.com/r/Entrepreneur ({or_terms}) problems',
    ]
    all_results = []
    for q in queries:
        all_results.extend(_search_snippets(q, max_results=2))
        if len(all_results) >= max_results * 2:
            break

    if not all_results:
        return "No relevant Reddit discussions found."

    # De-duplicate by title
    seen_titles = set()
    unique = []
    for r in all_results:
        t = r.get("title", "")
        if t not in seen_titles:
            seen_titles.add(t)
            unique.append(r)

    return _format_results(unique[:max_results], deep_scrape=False)

# ---------------------------------------------------------------------------
# Source 3: Hacker News
# ---------------------------------------------------------------------------

def get_hackernews_complaints(sector: str, max_results: int = 2) -> str:
    """Search Hacker News for developer/startup community pain points."""
    terms = _expand_sector(sector)
    or_terms = _build_or_query(terms)
    query = f'site:news.ycombinator.com ({or_terms}) problems OR challenges OR frustrations'
    results = _search_snippets(query, max_results=max_results)
    if not results:
        return "No relevant Hacker News discussions found."
    return _format_results(results, deep_scrape=False)

# ---------------------------------------------------------------------------
# Source 4: Product Hunt (deep scrape for richer feedback)
# ---------------------------------------------------------------------------

def get_producthunt_complaints(sector: str, max_results: int = 3) -> str:
    """Search Product Hunt for product feedback, feature gaps, and user reactions."""
    terms = _expand_sector(sector)
    or_terms = _build_or_query(terms)
    queries = [
        f'site:producthunt.com ({or_terms}) cons OR review OR feedback OR alternative',
        f'site:producthunt.com ({or_terms})',
    ]
    all_results = []
    for q in queries:
        all_results.extend(_search_snippets(q, max_results=max_results))
        if len(all_results) >= max_results:
            break

    if not all_results:
        return "No relevant Product Hunt discussions found."

    # De-duplicate
    seen = set()
    unique = []
    for r in all_results:
        t = r.get("title", "")
        if t not in seen:
            seen.add(t)
            unique.append(r)

    # Deep scrape Product Hunt pages to get actual user comments
    return _format_results(unique[:max_results], deep_scrape=True)

# ---------------------------------------------------------------------------
# Source 5: G2 / Capterra Reviews
# ---------------------------------------------------------------------------

def get_g2_capterra_complaints(sector: str, max_results: int = 2) -> str:
    """Search G2 and Capterra for enterprise/SaaS software review pain points."""
    terms = _expand_sector(sector)
    or_terms = _build_or_query(terms)
    query = f'(site:g2.com OR site:capterra.com) ({or_terms}) reviews OR complaints OR cons'
    results = _search_snippets(query, max_results=max_results)
    if not results:
        query = f'(site:g2.com OR site:capterra.com) {sector}'
        results = _search_snippets(query, max_results=max_results)
    if not results:
        return "No relevant G2/Capterra reviews found."
    return _format_results(results, deep_scrape=False)

# ---------------------------------------------------------------------------
# Source 6: Trustpilot
# ---------------------------------------------------------------------------

def get_trustpilot_complaints(sector: str, max_results: int = 2) -> str:
    """Search Trustpilot for consumer-facing service complaints."""
    terms = _expand_sector(sector)
    or_terms = _build_or_query(terms)
    query = f'site:trustpilot.com ({or_terms}) reviews'
    results = _search_snippets(query, max_results=max_results)
    if not results:
        return "No relevant Trustpilot reviews found."
    return _format_results(results, deep_scrape=False)

# ---------------------------------------------------------------------------
# Source 7: Stack Overflow / Dev Forums
# ---------------------------------------------------------------------------

def get_stackoverflow_complaints(sector: str, max_results: int = 2) -> str:
    """Search Stack Overflow for technical pain points and tooling gaps."""
    terms = _expand_sector(sector)
    or_terms = _build_or_query(terms)
    query = f'site:stackoverflow.com ({or_terms}) issues OR problems OR error OR alternative'
    results = _search_snippets(query, max_results=max_results)
    if not results:
        return "No relevant Stack Overflow discussions found."
    return _format_results(results, deep_scrape=False)

# ---------------------------------------------------------------------------
# Source 8: Industry Blogs & News (deep scrape)
# ---------------------------------------------------------------------------

def get_industry_blog_complaints(sector: str, max_results: int = 2) -> str:
    """Search industry blogs and news for market-level pain points."""
    terms = _expand_sector(sector)
    or_terms = _build_or_query(terms)
    query = f'({or_terms}) challenges OR problems OR "pain points" OR "biggest issues" 2025 OR 2026'
    results = _search_snippets(query, max_results=max_results)
    if not results:
        return "No relevant industry articles found."
    return _format_results(results, deep_scrape=True)

# ---------------------------------------------------------------------------
# Source 9: Quora
# ---------------------------------------------------------------------------

def get_quora_complaints(sector: str, max_results: int = 3) -> str:
    """Search Quora for user questions and frustrations about the sector."""
    terms = _expand_sector(sector)
    or_terms = _build_or_query(terms)
    queries = [
        f'site:quora.com ({or_terms}) problems OR frustrations OR "what is wrong with" OR "why is it hard"',
        f'site:quora.com ({or_terms}) "biggest challenge" OR "pain point" OR "wish there was"',
    ]
    all_results = []
    for q in queries:
        all_results.extend(_search_snippets(q, max_results=2))
        if len(all_results) >= max_results:
            break

    if not all_results:
        return "No relevant Quora discussions found."

    # De-duplicate by title
    seen = set()
    unique = []
    for r in all_results:
        t = r.get("title", "")
        if t not in seen:
            seen.add(t)
            unique.append(r)

    return _format_results(unique[:max_results], deep_scrape=False)

# ---------------------------------------------------------------------------
# Aggregator: gather_pain_points (runs all sources concurrently)
# ---------------------------------------------------------------------------

_SOURCES = [
    ("Google Play Store Complaints",   get_play_store_complaints,   {}),
    ("Reddit Discussions & Complaints", get_reddit_complaints,       {}),
    ("Hacker News Discussions",         get_hackernews_complaints,   {}),
    ("Product Hunt Feedback",           get_producthunt_complaints,  {}),
    ("G2 / Capterra Reviews",           get_g2_capterra_complaints,  {}),
    ("Trustpilot Reviews",              get_trustpilot_complaints,   {}),
    ("Stack Overflow / Dev Forums",     get_stackoverflow_complaints,{}),
    ("Industry Blogs & News",           get_industry_blog_complaints,{}),
    ("Quora Discussions",               get_quora_complaints,        {}),
]


def gather_pain_points(sector: str) -> str:
    """
    Scrape ALL sources concurrently and return a combined pain-points report.
    Uses ThreadPoolExecutor so total wall-clock time ≈ slowest single source.
    """
    results: dict[str, str] = {}

    def _run(title: str, func, kwargs):
        try:
            return title, func(sector, **kwargs)
        except Exception as e:
            logger.error(f"Source '{title}' failed: {e}")
            return title, f"Failed to fetch data from {title}."

    with ThreadPoolExecutor(max_workers=9) as pool:
        futures = {
            pool.submit(_run, title, func, kwargs): title
            for title, func, kwargs in _SOURCES
        }
        for future in as_completed(futures):
            title, data = future.result()
            results[title] = data

    # Build the report in source-order (not completion-order)
    report = f"## User Pain Points for '{sector}'\n\n"
    for title, _, _ in _SOURCES:
        report += f"### {title}\n"
        report += results.get(title, "No data.") + "\n\n"

    return report


if __name__ == "__main__":
    # Quick local test
    import time
    sector = "ecommerce platform"
    start = time.time()
    print(gather_pain_points(sector))
    print(f"\n⏱ Completed in {time.time() - start:.1f}s")
