"""
Analysis Tools
==============
LangChain tools that perform real web searches to gather market,
opportunity, and competitor data. Each tool uses Serper (Google Search API)
to fetch live information rather than relying on LLM training data.
"""

from langchain.tools import tool
from web_research import deep_web_research
from search_client import web_search
import logging

logger = logging.getLogger(__name__)


def _web_search(query: str, max_results: int = 3) -> str:
    """Run a Google search via Serper and return formatted snippets."""
    results = web_search(query, max_results=max_results)
    if not results:
        return "No results found."
    lines = []
    for r in results:
        title = r.get("title", "")
        body = r.get("body", "")
        lines.append(f"- **{title}**: {body}")
    return "\n".join(lines)


@tool
def analyze_market(topic: str) -> str:
    """Search the web for real market data about the specified topic including
    market size, segments, CAGR, growth drivers, risks, and industry outlook.

    Args:
        topic: The market or industry to analyze (e.g. 'AI fitness coaching').

    Returns:
        A structured summary of real web search results covering market data.
    """
    queries = {
        "Market Size & Structure": f'"{topic}" market size OR market value OR segments OR "value chain" 2025 OR 2026',
        "CAGR & Growth Rate": f'"{topic}" CAGR OR "growth rate" OR "compound annual" forecast',
        "Growth Drivers": f'"{topic}" growth drivers OR demand drivers OR opportunities OR trends',
        "Risks & Challenges": f'"{topic}" risks OR challenges OR barriers OR headwinds OR threats',
    }

    report = f"## Market Analysis: {topic}\n\n"
    for section, query in queries.items():
        data = _web_search(query, max_results=2)
        report += f"### {section}\n{data}\n\n"
    print("MAr Ananlysis toll in use")
    return report


@tool
def analyze_opportunity(topic: str) -> str:
    """Search the web for business opportunity data about the specified topic
    including customer pain points, value propositions, revenue models,
    competitive advantages, and execution challenges.

    Args:
        topic: The business opportunity or market to evaluate.

    Returns:
        A structured summary of real web search results covering opportunity data.
    """
    queries = {
        "Market Need & Pain Points": f'"{topic}" customer pain points OR unmet needs OR problems users face',
        "Value Propositions": f'"{topic}" value proposition OR unique solution OR benefits OR innovation',
        "Revenue Models": f'"{topic}" revenue model OR pricing OR monetization OR business model',
        "Execution Challenges": f'"{topic}" execution challenges OR barriers to entry OR implementation difficulty',
    }

    report = f"## Opportunity Analysis: {topic}\n\n"
    for section, query in queries.items():
        data = _web_search(query, max_results=2)
        report += f"### {section}\n{data}\n\n"
    print("OPp Ananlysis toll in use")

    return report


@tool
def analyze_competitors(topic: str) -> str:
    """Search the web for competitor data in the specified market including
    top companies, their strengths/weaknesses, funding, and market gaps.

    Args:
        topic: The market or business domain to find competitors for.

    Returns:
        A structured summary of real web search results covering competitor data.
    """
    queries = {
        "Top Competitors": f'"{topic}" top companies OR competitors OR market leaders OR startups',
        "Strengths & Weaknesses": f'"{topic}" competitor strengths OR weaknesses OR pros OR cons reviews',
        "Funding & Traction": f'"{topic}" startup funding OR raised OR Series A OR valuation OR users',
        "Whitespace Opportunities": f'"{topic}" market gaps OR underserved OR unmet needs OR opportunities',
    }

    report = f"## Competitor Analysis: {topic}\n\n"
    for section, query in queries.items():
        data = _web_search(query, max_results=2)
        report += f"### {section}\n{data}\n\n"
    print("Comp Ananlysis toll in use")
    return report
