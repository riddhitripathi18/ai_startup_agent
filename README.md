# AI Startup Idea Generator

A Streamlit-powered agent that researches market pain points, generates unique startup ideas, judges them for originality, and runs live market/competitor/VC analysis — all powered by local LLMs (Ollama) and real-time web search.

## Architecture

```
User Input (Market)  ──►  Pain Point Discovery (9 sources)
                              │
                    ┌─────────▼─────────┐
                    │   Top 3 Pain Pts  │
                    └─────────┬─────────┘
                              │ User selects one
                    ┌─────────▼─────────┐
                    │  LLM generates    │
                    │  startup idea     │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Judge            │
                    │  (embeddings +    │
                    │   LLM-as-Judge)   │
                    └─────────┬─────────┘
                         ┌────┴────┐
                      REJECTED  APPROVED
                         │         │
                    "Try Again"   ▼
                    ┌─────────────────────┐
                    │  Analysis Agent     │
                    │  • deep_web_research│
                    │  • analyze_market   │
                    │  • analyze_opp      │
                    │  • analyze_comp     │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  VC Research        │
                    │  (Serper + optional │
                    │   Crunchbase API)   │
                    └──────────┬──────────┘
                               ▼
                    Saved to SQLite + Displayed
```

## Prerequisites

- **Ollama** with model `qwen3:8b` pulled:
  ```bash
  ollama pull qwen3:8b
  ```
- **Python 3.11+**

### Search backend (pick one)

| Backend | Setup |
|---------|-------|
| **Serper** (default) | Get a free API key at https://serper.dev, set `SERPER_API_KEY` in `.env` |
| **DuckDuckGo** (free) | Set `SEARCH_PROVIDER=duckduckgo` in `.env` — no API key needed |

### Optional

- **Crunchbase Basic API key** — free tier at https://developers.crunchbase.com (set in `.env` for structured VC data)

## Setup

```bash
# Clone and enter the project
cd ai_startup_agent

# Create environment (using uv or venv)
uv sync
# OR
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Create .env — pick a search backend
echo "SERPER_API_KEY=your_key_here" > .env    # Serper (default)
# OR
echo "SEARCH_PROVIDER=duckduckgo" > .env      # DuckDuckGo (free, no key)

# Optional: echo "CRUNCHBASE_API_KEY=your_key_here" >> .env
```

## Run

```bash
streamlit run agent.py
```

## Project Structure

| File | Purpose |
|------|---------|
| `agent.py` | Streamlit UI — the main entry point and pipeline orchestrator |
| `pain_points.py` | Concurrent web scraping across 9 sources (Play Store, Reddit, HN, Product Hunt, G2, Trustpilot, Stack Overflow, blogs, Quora) |
| `opportunity_analysis.py` | LangChain tools for market, opportunity, and competitor analysis |
| `web_research.py` | Deep web research tool (multi-query Serper + trafilatura deep scrape) |
| `vc_research.py` | VC investment landscape research (Serper + optional Crunchbase API) |
| `judge.py` | Two-stage originality judge (cosine similarity on embeddings + LLM-as-a-Judge) |
| `embeddings.py` | HuggingFace SentenceTransformer embeddings for semantic similarity |
| `database.py` | SQLite persistence for ideas and pain points |
| `search_client.py` | Shared search wrapper — dispatches to Serper or DuckDuckGo based on `SEARCH_PROVIDER` |

## How It Works

1. **Pain Point Discovery** — Enter a market (e.g. "Fitness Apps"). The system concurrently scrapes 9 sources via `ThreadPoolExecutor` and uses an LLM to extract the top 3 pain points.

2. **Idea Generation** — Select a pain point. The LLM generates a unique startup name and description, with awareness of previously saved ideas to avoid duplicates.

3. **Judge** — Two-stage originality check:
   - **Stage 1**: Cosine similarity on 384-dim embeddings (> 0.85 → reject)
   - **Stage 2**: LLM-as-a-Judge for semantic nuance

4. **Analysis** — A LangChain agent with 4 tools runs live web research, market sizing, opportunity evaluation, and competitor analysis.

5. **VC Research** — Searches for VCs that invest in the market via Serper queries and (optionally) the Crunchbase Basic API.

6. **Persistence** — Approved ideas + all analyses are saved to `startup_ideas.db` and displayed in the sidebar.

## Configuration

Set these in `.env`:

| Variable | Required | Description |
|----------|----------|-------------|
| `SEARCH_PROVIDER` | No | `"serper"` (default) or `"duckduckgo"` |
| `SERPER_API_KEY` | With Serper | Google Search API key from https://serper.dev |
| `CRUNCHBASE_API_KEY` | No | Crunchbase Basic API key for structured VC data |

## Extending

To add a new tool to the pipeline:

1. Create a module with your function (e.g. `my_tool.py`)
2. Import and call it in `agent.py` at the appropriate Step
3. Add any required environment variables to `.env`

For LangChain agent tools, add `@tool`-decorated functions to `opportunity_analysis.py` and include them in the `analysis_agent` tool list.
