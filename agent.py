import streamlit as st
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from opportunity_analysis import analyze_market, analyze_opportunity, analyze_competitors, deep_web_research
from database import init_db, save_idea, get_all_ideas, delete_idea
from judge import judge_idea
from pain_points import gather_pain_points
from vc_research import find_vcs

# =====================================
# DATABASE INIT
# =====================================
init_db()

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Startup Idea Generator",
    page_icon="🚀",
    layout="wide"
)

# =====================================
# SESSION STATE INIT
# =====================================
if "step" not in st.session_state:
    st.session_state.step = 0
if "market" not in st.session_state:
    st.session_state.market = ""
if "top_pain_points" not in st.session_state:
    st.session_state.top_pain_points = []
if "raw_pain_points" not in st.session_state:
    st.session_state.raw_pain_points = ""

def reset_state():
    st.session_state.step = 0
    st.session_state.top_pain_points = []
    st.session_state.raw_pain_points = ""

# =====================================
# SIDEBAR — Approved Startup Ideas
# =====================================

st.sidebar.title("💡 Approved Startup Ideas")
saved_ideas = get_all_ideas()

if not saved_ideas:
    st.sidebar.info("No approved ideas yet. Generate your first one!")
else:
    for idea in saved_ideas:
        with st.sidebar.expander(f"🚀 {idea['idea_name']}"):
            st.caption(f"Topic: {idea['topic']}  •  {idea['timestamp']}")
            st.markdown(idea["idea_content"])
            if st.button("🗑️ Delete", key=f"del_{idea['id']}"):
                delete_idea(idea["id"])
                st.rerun()

# =====================================
# HEADER
# =====================================

st.title("🚀 AI Startup Idea Generator")
st.write(
    "Enter a market or domain. The agent will find user pain points and present the top 3. "
    "Select one, and it will generate a targeted idea, judge it, and run a full market/opportunity analysis."
)

# =====================================
# LOAD AGENTS
# =====================================

@st.cache_resource
def load_analysis_agent():
    return create_agent(
        model="ollama:qwen3:8b",
        tools=[deep_web_research, analyze_market, analyze_opportunity, analyze_competitors],
        system_prompt=(
            "You are an expert business analyst. ALWAYS call the deep_web_research tool FIRST "
            "to gather real-time market data from the web. Then use analyze_market, "
            "analyze_opportunity, and analyze_competitors to structure your findings. "
            "Return a combined research brief (under 500 words) grounded in the live data "
            "you gathered, covering market structure, CAGR, growth drivers, risks, "
            "market need, value proposition, competitor landscape, strengths/weaknesses, "
            "and whitespace gaps. Cite specific data points from your web research."
        ),
    )

@st.cache_resource
def load_llm():
    return ChatOllama(
        model="qwen3:8b",
        temperature=0.7,
    )

try:
    analysis_agent = load_analysis_agent()
    idea_llm = load_llm()
except Exception as e:
    st.error(f"❌ Failed to load models:\n\n{e}")
    st.stop()

# =====================================
# STEP 1: USER INPUT & PAIN POINT GATHERING
# =====================================

market_input = st.text_input(
    "Market / Domain",
    value=st.session_state.market,
    placeholder="e.g. Fitness Apps, AI Education, HealthTech"
)

if market_input != st.session_state.market:
    st.session_state.market = market_input
    reset_state()

if st.session_state.step == 0:
    if st.button("🔍 Find Pain Points"):
        if not st.session_state.market.strip():
            st.warning("Please enter a market or domain.")
            st.stop()

        with st.spinner("🌐 Deep-scraping 8 sources for domain pain points (Play Store, Reddit, HN, Product Hunt, G2, Trustpilot, StackOverflow, blogs)..."):
            pain_points_text = gather_pain_points(st.session_state.market)
            st.session_state.raw_pain_points = pain_points_text
            
        with st.spinner("🧠 Extracting candidate pain points and checking against memory..."):
            extraction_prompt = f"""/no_think
You are a startup researcher analyzing the {st.session_state.market} sector.
Identify the top 5 most distinct, significant, and actionable industry-wide pain points or market gaps from the complaints below.

RULES:
- Each pain point must follow this format: "In {st.session_state.market}, [who] cannot [do what] because [structural reason], costing them [impact]."
- Keep each line under 70 words.
- Focus on: market inefficiencies, unserved customer segments, broken workflows, or missing infrastructure.
- Explicitly ban: UI bugs, app crashes, customer support issues, subscription pricing complaints.
- Do NOT name specific companies. Focus on the sector problem.
- Output EXACTLY 5 lines, each starting with "- ". Do not add any other text.

## Raw Complaints
{pain_points_text}
"""
            candidates = []
            try:
                response = idea_llm.invoke(extraction_prompt)
                candidates = [l.strip().lstrip('-').strip() for l in response.content.strip().split('\n') if l.strip().startswith('-')]
            except Exception as e:
                pass

            if len(candidates) < 3:
                candidates = [
                    f"In {st.session_state.market}, customers cannot access reliable services because of fragmented providers, costing them extra search time.",
                    f"In {st.session_state.market}, small businesses cannot scale operationally due to high overhead software costs, limiting their profit margins.",
                    f"In {st.session_state.market}, users cannot verify provider credentials quickly, resulting in security vulnerabilities and loss of trust."
                ]

            try:
                from embeddings import filter_and_ensure_unique_pain_points
                synthesized_needs = filter_and_ensure_unique_pain_points(candidates, st.session_state.market, idea_llm)
            except Exception as e:
                synthesized_needs = candidates[:3]

            st.session_state.top_pain_points = synthesized_needs[:3]
            st.session_state.step = 1
            st.rerun()



# =====================================
# STEP 2: USER SELECTION & IDEA GENERATION
# =====================================

if st.session_state.step == 1:
    with st.expander("📋 View Raw Discovered Pain Points", expanded=False):
        st.markdown(st.session_state.raw_pain_points)
        
    st.subheader("Top 3 User Complaints Discovered:")
    selected_pain_point = st.radio(
        "Select the problem you want your startup to solve:",
        st.session_state.top_pain_points
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🧠 Generate Idea"):
            st.session_state.selected_pain_point = selected_pain_point
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("🔄 Start Over"):
            reset_state()
            st.rerun()

# =====================================
# STEP 3: PIPELINE EXECUTION
# =====================================

if st.session_state.step == 2:
    selected_pain_point = st.session_state.selected_pain_point
    
    existing_ideas = get_all_ideas()
    existing_ideas_text = ""
    if existing_ideas:
        existing_ideas_text = "\n\n## IMPORTANT — These startup ideas already exist. You MUST NOT repeat them:\n"
        for i, idea in enumerate(existing_ideas[:10], 1):
            existing_ideas_text += f"\n{i}. **{idea['idea_name']}**: {idea['idea_content'][:200]}...\n"

    idea_prompt = f"""/no_think
You are an expert startup founder. You have identified the following specific user pain point in the {st.session_state.market} market:
"{selected_pain_point}"

Generate ONE unique and detailed startup idea that directly solves this exact problem.

{existing_ideas_text}

## Output Format (follow exactly)
STARTUP NAME: <catchy startup name>
---
<Detailed startup idea description covering:
- Problem being solved (reference the pain point above)
- Target customers
- Core product/service
- Revenue model
- Key differentiators
- Why now (timing advantage)
- Initial go-to-market strategy
Keep it between 200-400 words.>
"""

    with st.spinner("💡 Generating startup idea from selected pain point..."):
        idea_response = idea_llm.invoke(idea_prompt)
        idea_raw = idea_response.content.strip()

    # Parse startup name and description
    idea_name = "Unnamed Startup"
    idea_content = idea_raw

    if "STARTUP NAME:" in idea_raw:
        parts = idea_raw.split("---", 1)
        name_line = parts[0].strip()
        idea_name = name_line.replace("STARTUP NAME:", "").strip()
        if len(parts) > 1:
            idea_content = parts[1].strip()

    with st.spinner("🧑‍⚖️ Judge is reviewing for originality..."):
        verdict = judge_idea(st.session_state.market, idea_name, idea_content)

    if not verdict["approved"]:
        st.markdown(f"## ❌ {idea_name} (Rejected)")
        st.markdown(idea_content)
        st.warning(
            f"⚠️ **Judge REJECTED** — {verdict['reason']}\n\n"
            f"This idea was **not saved** because it is too similar to an existing idea. "
            f"Try a different angle or niche."
        )
        if st.button("🔄 Try Again"):
            st.session_state.step = 1
            st.rerun()
        st.stop()

    with st.spinner("📊 Running live web research + market, opportunity, and competitor analysis..."):
        analysis_inputs = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Analyze the market and competitors for this startup idea:\nName: {idea_name}\nDescription: {idea_content}"
                }
            ]
        }
        analysis_response = analysis_agent.invoke(analysis_inputs)
        analysis_text = ""
        if "messages" in analysis_response and analysis_response["messages"]:
            analysis_text = analysis_response["messages"][-1].content

    with st.spinner("💰 Searching for relevant VCs..."):
        vc_report = find_vcs(
            market=st.session_state.market,
            idea_name=idea_name,
            idea_content=idea_content,
            pain_point=st.session_state.selected_pain_point,
        )

    # Save and display
    st.markdown("---")
    combined_content = f"{idea_content}\n\n### Comprehensive Analysis\n{analysis_text}\n\n### VC Investment Landscape\n{vc_report}"
    save_idea(st.session_state.market, idea_name, combined_content)

    st.markdown(f"## 🚀 {idea_name}")
    st.markdown(idea_content)
    st.success(f"✅ **Judge APPROVED** — {verdict['reason']}")
    
    st.markdown("---")
    st.markdown("## 📊 Comprehensive Analysis")
    st.markdown(analysis_text)

    st.markdown("---")
    st.markdown("## 💰 VC Investment Landscape")
    st.markdown(vc_report)
    
    st.success("💾 Startup idea, analysis, and VC research saved to database!")
    
    if st.button("🔄 Create Another Idea"):
        reset_state()
        st.rerun()