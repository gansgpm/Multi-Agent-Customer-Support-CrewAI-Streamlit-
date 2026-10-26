"""
Multi-Agent Customer Support Crew
Social Eagle - Gen AI Architect Program | Weekly Buildathon (CrewAI + Streamlit)

Three agents run one after another (sequential process):
  1. Assistant             -> answers the query from its own knowledge
  2. Web Search Assistant  -> searches the web (Serper) and answers from the results
  3. Entry Agent           -> writes query + answer 1 + answer 2 to answers.txt
                              and returns both answers to the UI

Run:  streamlit run app.py
"""

import os
import threading
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load API keys from the .env file BEFORE importing CrewAI tools,
# because SerperDevTool reads SERPER_API_KEY from the environment.
# override=True -> values in .env win over any key already set in the
# terminal (e.g. a leftover  $env:OPENAI_API_KEY="your-openai-key").
load_dotenv(override=True)

import streamlit as st
from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Configuration (nothing secret is hard-coded - keys come from .env)
# --------------------------------------------------------------------------
ANSWERS_FILE = Path(__file__).parent / "answers.txt"
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
REQUIRED_KEYS = ["OPENAI_API_KEY", "SERPER_API_KEY"]


# --------------------------------------------------------------------------
# Structured result the Entry Agent hands back to the UI
# --------------------------------------------------------------------------
class SupportResponse(BaseModel):
    query: str = Field(description="The customer's original query")
    assistant_answer: str = Field(description="Answer 1 - from the Assistant")
    web_search_answer: str = Field(description="Answer 2 - from the Web Search Assistant")
    saved_to: str = Field(description="Name of the file the entry was saved to")


# --------------------------------------------------------------------------
# Tool used by the Entry Agent to write the .txt file
# --------------------------------------------------------------------------
# The model sometimes fires the same tool call twice in parallel; this guard
# makes sure one query is written to answers.txt only once per run.
_save_lock = threading.Lock()
_saved_entries = set()


@tool("Save Answers To File")
def save_answers_tool(query: str, assistant_answer: str, web_search_answer: str) -> str:
    """Append the customer query, the Assistant's answer and the Web Search
    Assistant's answer to answers.txt. Pass each answer exactly as it was written."""
    with _save_lock:
        key = (query.strip(), assistant_answer.strip(), web_search_answer.strip())
        if key in _saved_entries:
            return f"Already saved to {ANSWERS_FILE.name} - no need to save again"
        _saved_entries.add(key)
        return _append_entry(query, assistant_answer, web_search_answer)


def _append_entry(query: str, assistant_answer: str, web_search_answer: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = (
        f"{'=' * 70}\n"
        f"Timestamp : {timestamp}\n"
        f"Query     : {query}\n"
        f"{'-' * 70}\n"
        f"ANSWER 1 - Assistant (own knowledge):\n{assistant_answer.strip()}\n"
        f"{'-' * 70}\n"
        f"ANSWER 2 - Web Search Assistant:\n{web_search_answer.strip()}\n\n"
    )
    with open(ANSWERS_FILE, "a", encoding="utf-8") as f:
        f.write(entry)
    return f"Saved the query and both answers to {ANSWERS_FILE.name}"


# --------------------------------------------------------------------------
# Build the crew: 3 agents + 3 tasks, sequential process
# --------------------------------------------------------------------------
def build_crew(task_callback=None) -> Crew:
    llm = LLM(model=MODEL_NAME, temperature=0.3)

    # Agent 1 - answers directly, no tools
    assistant = Agent(
        role="Assistant",
        goal="Answer the customer's query directly, clearly and accurately "
        "using your own knowledge.",
        backstory="You are a friendly, experienced customer support agent. You "
        "give practical, step-by-step answers and say so honestly when "
        "something depends on the specific product or account.",
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    # Agent 2 - the ONLY agent that holds the web-search tool
    web_search_assistant = Agent(
        role="Web Search Assistant",
        goal="Search the web for the customer's query and write an answer "
        "based on what the search results say.",
        backstory="You are a research-minded support agent. You never answer "
        "from memory alone - you search first, then answer only from the "
        "results and cite where the information came from.",
        tools=[SerperDevTool(n_results=5)],
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    # Agent 3 - records everything and returns both answers
    entry_agent = Agent(
        role="Entry Agent",
        goal="Record the customer query and both answers in answers.txt, "
        "then return both answers to the user unchanged.",
        backstory="You are a meticulous support desk record-keeper. You copy "
        "text exactly as written - you never summarise, shorten or edit answers.",
        tools=[save_answers_tool],
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    assistant_task = Task(
        description=(
            "Customer query: {query}\n\n"
            "Answer this query directly from your own knowledge. Do not use any tools. "
            "Be concise and practical; use numbered steps where it helps."
        ),
        expected_output="A clear, helpful answer to the query in under 200 words.",
        agent=assistant,
    )

    web_search_task = Task(
        description=(
            "Customer query: {query}\n\n"
            "Use the search tool to look up this query on the web. Then write an "
            "answer based only on the search results. End with a 'Sources:' list "
            "of the URLs you used."
        ),
        expected_output=(
            "An answer grounded in the web search results (under 200 words), "
            "followed by a 'Sources:' list of URLs."
        ),
        agent=web_search_assistant,
        # Empty context = this agent does NOT see Answer 1, so the two answers
        # stay independent (one from knowledge, one from the web).
        context=[],
    )

    entry_task = Task(
        description=(
            "Customer query: {query}\n\n"
            "You have received two answers in your context: the Assistant's answer "
            "and the Web Search Assistant's answer.\n"
            "1. Call the 'Save Answers To File' tool ONCE with the query, the "
            "Assistant's answer and the Web Search Assistant's answer, each copied "
            "exactly as written.\n"
            "2. Return the query, both answers (unchanged) and the file name."
        ),
        expected_output="The query, both answers unchanged, and the file name they were saved to.",
        agent=entry_agent,
        context=[assistant_task, web_search_task],  # needs BOTH earlier answers
        output_pydantic=SupportResponse,
    )

    return Crew(
        agents=[assistant, web_search_assistant, entry_agent],
        tasks=[assistant_task, web_search_task, entry_task],
        process=Process.sequential,
        verbose=True,
        task_callback=task_callback,
    )


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="Support Crew", page_icon="🤝", layout="wide")
st.title("🤝 Multi-Agent Customer Support")
st.caption("CrewAI sequential crew: Assistant → Web Search Assistant → Entry Agent")

def key_problem(name: str):
    """Return why a key is unusable, or None if it looks fine."""
    value = os.getenv(name, "").strip()
    if not value:
        return "missing"
    if value.lower().startswith("your-"):
        return "still the placeholder text from .env.example"
    return None


key_issues = {k: key_problem(k) for k in REQUIRED_KEYS if key_problem(k)}

with st.sidebar:
    st.subheader("Setup")
    for key in REQUIRED_KEYS:
        st.write(("❌ " if key in key_issues else "✅ ") + key)
    st.write(f"Model: `{MODEL_NAME}`")
    st.divider()
    if ANSWERS_FILE.exists():
        st.download_button(
            "Download answers.txt",
            data=ANSWERS_FILE.read_text(encoding="utf-8"),
            file_name="answers.txt",
            mime="text/plain",
        )
    else:
        st.caption("answers.txt will appear here after the first query.")

if key_issues:
    problems = "\n".join(f"- **{k}**: {why}" for k, why in key_issues.items())
    st.error(
        "Fix these API keys in your **.env** file (same folder as app.py), "
        f"save it, then refresh this page:\n\n{problems}"
    )
    st.stop()

with st.form("query_form"):
    query = st.text_area(
        "Enter your query or task",
        placeholder="e.g. How do I reset my password?",
        height=100,
    )
    submitted = st.form_submit_button("Ask the crew", type="primary")

if submitted:
    if not query.strip():
        st.warning("Please type a query first.")
        st.stop()

    # Live progress while the crew runs (labels for after task 1 and task 2)
    next_step = [
        "🌐 Web Search Assistant is searching the web...",
        "📝 Entry Agent is saving to answers.txt...",
    ]
    with st.status("🧠 Assistant is answering from its own knowledge...", expanded=True) as status:

        def on_task_done(output):
            # CrewAI calls this after each task finishes.
            st.write(f"✅ {output.agent} finished")
            if next_step:
                status.update(label=next_step.pop(0))

        try:
            result = build_crew(task_callback=on_task_done).kickoff(
                inputs={"query": query.strip()}
            )
        except Exception as e:
            status.update(label="Something went wrong", state="error")
            st.exception(e)
            st.stop()
        status.update(label="All three agents finished", state="complete", expanded=False)

    # Prefer the Entry Agent's structured return; fall back to raw task outputs.
    response = result.pydantic
    if response is None:
        outputs = result.tasks_output
        response = SupportResponse(
            query=query.strip(),
            assistant_answer=outputs[0].raw,
            web_search_answer=outputs[1].raw,
            saved_to=ANSWERS_FILE.name,
        )
    st.session_state["last_response"] = response

# Show the latest result (stays on screen across reruns)
if "last_response" in st.session_state:
    r = st.session_state["last_response"]
    st.subheader(f"Query: {r.query}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🧠 Answer 1 — Assistant")
        st.markdown(r.assistant_answer)
    with col2:
        st.markdown("#### 🌐 Answer 2 — Web Search Assistant")
        st.markdown(r.web_search_answer)
    st.success(f"Entry Agent saved the query and both answers to {r.saved_to}")
