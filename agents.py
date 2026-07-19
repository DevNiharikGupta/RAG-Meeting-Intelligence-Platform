import os
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "phi3")


# ---- Shared State ----
# This is the "bag of data" that gets passed between agents.
# Each agent reads what it needs and writes its results here.

class AgentState(TypedDict, total=False):
    user_input: str           # what the user asked
    file_path: str            # path to transcript file (for insights)
    route: str                # "question", "insights", or "both"
    retrieved_answer: str     # answer from Retrieval Agent
    retrieved_sources: list   # sources from Retrieval Agent
    insights: dict            # structured insights from Insight Agent
    final_response: dict      # combined output from Summary Agent


# ---- Agent 1: Router ----
# Looks at the user input and decides which agents to call.
# Uses the LLM to classify the request.

ROUTER_PROMPT = """You are a router. Read the user's request and classify it into ONE of these categories:

- "question" — the user is asking a specific question about a meeting
- "insights" — the user wants action items, decisions, or a summary extracted from a transcript
- "both" — the user wants to ask a question AND get insights

Reply with ONLY one word: question, insights, or both.

User request: {user_input}

Category:"""


def router_agent(state: AgentState) -> AgentState:
    """Decide what kind of request this is."""
    print("[Router Agent] Classifying request...")

    user_input = state["user_input"]
    file_path = state.get("file_path", "")

    # if there's no file_path, it can only be a question
    if not file_path:
        print("[Router Agent] No file path given → routing to 'question'")
        state["route"] = "question"
        return state

    # if there's no question text (just a file), it's insights only
    if not user_input or not user_input.strip():
        print("[Router Agent] No question given → routing to 'insights'")
        state["route"] = "insights"
        return state

    # both file and question provided — ask LLM to classify
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0, num_predict=10)
    result = llm.invoke(ROUTER_PROMPT.format(user_input=user_input))
    result = result.strip().lower()

    # extract just the keyword
    if "both" in result:
        route = "both"
    elif "insight" in result:
        route = "insights"
    else:
        route = "question"

    print(f"[Router Agent] Route decided: {route}")
    state["route"] = route
    return state


# ---- Agent 2: Retrieval Agent ----
# Searches ChromaDB and gets an answer using the RAG pipeline from query.py

def retrieval_agent(state: AgentState) -> AgentState:
    """Search ChromaDB and answer the question using RAG."""
    print("[Retrieval Agent] Searching for relevant chunks...")

    from query import query as run_query

    question = state["user_input"]
    result = run_query(question)

    state["retrieved_answer"] = result["answer"]
    state["retrieved_sources"] = result["sources"]

    print(f"[Retrieval Agent] Found answer with {len(result['sources'])} sources")
    return state


# ---- Agent 3: Insight Agent ----
# Extracts structured insights from a transcript file using Task 3 logic

def insight_agent(state: AgentState) -> AgentState:
    """Extract action items, decisions, etc. from the transcript."""
    print("[Insight Agent] Extracting insights from transcript...")

    from ingest import read_transcript
    from insights import extract_insights

    file_path = state["file_path"]
    text = read_transcript(file_path)
    insights = extract_insights(text)

    state["insights"] = insights
    print("[Insight Agent] Insights extracted")
    return state


# ---- Agent 4: Summary Agent ----
# Combines outputs from other agents into one clean response

SUMMARY_PROMPT = """You are a meeting assistant. Combine the following information into a clear, helpful response.

{sections}

Write a brief, well-organized summary for the user. Be concise."""


def summary_agent(state: AgentState) -> AgentState:
    """Combine all agent outputs into a final response."""
    print("[Summary Agent] Building final response...")

    final = {}
    sections = []

    # add RAG answer if available
    if state.get("retrieved_answer"):
        final["answer"] = state["retrieved_answer"]
        final["sources"] = state.get("retrieved_sources", [])
        sections.append(f"Question Answer:\n{state['retrieved_answer']}")

    # add insights if available
    if state.get("insights"):
        final["insights"] = state["insights"]
        sections.append("Insights: extracted successfully")

    # generate a short combined summary using LLM
    if len(sections) > 1:
        llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0.2, num_predict=512)
        combined = SUMMARY_PROMPT.format(sections="\n\n".join(sections))
        summary = llm.invoke(combined)
        final["summary"] = summary.strip()
    elif sections:
        final["summary"] = sections[0]

    final["route_used"] = state.get("route", "unknown")
    state["final_response"] = final

    print("[Summary Agent] Done!")
    return state


# ---- Build the LangGraph ----
# This is where we wire the agents together into a workflow.

def decide_next_after_router(state: AgentState) -> str:
    """After the router, decide which agent to call next."""
    route = state.get("route", "question")
    if route == "question":
        return "retrieval"
    elif route == "insights":
        return "insight"
    else:
        return "both_retrieval"


def build_agent_graph():
    """Build and compile the multi-agent LangGraph workflow.

    The graph looks like this:

    router ──→ retrieval ──→ summary ──→ END
           │
           ├─→ insight ──→ summary ──→ END
           │
           └─→ both_retrieval ──→ both_insight ──→ summary ──→ END
    """
    graph = StateGraph(AgentState)

    # add nodes (each agent is a node)
    graph.add_node("router", router_agent)
    graph.add_node("retrieval", retrieval_agent)
    graph.add_node("insight", insight_agent)
    graph.add_node("both_retrieval", retrieval_agent)
    graph.add_node("both_insight", insight_agent)
    graph.add_node("summary", summary_agent)

    # set the entry point
    graph.set_entry_point("router")

    # router decides where to go next
    graph.add_conditional_edges(
        "router",
        decide_next_after_router,
        {
            "retrieval": "retrieval",
            "insight": "insight",
            "both_retrieval": "both_retrieval",
        }
    )

    # question-only path: retrieval → summary → END
    graph.add_edge("retrieval", "summary")

    # insights-only path: insight → summary → END
    graph.add_edge("insight", "summary")

    # both path: retrieval → insight → summary → END
    graph.add_edge("both_retrieval", "both_insight")
    graph.add_edge("both_insight", "summary")

    # summary always ends the graph
    graph.add_edge("summary", END)

    # compile the graph into a runnable
    compiled = graph.compile()
    return compiled


# ---- Main function to run the agent graph ----

def run_agents(user_input, file_path=None):
    """Run the multi-agent workflow.

    Args:
        user_input: the user's question or request
        file_path: optional path to a transcript file (needed for insights)

    Returns:
        dict with the final response from the agents
    """
    graph = build_agent_graph()

    initial_state = {
        "user_input": user_input,
        "file_path": file_path or "",
    }

    print(f"\n{'=' * 50}")
    print(f"Running Agent Pipeline")
    print(f"Input: {user_input}")
    if file_path:
        print(f"File: {file_path}")
    print(f"{'=' * 50}\n")

    result = graph.invoke(initial_state)

    return result.get("final_response", {})


# ---- Run directly to test ----
if __name__ == "__main__":
    transcript = "./data/transcripts/sprint_planning.txt"

    print("\n\n--- TEST 1: Question only ---")
    result1 = run_agents("What was decided about rate limiting?")
    print(f"\nResult: {result1}")

    print("\n\n--- TEST 2: Insights only ---")
    result2 = run_agents("", file_path=transcript)
    print(f"\nResult keys: {list(result2.keys())}")

    print("\n\n--- TEST 3: Both question + insights ---")
    result3 = run_agents("Who has action items?", file_path=transcript)
    print(f"\nResult keys: {list(result3.keys())}")
