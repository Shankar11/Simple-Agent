import time
from typing import List
from pydantic import BaseModel
from langgraph.graph import StateGraph, END, START
from dotenv import load_dotenv

from utils import load_all_ipl_documents, GroupIndex, embeddings, parse_response
from models import generation_models, rerank_local, web_search, RERANK_MIN, RERANK_BACKEND, RERANK_MODEL
from prompt_set import router_prompt, generation_prompt

load_dotenv(override=True)

DATA_DIR = "./data"

# ---------- build once at import ----------
indexes: dict[str, GroupIndex] = {
    name: GroupIndex(name, docs)
    for name, docs in load_all_ipl_documents(DATA_DIR).items()
    if docs
}


# ---------- state ----------
class AgentState(BaseModel):
    question: str
    router: List[str] = []
    candidates: List[str] = []
    rerank_context_list: List[str] = []
    top_score: float = 0.0
    used_web: bool = False
    final_answer: str = ""


def _format(ctx: list[str]) -> str:
    return "".join(f"## Content {i}:\n{c}\n" for i, c in enumerate(ctx))


# ---------- nodes ----------
def route_database(state: AgentState):
    t0 = time.perf_counter()
    try:
        raw = generation_models(router_prompt(state.question), enable_thinking=False, max_tokens=60)
    except RuntimeError as e:  # empty reply -> don't crash, just search every group
        print(f"  [ROUTE] LLM returned nothing ({e}); using all groups")
        raw = ""
    parsed = parse_response(raw)
    if not isinstance(parsed, list):
        parsed = []
    chosen = [r for r in parsed if r in indexes]
    if not chosen:  # JSON failed -> look for group names mentioned in the raw text, else search all
        chosen = [k for k in indexes if k in raw] or list(indexes)
    print(f"  [ROUTE] {chosen} ({time.perf_counter() - t0:.2f}s)")
    return {"router": chosen}


def retrieve(state: AgentState):
    t0 = time.perf_counter()
    qvec = embeddings.embed_query(state.question)  # ONE embedding call, reused for every group
    seen, candidates = set(), []
    for name in state.router:
        for text in indexes[name].search(state.question, qvec, k=8):
            if text not in seen:  # order-preserving de-dup (replaces list(set(...)))
                seen.add(text)
                candidates.append(text)
    print(f"  [RETRIEVE] {len(candidates)} candidates ({time.perf_counter() - t0:.2f}s)")
    return {"candidates": candidates}


def rerank_context(state: AgentState):
    t0 = time.perf_counter()
    ranked = rerank_local(state.question, state.candidates, top_n=6)
    top = ranked[0][1] if ranked else 0.0
    print(f"  [RERANK:{RERANK_BACKEND}] top score {top:.2f} ({time.perf_counter() - t0:.2f}s)")
    return {"rerank_context_list": [p for p, _ in ranked], "top_score": top}


def search_node(state: AgentState):
    t0 = time.perf_counter()
    res = web_search(f"IPL 2026 {state.question}")  # raw question, no extra LLM call
    snippets = [f"[Web Result]: {r['content']}" for r in res.get("results", []) if "content" in r]
    print(f"  [WEB SEARCH] +{len(snippets)} snippets ({time.perf_counter() - t0:.2f}s)")
    return {"rerank_context_list": (snippets + state.rerank_context_list)[:8], "used_web": True}


def generate_answer(state: AgentState):
    t0 = time.perf_counter()
    print("  [GENERATE] ", end="", flush=True)
    print()
    answer = generation_models(
        generation_prompt(state.question, _format(state.rerank_context_list)),
        enable_thinking=False,
        max_tokens=250,
        stream=True,  # tokens print as they arrive
    )
    print(f"  [GENERATE] done ({time.perf_counter() - t0:.2f}s)")
    return {"final_answer": answer}


def needs_web(state: AgentState):
    # Cheap, LLM-free check: is even the best passage clearly relevant?
    return "web" if state.top_score < RERANK_MIN else "generate"


# ---------- graph ----------
workflow = StateGraph(AgentState)
workflow.add_node("route_databases", route_database)
workflow.add_node("retrieve", retrieve)
workflow.add_node("rerank_context", rerank_context)
workflow.add_node("search_node", search_node)
workflow.add_node("generate_answer", generate_answer)

workflow.add_edge(START, "route_databases")
workflow.add_edge("route_databases", "retrieve")
workflow.add_edge("retrieve", "rerank_context")
workflow.add_conditional_edges("rerank_context", needs_web, {"web": "search_node", "generate": "generate_answer"})
workflow.add_edge("search_node", "generate_answer")
workflow.add_edge("generate_answer", END)

app = workflow.compile()


if __name__ == "__main__":
    print(f"Reranker: {RERANK_BACKEND} ({RERANK_MODEL if RERANK_BACKEND == 'llm' else 'embeddings'}), "
          f"web fallback below {RERANK_MIN}")
    queries = [
        "Which team won the final match in IPL 2026 and at which venue?",
        "Who scored the most runs or hit the most sixes in the tournament?",
        "What is the capacity of Narendra Modi Stadium, and which team tops the points table?",
        "Who has the highest strike rate among batters?",
    ]
    for q in queries:
        print("\n" + "=" * 65 + f"\nQUERY: {q}\n" + "=" * 65)
        t0 = time.perf_counter()
        state = app.invoke({"question": q})
        print(f"[Total {time.perf_counter() - t0:.2f}s | web used: {state.get('used_web', False)} | "
              f"top score: {state.get('top_score', 0.0):.2f}]")