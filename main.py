import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from pydantic import BaseModel
from langgraph.graph import StateGraph, END, START
from dotenv import load_dotenv

from utils import load_all_ipl_documents, build_retrievers, parse_response
from models import generation_models, rerank_local, web_search
from prompt_set import (
    router_prompt,
    generation_prompt,
    extract_keyinfo_prompt,
    validate_contents_prompt,
    supplement_query_prompt
)

load_dotenv(override=True)

DATA_DIR = "./data"

print("Scanning and indexing IPL dataset CSVs...")
docs_grouped = load_all_ipl_documents(DATA_DIR)
retriever_dict = {}

for group_name, doc_list in docs_grouped.items():
    if doc_list:
        v_retriever, bm_retriever = build_retrievers(doc_list)
        retriever_dict[f"{group_name}_vector"] = v_retriever
        retriever_dict[f"{group_name}_bm25"] = bm_retriever
        print(f"Indexed [{group_name}]: {len(doc_list)} records")
    else:
        print(f"[Notice] No documents found for [{group_name}]. Ensure CSVs exist in {DATA_DIR}.")

class AgentState(BaseModel):
    question: str
    router: List[str] = []
    contexts: Dict[str, List[str]] = {}
    final_answer: str = ""
    rerank_context_list: List[str] = []
    retry_count: int = 0
    is_content_valid: bool = False

def route_database(state: AgentState):
    prompt = router_prompt(state.question)
    response = generation_models(prompt, enable_thinking=False)
    router_res = parse_response(response)
    
    valid_keys = [k for k in docs_grouped.keys() if f"{k}_vector" in retriever_dict]
    chosen = [r for r in router_res if r in valid_keys] if isinstance(router_res, list) else []
    
    return {"router": chosen if chosen else valid_keys}

executor = ThreadPoolExecutor(max_workers=4)

def _retrieve_one(db_name: str, question: str):
    potential_res = []
    try:
        vec_name = f"{db_name}_vector"
        if vec_name in retriever_dict and retriever_dict[vec_name]:
            docs = retriever_dict[vec_name].invoke(question)
            for d in docs:
                potential_res.append(d.page_content)

        bm_name = f"{db_name}_bm25"
        if bm_name in retriever_dict and retriever_dict[bm_name]:
            docs = retriever_dict[bm_name](question)
            for item in docs:
                potential_res.append(item["page_content"])

        return db_name, list(set(potential_res))
    except Exception as e:
        return db_name, [f"[retrieval fail: {str(e)}]"]

def retrieve_parallel(state: AgentState):
    futures = [
        executor.submit(_retrieve_one, db_name, state.question)
        for db_name in state.router
    ]
    results = [future.result() for future in futures]
    return {"contexts": {db: content for db, content in results}}

def rerank_context(state: AgentState):
    parts = []
    for content in state.contexts.values():
        parts.extend(content)
    
    ranked_docs = rerank_local(state.question, parts, top_n=6)
    return {"rerank_context_list": ranked_docs}

def filter_content(state: AgentState):
    if not state.rerank_context_list:
        return {"rerank_context_list": []}
        
    contents = "".join([f"## Content {i}:\n{ct}\n" for i, ct in enumerate(state.rerank_context_list)])
    prompt = extract_keyinfo_prompt(state.question, contents)
    response = generation_models(prompt, enable_thinking=False)
    relative_idx = parse_response(response)

    chosen = []
    for idx in relative_idx:
        if isinstance(idx, int) and 0 <= idx < len(state.rerank_context_list):
            chosen.append(state.rerank_context_list[idx])
            
    updated_list = chosen if chosen else state.rerank_context_list
    return {"rerank_context_list": updated_list}

def valid_and_refine_node(state: AgentState):
    if not state.rerank_context_list:
        return {"is_content_valid": False}
        
    contents = "".join([f"## Content {i}:\n{ct}\n" for i, ct in enumerate(state.rerank_context_list)])
    prompt = validate_contents_prompt(state.question, contents)
    response = generation_models(prompt, enable_thinking=False)
    res = parse_response(response)
    
    is_valid = bool(res and str(res[0]).strip().lower() == "yes")
    return {"is_content_valid": is_valid}

def search_node(state: AgentState):
    contents = "".join([f"## Content {i}:\n{ct}\n" for i, ct in enumerate(state.rerank_context_list)])
    prompt = supplement_query_prompt(state.question, contents)
    response = generation_models(prompt, enable_thinking=False)
    supple_query = parse_response(response)

    query = supple_query[0] if (supple_query and isinstance(supple_query, list)) else state.question
    search_res = web_search(query)
    
    new_contents = [f"[Web Result]: {item['content']}" for item in search_res.get("results", []) if "content" in item]
    combined = new_contents + list(state.rerank_context_list)
    
    return {
        "retry_count": state.retry_count + 1,
        "rerank_context_list": combined[:8]
    }

def generate_answer(state: AgentState):
    contents = "".join([f"## Content {i}:\n{ct}\n" for i, ct in enumerate(state.rerank_context_list)])
    prompt = generation_prompt(state.question, contents)
    answer = generation_models(prompt, enable_thinking=True)
    return {"final_answer": answer}

def should_generate(state: AgentState):
    if state.retry_count >= 2 or state.is_content_valid:
        return "generate"
    return "retry"

workflow = StateGraph(AgentState)

workflow.add_node("route_databases", route_database)
workflow.add_node("retrieve_parallel", retrieve_parallel)
workflow.add_node("rerank_context", rerank_context)
workflow.add_node("filter_content", filter_content)
workflow.add_node("valid_and_refine", valid_and_refine_node)
workflow.add_node("search_node", search_node)
workflow.add_node("generate_answer", generate_answer)

workflow.add_edge(START, "route_databases")
workflow.add_edge("route_databases", "retrieve_parallel")
workflow.add_edge("retrieve_parallel", "rerank_context")
workflow.add_edge("rerank_context", "filter_content")
workflow.add_edge("filter_content", "valid_and_refine")

workflow.add_conditional_edges(
    "valid_and_refine",
    should_generate,
    {
        "generate": "generate_answer",
        "retry": "search_node"
    }
)

workflow.add_edge("search_node", "valid_and_refine")
workflow.add_edge("generate_answer", END)

app = workflow.compile()

if __name__ == "__main__":
    sample_queries = [
        "Which team won the final match in IPL 2026 and at which venue?",
        "Who scored the most runs or hit the most sixes in the tournament?",
        "What is the capacity of Narendra Modi Stadium, and which team tops the points table?",
        "Who has the highest strike rate among batters?"
    ]
    
    for q in sample_queries:
        print("\n" + "=" * 60)
        print(f"Query: {q}")
        print("=" * 60)
        inputs = AgentState(question=q)
        final_answer = ""
        
        for event in app.stream(inputs):
            for node_name, node_val in event.items():
                print(f"[Node Complete]: {node_name}")
                if node_name == "generate_answer":
                    if isinstance(node_val, dict):
                        final_answer = node_val.get("final_answer", "")
                    else:
                        final_answer = getattr(node_val, "final_answer", "")

        print(f"\nFinal Answer:\n{final_answer}")