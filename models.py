import os
import re
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv(override=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
DEFAULT_LLM = os.getenv("OLLAMA_LLM", "qwen3.5:4b")
DEFAULT_EMBED = os.getenv("OLLAMA_EMBED", "mxbai-embed-large:latest")

def get_ollama_client() -> OpenAI:
    return OpenAI(
        api_key="ollama",
        base_url=OLLAMA_BASE_URL
    )

def generation_models(prompt: str, model: str = DEFAULT_LLM, enable_thinking: bool = False) -> str:
    """Local inference. Thinking enabled for reasoned answer generation, disabled for routing/JSON."""
    with get_ollama_client() as client:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an IPL cricket assistant. Follow formatting rules directly."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3 if enable_thinking else 0.0,
                extra_body={"enable_thinking": enable_thinking}
            )
            content = completion.choices[0].message.content or ""
        except Exception:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a cricket assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            content = completion.choices[0].message.content or ""

    cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return cleaned

def embedding_models(doc: str | list[str], model: str = DEFAULT_EMBED):
    input_text = [doc] if isinstance(doc, str) else doc
    if not input_text:
        return []
    with get_ollama_client() as client:
        completion = client.embeddings.create(
            model=model,
            input=input_text,
            encoding_format="float"
        )
    data = json.loads(completion.model_dump_json())['data']
    return data[0]['embedding'] if isinstance(doc, str) else [d['embedding'] for d in data]

class QwenEmbeddings(Embeddings):
    def __init__(self, model_name: str = DEFAULT_EMBED):
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with get_ollama_client() as client:
            response = client.embeddings.create(
                model=self.model_name,
                input=texts,
                encoding_format="float"
            )
        result = json.loads(response.model_dump_json())
        return [d['embedding'] for d in result['data']]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

def rerank_local(query: str, passages: list[str], top_n: int = 6) -> list[str]:
    if not passages:
        return []
    
    query_vec = np.array(embedding_models(query))
    doc_vecs = np.array(embedding_models(passages))
    
    scores = np.dot(doc_vecs, query_vec) / (
        np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec) + 1e-10
    )
    ranked_indices = np.argsort(scores)[::-1][:top_n]
    return [passages[idx] for idx in ranked_indices]

def web_search(query: str) -> dict:
    """Tavily search with fallback to ddgs."""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            from tavily import TavilyClient
            tavily_client = TavilyClient(api_key=tavily_key)
            result = tavily_client.search(query, max_results=5)
            if result.get("results"):
                return result
        except Exception:
            pass

    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=5))

        formatted_results = [
            {"content": f"{item.get('title', '')}: {item.get('body', '')}"}
            for item in raw_results if item.get('body')
        ]
        return {"results": formatted_results}
    except Exception as e:
        return {"results": [{"content": f"[Web search unavailable: {str(e)}]"}]}