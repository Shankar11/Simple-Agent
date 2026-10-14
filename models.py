import os
import re
import json
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv(override=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
DEFAULT_LLM = os.getenv("OLLAMA_LLM", "qwen3.5:4b")
DEFAULT_EMBED = os.getenv("OLLAMA_EMBED", "mxbai-embed-large:latest")
RERANK_MODEL = os.getenv("OLLAMA_RERANK", "dengcao/Qwen3-Reranker-0.6B:F16")
RERANK_BACKEND = os.getenv("RERANK_BACKEND", "llm")  # "llm" (Qwen3-Reranker) or "embed" (cosine)

# Score below which retrieval is considered "weak" -> web search fallback.
# llm backend = probability of "yes" (0..1); embed backend = cosine similarity.
RERANK_MIN = float(os.getenv("RERANK_MIN", "0.3" if RERANK_BACKEND == "llm" else "0.5"))

_client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
def generation_models(
    prompt: str,
    model: str = DEFAULT_LLM,
    enable_thinking: bool = False,
    max_tokens: int = 300,
    stream: bool = False,
) -> str:
    """Single chat call. With stream=True tokens are printed as they arrive."""
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system",
             "content": "You are a concise, factual cricket analyst. Answer clearly and directly."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1 if enable_thinking else 0.0,
        "max_tokens": max(max_tokens, 1200) if enable_thinking else max_tokens,
        # Ollama's OpenAI endpoint honors reasoning_effort; "think" is for the native API.
        # Send both when thinking is off (this is what the original code did).
        "extra_body": {"think": True} if enable_thinking else {"think": False, "reasoning_effort": "none"},
        "stream": stream,
    }

    try:
        completion = _client.chat.completions.create(**kwargs)
    except Exception as e:
        print(f"    [WARN] Generation error: {e}. Retrying without extra params...")
        kwargs.pop("extra_body", None)
        completion = _client.chat.completions.create(**kwargs)

    if stream:
        out = []
        for chunk in completion:
            tok = chunk.choices[0].delta.content or ""
            print(tok, end="", flush=True)
            out.append(tok)
        print()
        content = "".join(out)
    else:
        content = completion.choices[0].message.content or ""

    # Strip residual <think> blocks if the model leaks them
    if "</think>" in content:
        content = content.split("</think>")[-1]
    content = content.replace("<think>", "").strip()

    if not content:
        # Do NOT fall back to reasoning_content: that is the model's thinking, not an answer.
        raise RuntimeError("Model returned empty content. Check the model name / think flag.")
    return content


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------
def embedding_models(doc: str | list[str], model: str = DEFAULT_EMBED) -> list:
    input_text = [doc] if isinstance(doc, str) else doc
    if not input_text:
        return []
    completion = _client.embeddings.create(model=model, input=input_text, encoding_format="float")
    data = json.loads(completion.model_dump_json())["data"]
    return data[0]["embedding"] if isinstance(doc, str) else [d["embedding"] for d in data]


class QwenEmbeddings(Embeddings):
    def __init__(self, model_name: str = DEFAULT_EMBED):
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = _client.embeddings.create(model=self.model_name, input=texts, encoding_format="float")
        result = json.loads(response.model_dump_json())
        return [d["embedding"] for d in result["data"]]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


# --------------------------------------------------------------------------
# Reranking
# --------------------------------------------------------------------------
_RERANK_SYSTEM = (
    'Judge whether the Document meets the requirements based on the Query and the Instruct '
    'provided. Note that the answer can only be "yes" or "no".'
)
_RERANK_INSTRUCT = "Given an IPL cricket question, judge whether the cricket record helps answer it"


def _score_pair(query: str, passage: str) -> float:
    """Return P(yes) that `passage` is relevant to `query` using a Qwen3-Reranker model."""
    resp = _client.chat.completions.create(
        model=RERANK_MODEL,
        messages=[
            {"role": "system", "content": _RERANK_SYSTEM},
            {"role": "user",
             "content": f"<Instruct>: {_RERANK_INSTRUCT}\n<Query>: {query}\n<Document>: {passage}"},
        ],
        temperature=0,
        max_tokens=8,
        logprobs=True,
        top_logprobs=10,
    )
    choice = resp.choices[0]
    text = choice.message.content or ""
    if os.getenv("RERANK_DEBUG"):
        print(f"      [RERANK_DEBUG] output={text!r} has_logprobs={bool(choice.logprobs)}")

    # Soft score: first generated position where "yes"/"no" carry real probability mass.
    # This skips any leading <think></think> tokens the model template may emit.
    positions = choice.logprobs.content if (choice.logprobs and choice.logprobs.content) else []
    for pos in positions:
        yes = sum(math.exp(t.logprob) for t in pos.top_logprobs if t.token.strip().lower() == "yes")
        no = sum(math.exp(t.logprob) for t in pos.top_logprobs if t.token.strip().lower() == "no")
        if yes + no > 0.5:
            return yes / (yes + no)

    # Fallback: hard yes/no from text, ignoring any think block
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip().lower()
    return 1.0 if text.startswith("yes") else 0.0


_rerank_pool = ThreadPoolExecutor(max_workers=4)  # pair with OLLAMA_NUM_PARALLEL=4


def rerank_local(query: str, passages: list[str], top_n: int = 6) -> list[tuple[str, float]]:
    """Return [(passage, score)] sorted best-first. Same output for both backends."""
    if not passages:
        return []

    if RERANK_BACKEND == "llm":
        scores = list(_rerank_pool.map(lambda p: _score_pair(query, p), passages))
    else:
        q = np.array(embedding_models(query))
        d = np.array(embedding_models(passages))
        scores = (d @ q / (np.linalg.norm(d, axis=1) * np.linalg.norm(q) + 1e-10)).tolist()

    # Sort by score desc; ties keep the original (RRF) order. argsort()[::-1] would reverse ties.
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:top_n]
    return [(passages[i], float(scores[i])) for i in order]


# --------------------------------------------------------------------------
# Web search fallback
# --------------------------------------------------------------------------
def web_search(query: str) -> dict:
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            from tavily import TavilyClient
            result = TavilyClient(api_key=tavily_key).search(query, max_results=4)
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
            raw = list(ddgs.text(query, max_results=4))
        return {"results": [{"content": f"{i.get('title', '')}: {i.get('body', '')}"}
                            for i in raw if i.get("body")]}
    except Exception as e:
        return {"results": [{"content": f"[Web search unavailable: {e}]"}]}