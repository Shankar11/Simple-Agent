import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv

from main import app, AgentState
from models import generation_models
from utils import parse_response

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH if ENV_PATH.exists() else None, override=True)

EVAL_FILE = BASE_DIR / "eval_dataset.json"


def evaluate_with_llm_judge(question: str, ground_truth: str, generated_answer: str) -> dict:
    # Using a list join prevents triple-quote and markdown copy-paste syntax errors
    prompt_lines = [
        "# Task",
        "You are an unbiased Cricket Evaluation Judge. Compare a generated answer against the ground truth answer.",
        "",
        "# Inputs",
        f"Question: {question}",
        f"Ground Truth: {ground_truth}",
        f"Generated Answer: {generated_answer}",
        "",
        "# Criteria",
        "- Score 1: The generated answer is factually correct and includes all key statistics/names from ground truth.",
        "- Score 0: The generated answer contradicts ground truth, hallucinates data, or misses key facts.",
        "",
        "# Output format",
        'Return strictly a valid JSON object: {"score": 1, "explanation": "brief reason"}',
        "# Response:"
    ]
    prompt = "\n".join(prompt_lines)

    resp = generation_models(prompt, enable_thinking=False, max_tokens=120)
    try:
        parsed = parse_response(resp)
        if isinstance(parsed, dict):
            return {
                "score": int(parsed.get("score", 0)),
                "explanation": str(parsed.get("explanation", ""))
            }
        if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
            return {
                "score": int(parsed[0].get("score", 0)),
                "explanation": str(parsed[0].get("explanation", ""))
            }
    except Exception:
        pass

    gt_tokens = [w.lower() for w in re.findall(r"\b\w{3,}\b", ground_truth)]
    matched = sum(1 for w in gt_tokens if w in generated_answer.lower())
    fallback_score = 1 if (matched / max(1, len(gt_tokens))) >= 0.5 else 0
    return {
        "score": fallback_score,
        "explanation": "Evaluated via token overlap fallback."
    }


def run_benchmarks():
    if not EVAL_FILE.exists():
        print(f"[Error] Benchmark file not found at: {EVAL_FILE}")
        return

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)

    total = len(cases)
    passed_routing = 0
    passed_entities = 0
    llm_judge_score = 0

    print("\n" + "=" * 75)
    print(f"Running IPL 2026 Golden Evals ({total} test cases)")
    print("=" * 75 + "\n")

    for i, test in enumerate(cases, 1):
        q = test["question"]
        print(f"[{i:02d}/{total:02d}] Evaluating: {q}")

        final_state = app.invoke({"question": q})

        actual_router = final_state.get("router", [])
        final_answer = final_state.get("final_answer", "").strip()

        expected_routers = test.get("target_router", [])
        router_hit = any(r in actual_router for r in expected_routers) if expected_routers else True
        if router_hit:
            passed_routing += 1

        key_entities = test.get("key_entities", [])
        entities_present = sum(1 for e in key_entities if e.lower() in final_answer.lower())
        entity_ratio = entities_present / len(key_entities) if key_entities else 1.0
        entity_hit = entity_ratio >= 0.6
        if entity_hit:
            passed_entities += 1

        eval_result = evaluate_with_llm_judge(q, test["expected_answer"], final_answer)
        score = eval_result.get("score", 0)
        llm_judge_score += score

        print(f" -> Answer       : {final_answer[:120]}...")
        print(f" -> Router Match : {'PASS' if router_hit else 'FAIL'} (Expected: {expected_routers}, Got: {actual_router})")
        print(f" -> Entity Recall: {'PASS' if entity_hit else 'FAIL'} ({entities_present}/{len(key_entities)} matched)")
        print(f" -> LLM Judge    : {'PASS' if score == 1 else 'FAIL'} - {eval_result.get('explanation', '')}")
        print("-" * 75)

    print("\n" + "=" * 60)
    print("Evaluation Results Summary")
    print("=" * 60)
    print(f"Routing Accuracy       : {(passed_routing / total) * 100:.1f}% ({passed_routing}/{total})")
    print(f"Key Entity Recall Rate : {(passed_entities / total) * 100:.1f}% ({passed_entities}/{total})")
    print(f"LLM Judge Factual Score: {(llm_judge_score / total) * 100:.1f}% ({llm_judge_score}/{total})")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_benchmarks()