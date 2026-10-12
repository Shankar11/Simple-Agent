import json
import os
from main import app, AgentState
from models import generation_models

EVAL_FILE = "eval_dataset.json"

def evaluate_with_llm_judge(question: str, ground_truth: str, generated_answer: str) -> dict:
    """Uses local Qwen model to assess semantic correctness and factual consistency."""
    prompt = f"""# Task
You are an unbiased Cricket Evaluation Judge. Compare a generated answer against the ground truth answer.

# Inputs
Question: {question}
Ground Truth: {ground_truth}
Generated Answer: {generated_answer}

# Criteria
- Score 1: The generated answer is factually correct and includes all key statistics/names from ground truth.
- Score 0: The generated answer contradicts ground truth, hallucinates data, or misses key facts.

# Output format
Return strictly a JSON object:
```json
{{"score": 1, "explanation": "brief reason"}}