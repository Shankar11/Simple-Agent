"""
evaluate.py
Benchmark evaluation runner for the IPL 2026 RAG pipeline.
Evaluates Routing Accuracy, Key Entity Recall, and Factual Accuracy (LLM-as-a-Judge).
"""

import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv

# Import compiled LangGraph workflow and state
from main import app, AgentState
from models import generation_models
from utils import parse_response

# Load environment configuration
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH if ENV_PATH.exists() else None, override=True)

EVAL_FILE = BASE_DIR / "eval_dataset.json"


def evaluate_with_llm_judge(question: str, ground_truth: str, generated_answer: str) -> dict:
    """Uses local Qwen model to assess semantic correctness and factual consistency."""
    # Note: Double curly braces {{ }} prevent Python f-string SyntaxError
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
{{
  "score": 1,
  "explanation": "brief reason"
}}