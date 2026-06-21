"""
Central configuration: project paths, .env loading, API keys, default models.
Single entry point so no module hard-codes absolute paths or inline keys.
"""
from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):  # type: ignore
        return False


ROOT = Path(__file__).resolve().parent

_env = ROOT / ".env"
if _env.exists():
    load_dotenv(_env, override=False)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = ROOT / "data"
RAG_DATASETS_DIR = DATA_DIR / "rag_datasets"

PERSONA_PROMPT_PATH = DATA_DIR / "chim_persona_prompt.json"
CLAIMS_PATH = DATA_DIR / "chim_claims.json"
TESTSET_SOURCE = DATA_DIR / "chim_qa_1.json"          # held-out test source
ABSURD_QA_PATH = DATA_DIR / "chim_absurd_qa_1000.json"
RAG_CACHE_PATH = DATA_DIR / "rag_cache.pkl"           # built by rag/build_index.py

RESULTS_DIR = ROOT / "results"
TESTSET_PATH = RESULTS_DIR / "testset_100.jsonl"
SFT_DATA_PATH = ROOT / "finetune" / "sft_data.jsonl"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "finetune").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# API keys / providers (filled from .env)
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_RAG_MODEL = os.getenv("RAG_MODEL", "google/gemini-2.5-flash")
DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-4o-mini")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "jhgan/ko-sroberta-multitask")


@lru_cache(maxsize=4)
def get_openai_client(base_url: str | None = OPENROUTER_BASE_URL):
    """OpenAI-compatible client (defaults to OpenRouter)."""
    from openai import OpenAI

    if base_url == OPENROUTER_BASE_URL:
        key = OPENROUTER_API_KEY
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is missing in .env")
    else:
        key = OPENAI_API_KEY
        if not key:
            raise RuntimeError("OPENAI_API_KEY is missing in .env")
    return OpenAI(base_url=base_url, api_key=key)
