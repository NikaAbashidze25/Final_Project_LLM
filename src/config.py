"""Central configuration for the Multi-LLM Debate System.

All tunable constants, API keys, model names, and role mappings live here.
Values are read from environment variables (loaded from a `.env` file) with
sensible defaults so the project runs out-of-the-box in offline mode.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# ── Paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
PROBLEMS_PATH = DATA_DIR / "problems.json"
PROMPTS_DIR = PROJECT_ROOT / "src" / "prompts"
PLOTS_DIR = PROJECT_ROOT / "plots"
LOGS_DIR = PROJECT_ROOT / "logs"

STAGE_DIRS = {
    "stage0": RESULTS_DIR / "stage0_role_assessments",
    "stage1": RESULTS_DIR / "stage1_solutions",
    "stage2": RESULTS_DIR / "stage2_reviews",
    "stage3": RESULTS_DIR / "stage3_refinements",
    "stage4": RESULTS_DIR / "stage4_judgments",
}

# ── Backend selection ────────────────────────────────────────
# "openai" -> real API calls; "offline" -> deterministic local simulation.
LLM_BACKEND = os.getenv("LLM_BACKEND", "offline").strip().lower()

# ── Free / single-model mode ─────────────────────────────────
FREE_MODEL_MODE = _get_bool("FREE_MODEL_MODE", True)
FREE_MODEL_PROVIDER = os.getenv("FREE_MODEL_PROVIDER", "openai").strip().lower()

# ── API keys ─────────────────────────────────────────────────
API_KEYS = {
    "gpt-4": os.getenv("OPENAI_API_KEY", ""),
    "openai": os.getenv("OPENAI_API_KEY", ""),
    "claude": os.getenv("ANTHROPIC_API_KEY", ""),
    "gemini": os.getenv("GOOGLE_API_KEY", ""),
    "grok": os.getenv("GROK_API_KEY", ""),
}

# ── Model names per provider ─────────────────────────────────
MODEL_NAMES = {
    "gpt-4": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "claude": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
    "gemini": os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
    "grok": os.getenv("GROK_MODEL", "grok-2-latest"),
}

# In split-role mode, solvers and judge can use different models.
SOLVER_MODEL = os.getenv("SOLVER_MODEL", MODEL_NAMES["openai"])
JUDGE_MODEL = os.getenv("JUDGE_MODEL", MODEL_NAMES["openai"])

# The four canonical model identities used throughout the system.
ALL_MODELS = ["gpt-4", "claude", "gemini", "grok"]

# Deterministic tie-break priority for role assignment (Stage 0.5).
MODEL_PRIORITY = ["gpt-4", "claude", "gemini", "grok"]

# Default role->model map (overridden per-problem by Stage 0.5 at runtime).
ROLE_MODEL_MAP = {
    "Solver_1": "claude",
    "Solver_2": "gpt-4",
    "Solver_3": "gemini",
    "Judge": "grok",
}

# ── Runtime tuning ───────────────────────────────────────────
API_DELAY_SECONDS = _get_float("API_DELAY_SECONDS", 1.0)
MAX_RETRIES = _get_int("MAX_RETRIES", 3)
TEMPERATURE = _get_float("TEMPERATURE", 0.7)
RANDOM_SEED = _get_int("RANDOM_SEED", 42)


def ensure_directories() -> None:
    """Create all output directories if they do not already exist."""
    for path in [DATA_DIR, RESULTS_DIR, PLOTS_DIR, LOGS_DIR, *STAGE_DIRS.values()]:
        path.mkdir(parents=True, exist_ok=True)
