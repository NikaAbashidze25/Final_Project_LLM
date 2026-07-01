"""Shared utilities: logging, JSON extraction/validation, retry logic, answer matching.

These helpers enforce the project's non-negotiable rules:
  * Every LLM response is parsed and validated against a pydantic schema before
    being used by the next stage.
  * Malformed responses trigger up to MAX_RETRIES retries with an adjusted prompt.
  * All errors are logged with timestamp, model, stage, and description.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError, field_validator

from . import config
from .models.base_llm import BaseLLM

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
_LOGGER_NAME = "multi_llm_debate"


def setup_logger(run_tag: Optional[str] = None) -> logging.Logger:
    """Configure and return the run logger (writes to console + a logs/ file)."""
    config.ensure_directories()
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stamp = run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(config.LOGS_DIR / f"run_{stamp}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)


def log_error(logger: logging.Logger, model: str, stage: str, description: str) -> None:
    """Log a structured error line (timestamp is added by the formatter)."""
    logger.error("model=%s | stage=%s | %s", model, stage, description)


# ─────────────────────────────────────────────────────────────
# JSON extraction
# ─────────────────────────────────────────────────────────────
def extract_json(text: str) -> Optional[dict]:
    """Best-effort extraction of a JSON object from raw model text."""
    if not text:
        return None
    text = text.strip()
    # Strip markdown code fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: grab the substring from the first { to the last }.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


# ─────────────────────────────────────────────────────────────
# Pydantic schemas (one per stage output)
# ─────────────────────────────────────────────────────────────
class RoleAssessment(BaseModel):
    role_preferences: list[str]
    confidence_by_role: dict[str, float]
    reasoning: str = ""

    @field_validator("confidence_by_role")
    @classmethod
    def _has_both_roles(cls, v: dict[str, float]) -> dict[str, float]:
        for role in ("Solver", "Judge"):
            if role not in v:
                raise ValueError(f"confidence_by_role missing role {role!r}")
            if not 0.0 <= float(v[role]) <= 1.0:
                raise ValueError(f"confidence for {role} out of [0,1]")
        return v


class SolutionStep(BaseModel):
    step: int
    description: str
    reasoning: str = ""


class Solution(BaseModel):
    solver_id: str
    model: str
    solution_steps: list[SolutionStep] = Field(min_length=1)
    final_answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    assumptions: list[str] = []


class ReviewError(BaseModel):
    location: str = ""
    error_type: str
    description: str = ""
    severity: str

    @field_validator("error_type")
    @classmethod
    def _valid_error_type(cls, v: str) -> str:
        if v not in config.ERROR_TYPE_VALUES:
            raise ValueError(f"invalid error_type {v!r}")
        return v

    @field_validator("severity")
    @classmethod
    def _valid_severity(cls, v: str) -> str:
        if v not in config.SEVERITY_VALUES:
            raise ValueError(f"invalid severity {v!r}")
        return v


class ReviewEvaluation(BaseModel):
    strengths: list[str] = []
    weaknesses: list[str] = []
    errors: list[ReviewError] = []
    suggested_changes: list[str] = []


class Review(BaseModel):
    reviewer_id: str
    solution_reviewed: str
    evaluation: ReviewEvaluation
    overall_assessment: str

    @field_validator("overall_assessment")
    @classmethod
    def _valid_assessment(cls, v: str) -> str:
        if v not in config.OVERALL_ASSESSMENT_VALUES:
            raise ValueError(f"invalid overall_assessment {v!r}")
        return v


class CritiqueResponse(BaseModel):
    from_reviewer: str
    critique: str = ""
    response: str = ""
    accepted: bool


class Refinement(BaseModel):
    solver_id: str
    model: str
    critique_responses: list[CritiqueResponse] = []
    refined_solution_steps: list[SolutionStep] = Field(min_length=1)
    refined_final_answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    changes_summary: str = ""


class Judgment(BaseModel):
    judge_model: str
    winner: str
    confidence: float = Field(ge=0.0, le=1.0)
    ranking: dict[str, int]
    reasoning: str = ""
    correct_answer: str = ""
    notes: str = ""


T = TypeVar("T", bound=BaseModel)


# ─────────────────────────────────────────────────────────────
# Validated LLM call with retry
# ─────────────────────────────────────────────────────────────
def call_llm_validated(
    model: BaseLLM,
    *,
    system_prompt: str,
    user_prompt: str,
    schema: Type[T],
    stage: str,
    logger: logging.Logger,
    response_context: Optional[dict[str, Any]] = None,
    temperature: Optional[float] = None,
) -> Optional[T]:
    """Call the model, parse + validate JSON, retrying up to MAX_RETRIES times.

    Returns a validated pydantic object, or None if all attempts fail.
    """
    temp = config.TEMPERATURE if temperature is None else temperature
    # Inter-call delay only matters for real providers (rate limits); skip offline.
    delay = 0.0 if config.LLM_BACKEND == "offline" else config.API_DELAY_SECONDS
    attempt_prompt = user_prompt
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            raw = model.generate(
                system_prompt,
                attempt_prompt,
                temperature=temp,
                response_context=response_context,
            )
        except Exception as err:  # network / provider error
            log_error(logger, model.identity, stage,
                      f"generation error on attempt {attempt}: {err}")
            time.sleep(delay)
            continue

        data = extract_json(raw)
        if data is None:
            log_error(logger, model.identity, stage,
                      f"attempt {attempt}: response was not valid JSON")
        else:
            try:
                obj = schema.model_validate(data)
                if attempt > 1:
                    logger.info("model=%s | stage=%s | recovered on attempt %d",
                                model.identity, stage, attempt)
                # polite delay between successful provider calls (rate limiting)
                time.sleep(delay)
                return obj
            except ValidationError as err:
                log_error(logger, model.identity, stage,
                          f"attempt {attempt}: schema validation failed: "
                          f"{err.errors()[:2]}")

        # Adjust the prompt to nudge a clean response on the next try.
        attempt_prompt = (
            user_prompt
            + "\n\nIMPORTANT: Your previous response was invalid. Respond with ONLY a "
              "single valid JSON object that exactly matches the required schema. "
              "Do not include any text, explanation, or markdown fences outside the JSON."
        )
        time.sleep(delay)

    log_error(logger, model.identity, stage,
              f"FAILED to obtain a valid response after {config.MAX_RETRIES} attempts")
    return None


# ─────────────────────────────────────────────────────────────
# Answer normalization & matching
# ─────────────────────────────────────────────────────────────
def normalize_answer(answer: str) -> str:
    """Lowercase, strip punctuation/whitespace for robust comparison."""
    if answer is None:
        return ""
    s = str(answer).strip().lower()
    s = s.replace("≈", "").replace("approximately", "").replace("about", "")
    s = re.sub(r"[\s]+", " ", s)
    return s.strip().strip(".")


def _numbers(s: str) -> list[float]:
    out = []
    for tok in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", s.replace(",", "")):
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return out


def answers_match(candidate: str, truth: str) -> bool:
    """Robustly decide whether a candidate answer matches the ground truth.

    Handles: exact text, substring containment, numeric closeness (1% tol),
    and simple yes/no equivalence.
    """
    c, t = normalize_answer(candidate), normalize_answer(truth)
    if not c or not t:
        return False
    if c == t or t in c or c in t:
        return True

    # Numeric comparison: compare the most salient numbers.
    cn, tn = _numbers(c), _numbers(t)
    if tn:
        for tv in tn:
            for cv in cn:
                tol = max(0.01 * abs(tv), 0.05)
                if abs(cv - tv) <= tol:
                    return True

    # Yes/No equivalence.
    yes = {"yes", "true", "first player wins", "first player"}
    no = {"no", "false", "second player wins"}
    if (c in yes and t in yes) or (c in no and t in no):
        return True
    return False


# ─────────────────────────────────────────────────────────────
# Disk helpers
# ─────────────────────────────────────────────────────────────
def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_prompt(filename: str) -> str:
    return (config.PROMPTS_DIR / filename).read_text(encoding="utf-8")
