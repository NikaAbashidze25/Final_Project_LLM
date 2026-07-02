"""Stage 4: Final Judgment.

The Judge LLM receives the full debate record (original solutions, all peer
reviews, all refined solutions) and selects the winning solution. If the Judge
fails, the system falls back to the refined solution with the highest confidence.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from .. import utils
from ..models import get_judge_model


def run_stage4(
    problem: dict[str, Any],
    assignment: dict[str, str],
    solutions: dict[str, dict],
    reviews: list[dict],
    refinements: dict[str, dict],
    logger: logging.Logger,
) -> dict[str, Any]:
    """Return a validated judgment dict (with a fallback flag if needed)."""
    system_prompt = utils.read_prompt("judge_system_prompt.txt")
    judge_model_id = assignment["Judge"]
    model = get_judge_model(judge_model_id)

    def strip(d: dict) -> dict:
        return {k: v for k, v in d.items() if not k.startswith("_")}

    user_prompt = (
        f"PROBLEM:\n{problem['problem']}\n\n"
        f"ORIGINAL SOLUTIONS:\n{json.dumps([strip(s) for s in solutions.values()], indent=2)}\n\n"
        f"PEER REVIEWS:\n{json.dumps([strip(r) for r in reviews], indent=2)}\n\n"
        f"REFINED SOLUTIONS:\n{json.dumps([strip(r) for r in refinements.values()], indent=2)}\n\n"
        f"You are the Judge (model: {judge_model_id}). Evaluate everything and return "
        f"the required JSON. Rank every solver present and pick a winner."
    )

    candidates = [
        {
            "solver_id": sid,
            "answer": ref["refined_final_answer"],
            "correct": bool(ref.get("_correct", False)),
            "confidence": ref.get("confidence", 0.0),
        }
        for sid, ref in refinements.items()
    ]
    ctx = {
        "task": "judge",
        "problem_id": problem["id"],
        "candidates": candidates,
    }
    result = utils.call_llm_validated(
        model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=utils.Judgment,
        stage="stage4_judge",
        logger=logger,
        response_context=ctx,
    )

    if result is None or result.winner not in refinements:
        # Fallback: highest-confidence refined solution.
        logger.warning("stage4 | judge failed/invalid; falling back to highest-confidence solver")
        winner = max(refinements.values(), key=lambda r: r.get("confidence", 0))["solver_id"]
        judgment = {
            "judge_model": judge_model_id,
            "winner": winner,
            "confidence": refinements[winner].get("confidence", 0.0),
            "reasoning": "Fallback: Judge unavailable; selected the highest-confidence "
                         "refined solution.",
            "correct_answer": refinements[winner]["refined_final_answer"],
            "_fallback": True,
        }
    else:
        judgment = result.model_dump()
        judgment["judge_model"] = judge_model_id
        judgment["_fallback"] = False

    logger.info("stage4 | judge=%s winner=%s conf=%.2f",
                judge_model_id, judgment["winner"], judgment["confidence"])
    return judgment
