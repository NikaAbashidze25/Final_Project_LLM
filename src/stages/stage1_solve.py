"""Stage 1: Independent Solution Generation.

Each of the three Solvers independently solves the problem with no communication.
A Solver that fails after retries is excluded; the remaining Solvers proceed.
"""
from __future__ import annotations

import logging
from typing import Any

from .. import utils
from ..models import get_model


def run_stage1(
    problem: dict[str, Any],
    assignment: dict[str, str],
    logger: logging.Logger,
) -> dict[str, dict]:
    """Return {solver_id -> validated solution dict} for solvers that succeeded."""
    system_prompt = utils.read_prompt("solver_system_prompt.txt")
    solutions: dict[str, dict] = {}

    for seat in ("Solver_1", "Solver_2", "Solver_3"):
        solver_id = seat.lower()  # "solver_1"
        model_id = assignment[seat]
        model = get_model(model_id)
        user_prompt = (
            f"PROBLEM:\n{problem['problem']}\n\n"
            f"You are {solver_id} (model: {model_id}). Solve this INDEPENDENTLY and "
            f"return the required JSON. Your solver_id is '{solver_id}' and your model "
            f"is '{model_id}'."
        )
        ctx = {
            "task": "solve",
            "problem_id": problem["id"],
            "solver_id": solver_id,
            "difficulty": problem.get("difficulty", "hard"),
            "ground_truth": problem.get("correct_answer", ""),
        }
        result = utils.call_llm_validated(
            model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=utils.Solution,
            stage="stage1_solve",
            logger=logger,
            response_context=ctx,
        )
        if result is None:
            logger.warning("stage1 | %s (%s) failed; excluded from peer review",
                           solver_id, model_id)
            continue
        sol = result.model_dump()
        # Force identity fields to the assigned values for consistency.
        sol["solver_id"] = solver_id
        sol["model"] = model_id
        # Record correctness for downstream simulation / analysis (not sent to models).
        sol["_correct"] = utils.answers_match(sol["final_answer"], problem.get("correct_answer", ""))
        solutions[solver_id] = sol
        logger.info("stage1 | %s (%s) -> answer=%r conf=%.2f",
                    solver_id, model_id, sol["final_answer"], sol["confidence"])

    return solutions
