"""Stage 1: Independent Solution Generation.

Each of the three assigned Solvers independently produces a complete,
step-by-step solution to the problem. No cross-communication between Solvers
happens at this stage -- each call is made in isolation.
"""
from __future__ import annotations

import logging
from typing import Optional

from .. import config, utils
from ..models import get_model


def _build_user_prompt(problem: dict, solver_id: str) -> str:
    return (
        f"solver_id: {solver_id}\n\n"
        f"Problem ({problem['category']}, difficulty={problem['difficulty']}):\n"
        f"{problem['problem']}\n\n"
        "Solve this independently and return your answer in the required JSON shape."
    )


def run_stage1(
    problem: dict,
    role_assignment: dict[str, str],
    logger: logging.Logger,
) -> dict[str, Optional[utils.Solution]]:
    """Generate one independent solution per Solver seat.

    Returns a mapping {solver_id ("solver_1".."solver_3"): Solution | None}.
    A `None` entry means that Solver failed after retries and must be
    excluded from peer review per the README's error-handling rules.
    """
    system_prompt = utils.read_prompt("solver_system_prompt.txt")
    solutions: dict[str, Optional[utils.Solution]] = {}

    for seat in ("Solver_1", "Solver_2", "Solver_3"):
        solver_id = seat.lower()
        identity = role_assignment[seat]
        model = get_model(identity)
        user_prompt = _build_user_prompt(problem, solver_id)
        response_context = {
            "task": "solve",
            "problem_id": problem["id"],
            "solver_id": solver_id,
            "difficulty": problem["difficulty"],
            "ground_truth": problem["correct_answer"],
        }
        result = utils.call_llm_validated(
            model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=utils.Solution,
            stage="stage1_solve",
            logger=logger,
            response_context=response_context,
        )
        solutions[solver_id] = result
        if result is None:
            utils.log_error(
                logger, identity, "stage1_solve",
                f"{solver_id} failed to produce a valid solution; excluded from peer review",
            )
    return solutions


def save_stage1(problem_id: str, solutions: dict[str, Optional[utils.Solution]]) -> None:
    payload = {
        solver_id: (solution.model_dump() if solution is not None else None)
        for solver_id, solution in solutions.items()
    }
    utils.save_json(payload, config.STAGE_DIRS["stage1"] / f"{problem_id}.json")
