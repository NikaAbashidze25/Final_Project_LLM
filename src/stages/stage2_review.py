"""Stage 2: Peer Review Round.

Each surviving Solver reviews the solutions produced by the other two
Solvers (2 reviews per Solver, 6 total when all three are present). A Solver
that failed Stage 1 is excluded both as a reviewer and as a review target,
per the README's error-handling rules.
"""
from __future__ import annotations

import itertools
import logging
from typing import Optional

from .. import config, utils
from ..models import get_model


def _build_user_prompt(problem: dict, reviewer_id: str, target: utils.Solution) -> str:
    steps = "\n".join(
        f"  Step {step.step}: {step.description} (reasoning: {step.reasoning})"
        for step in target.solution_steps
    )
    return (
        f"reviewer_id: {reviewer_id}\n"
        f"solution_reviewed: {target.solver_id}\n\n"
        f"Problem:\n{problem['problem']}\n\n"
        f"Solution to review (by {target.solver_id}):\n{steps}\n"
        f"  Final answer: {target.final_answer}\n"
        f"  Stated confidence: {target.confidence}\n"
        f"  Assumptions: {target.assumptions}\n\n"
        "Critically review this solution and return your evaluation in the required JSON shape."
    )


def run_stage2(
    problem: dict,
    role_assignment: dict[str, str],
    solutions: dict[str, Optional[utils.Solution]],
    logger: logging.Logger,
) -> dict[str, Optional[utils.Review]]:
    """Produce one review per (reviewer, target) pair among surviving Solvers.

    Returns a mapping keyed by "{reviewer_id}__reviews__{target_id}" to
    Review | None.
    """
    system_prompt = utils.read_prompt("reviewer_system_prompt.txt")
    seat_by_solver_id = {"solver_1": "Solver_1", "solver_2": "Solver_2", "solver_3": "Solver_3"}
    surviving = [sid for sid, sol in solutions.items() if sol is not None]

    reviews: dict[str, Optional[utils.Review]] = {}
    for reviewer_id, target_id in itertools.permutations(surviving, 2):
        target = solutions[target_id]
        identity = role_assignment[seat_by_solver_id[reviewer_id]]
        model = get_model(identity)
        user_prompt = _build_user_prompt(problem, reviewer_id, target)
        response_context = {
            "task": "review",
            "problem_id": problem["id"],
            "reviewer_id": reviewer_id,
            "solution_reviewed": target_id,
            "target_correct": utils.answers_match(target.final_answer, problem["correct_answer"]),
        }
        result = utils.call_llm_validated(
            model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=utils.Review,
            stage="stage2_review",
            logger=logger,
            response_context=response_context,
        )
        key = f"{reviewer_id}__reviews__{target_id}"
        reviews[key] = result
        if result is None:
            utils.log_error(
                logger, identity, "stage2_review",
                f"{reviewer_id} failed to produce a valid review of {target_id}",
            )
    return reviews


def reviews_for_target(
    reviews: dict[str, Optional[utils.Review]], target_id: str
) -> list[utils.Review]:
    """Return the (valid) reviews received by a given solver, for Stage 3."""
    return [
        review for key, review in reviews.items()
        if review is not None and key.endswith(f"__reviews__{target_id}")
    ]


def save_stage2(problem_id: str, reviews: dict[str, Optional[utils.Review]]) -> None:
    payload = {
        key: (review.model_dump() if review is not None else None)
        for key, review in reviews.items()
    }
    utils.save_json(payload, config.STAGE_DIRS["stage2"] / f"{problem_id}.json")
