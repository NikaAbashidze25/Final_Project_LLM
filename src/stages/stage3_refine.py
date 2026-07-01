"""Stage 3: Refinement Based on Feedback.

Each surviving Solver receives the peer reviews written about their Stage 1
solution and produces a refined solution that explicitly addresses every
critique (accept-and-fix, or reject-with-justification).
"""
from __future__ import annotations

import logging
from typing import Optional

from .. import config, utils
from ..models import get_model
from .stage2_review import reviews_for_target


def _critique_summary(review: utils.Review) -> str:
    if review.evaluation.errors:
        first = review.evaluation.errors[0]
        return f"[{first.location}] {first.description}"
    if review.evaluation.weaknesses:
        return review.evaluation.weaknesses[0]
    return "No specific critique raised."


def _build_user_prompt(
    problem: dict, solution: utils.Solution, received_reviews: list[utils.Review]
) -> str:
    steps = "\n".join(
        f"  Step {step.step}: {step.description} (reasoning: {step.reasoning})"
        for step in solution.solution_steps
    )
    critiques_text = "\n\n".join(
        f"  From {review.reviewer_id} (assessment: {review.overall_assessment}):\n"
        f"    Weaknesses: {review.evaluation.weaknesses}\n"
        f"    Errors: {[e.model_dump() for e in review.evaluation.errors]}\n"
        f"    Suggested changes: {review.evaluation.suggested_changes}"
        for review in received_reviews
    ) or "  (No peer reviews were received; still confirm your solution is complete.)"

    return (
        f"solver_id: {solution.solver_id}\n\n"
        f"Problem:\n{problem['problem']}\n\n"
        f"Your original solution:\n{steps}\n"
        f"  Final answer: {solution.final_answer}\n"
        f"  Confidence: {solution.confidence}\n\n"
        f"Peer reviews received:\n{critiques_text}\n\n"
        "Address every critique above, then return your refined solution in the "
        "required JSON shape."
    )


def run_stage3(
    problem: dict,
    role_assignment: dict[str, str],
    solutions: dict[str, Optional[utils.Solution]],
    reviews: dict[str, Optional[utils.Review]],
    logger: logging.Logger,
) -> dict[str, Optional[utils.Refinement]]:
    """Produce one refinement per surviving Solver."""
    system_prompt = utils.read_prompt("refiner_system_prompt.txt")
    seat_by_solver_id = {"solver_1": "Solver_1", "solver_2": "Solver_2", "solver_3": "Solver_3"}

    refinements: dict[str, Optional[utils.Refinement]] = {}
    for solver_id, solution in solutions.items():
        if solution is None:
            continue
        identity = role_assignment[seat_by_solver_id[solver_id]]
        model = get_model(identity)
        received_reviews = reviews_for_target(reviews, solver_id)
        user_prompt = _build_user_prompt(problem, solution, received_reviews)

        critiques_ctx = [
            {
                "from": review.reviewer_id,
                "summary": _critique_summary(review),
                "had_error": bool(review.evaluation.errors),
            }
            for review in received_reviews
        ]
        response_context = {
            "task": "refine",
            "problem_id": problem["id"],
            "solver_id": solver_id,
            "ground_truth": problem["correct_answer"],
            "was_correct": utils.answers_match(solution.final_answer, problem["correct_answer"]),
            "received_valid_critique": any(c["had_error"] for c in critiques_ctx),
            "critiques": critiques_ctx,
        }
        result = utils.call_llm_validated(
            model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=utils.Refinement,
            stage="stage3_refine",
            logger=logger,
            response_context=response_context,
        )
        refinements[solver_id] = result
        if result is None:
            utils.log_error(
                logger, identity, "stage3_refine",
                f"{solver_id} failed to produce a valid refinement; "
                "original Stage 1 solution will stand in for it",
            )
    return refinements


def save_stage3(problem_id: str, refinements: dict[str, Optional[utils.Refinement]]) -> None:
    payload = {
        solver_id: (refinement.model_dump() if refinement is not None else None)
        for solver_id, refinement in refinements.items()
    }
    utils.save_json(payload, config.STAGE_DIRS["stage3"] / f"{problem_id}.json")
