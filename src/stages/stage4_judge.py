"""Stage 4: Final Judgment.

The assigned Judge receives the full debate record -- all original solutions,
all peer reviews, and all refined solutions -- and selects the strongest
final answer. If the Judge fails after retries, the README requires falling
back to the refined solution with the highest self-reported confidence.
"""
from __future__ import annotations

import logging
from typing import Optional

from .. import config, utils
from ..models import get_model


def _effective_answer(solver_id: str, solutions, refinements) -> tuple[str, float]:
    """Return (answer, confidence) for a solver, preferring the refinement."""
    refinement = refinements.get(solver_id)
    if refinement is not None:
        return refinement.refined_final_answer, refinement.confidence
    solution = solutions.get(solver_id)
    if solution is not None:
        return solution.final_answer, solution.confidence
    return "", 0.0


def _build_user_prompt(
    problem: dict,
    solutions: dict[str, Optional[utils.Solution]],
    reviews: dict[str, Optional[utils.Review]],
    refinements: dict[str, Optional[utils.Refinement]],
) -> str:
    parts = [f"Problem:\n{problem['problem']}\n"]

    parts.append("Original solutions:")
    for solver_id, solution in solutions.items():
        if solution is None:
            parts.append(f"  {solver_id}: (failed to produce a solution)")
            continue
        parts.append(
            f"  {solver_id} (model={solution.model}): final_answer={solution.final_answer!r} "
            f"confidence={solution.confidence}"
        )

    parts.append("\nPeer reviews:")
    for key, review in reviews.items():
        if review is None:
            continue
        parts.append(
            f"  {key}: overall_assessment={review.overall_assessment!r} "
            f"errors={[e.error_type for e in review.evaluation.errors]}"
        )

    parts.append("\nRefined solutions:")
    for solver_id, refinement in refinements.items():
        if refinement is None:
            parts.append(f"  {solver_id}: (no refinement; original solution stands)")
            continue
        parts.append(
            f"  {solver_id}: refined_final_answer={refinement.refined_final_answer!r} "
            f"confidence={refinement.confidence} changes_summary={refinement.changes_summary!r}"
        )

    parts.append(
        "\nSelect the strongest final answer and return your judgment in the required JSON shape."
    )
    return "\n".join(parts)


def _fallback_judgment(
    judge_identity: str,
    solutions: dict[str, Optional[utils.Solution]],
    refinements: dict[str, Optional[utils.Refinement]],
) -> utils.Judgment:
    """Highest-confidence refined solution wins, per the README's error handling."""
    candidates = [sid for sid, sol in solutions.items() if sol is not None]
    scored = [
        (sid, *_effective_answer(sid, solutions, refinements))
        for sid in candidates
    ]
    winner_id, winner_answer, _ = max(scored, key=lambda item: item[2], default=(None, "", 0.0))
    ordered = sorted(scored, key=lambda item: -item[2])
    ranking = {sid: i + 1 for i, (sid, _, _) in enumerate(ordered)}
    return utils.Judgment(
        judge_model=judge_identity,
        winner=winner_id or "",
        confidence=next((conf for sid, _, conf in scored if sid == winner_id), 0.0),
        ranking=ranking,
        reasoning="Judge failed to produce a valid response after retries; fell back to the "
                  "refined solution with the highest self-reported confidence.",
        correct_answer=winner_answer,
        notes="Fallback judgment (see README Error Handling Requirements).",
    )


def run_stage4(
    problem: dict,
    role_assignment: dict[str, str],
    solutions: dict[str, Optional[utils.Solution]],
    reviews: dict[str, Optional[utils.Review]],
    refinements: dict[str, Optional[utils.Refinement]],
    logger: logging.Logger,
) -> utils.Judgment:
    judge_identity = role_assignment["Judge"]
    model = get_model(judge_identity)
    system_prompt = utils.read_prompt("judge_system_prompt.txt")
    user_prompt = _build_user_prompt(problem, solutions, reviews, refinements)

    candidates_ctx = []
    for solver_id, solution in solutions.items():
        if solution is None:
            continue
        answer, confidence = _effective_answer(solver_id, solutions, refinements)
        candidates_ctx.append({
            "solver_id": solver_id,
            "answer": answer,
            "confidence": confidence,
            "correct": utils.answers_match(answer, problem["correct_answer"]),
        })

    response_context = {
        "task": "judge",
        "problem_id": problem["id"],
        "candidates": candidates_ctx,
    }
    result = utils.call_llm_validated(
        model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=utils.Judgment,
        stage="stage4_judge",
        logger=logger,
        response_context=response_context,
    )
    if result is None:
        utils.log_error(
            logger, judge_identity, "stage4_judge",
            "Judge failed to produce a valid judgment; using confidence-based fallback",
        )
        result = _fallback_judgment(judge_identity, solutions, refinements)
    return result


def save_stage4(problem_id: str, judgment: utils.Judgment) -> None:
    utils.save_json(judgment.model_dump(), config.STAGE_DIRS["stage4"] / f"{problem_id}.json")
