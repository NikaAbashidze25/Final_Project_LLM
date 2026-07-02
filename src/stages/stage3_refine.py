"""Stage 3: Refinement Based on Feedback.

Each Solver receives the two peer reviews written about its solution and produces
a refined solution that addresses every critique point-by-point.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from .. import utils
from ..models import get_model


def run_stage3(
    problem: dict[str, Any],
    assignment: dict[str, str],
    solutions: dict[str, dict],
    reviews: list[dict],
    logger: logging.Logger,
) -> dict[str, dict]:
    """Return {solver_id -> validated refinement dict}."""
    system_prompt = utils.read_prompt("refiner_system_prompt.txt")
    refinements: dict[str, dict] = {}
    seat_of = {v.lower(): k for k, v in assignment.items() if k.startswith("Solver")}

    for solver_id, solution in solutions.items():
        my_reviews = [r for r in reviews if r["solution_reviewed"] == solver_id]
        model_id = assignment[seat_of[solver_id]] if solver_id in seat_of else solution["model"]
        model = get_model(model_id)

        critiques_struct = []
        for r in my_reviews:
            had_error = bool(r["evaluation"]["errors"])
            summary = "; ".join(
                e.get("description", "") for e in r["evaluation"]["errors"]
            ) or "General concerns about rigor."
            critiques_struct.append({
                "from": r["reviewer_id"],
                "summary": summary[:200],
                "had_error": had_error,
            })

        received_valid_critique = any(
            c["had_error"] for c in critiques_struct
        ) and not solution.get("_correct", False)

        user_prompt = (
            f"PROBLEM:\n{problem['problem']}\n\n"
            f"You are {solver_id} (model: {model_id}). Here was YOUR original solution:\n"
            f"{json.dumps({k: v for k, v in solution.items() if not k.startswith('_')}, indent=2)}\n\n"
            f"Here are the {len(my_reviews)} peer reviews about your solution:\n"
            f"{json.dumps([{k: v for k, v in r.items() if not k.startswith('_')} for r in my_reviews], indent=2)}\n\n"
            f"Refine your solution and return the required JSON. Provide a critique_responses "
            f"entry for each review."
        )
        ctx = {
            "task": "refine",
            "problem_id": problem["id"],
            "solver_id": solver_id,
            "ground_truth": problem.get("correct_answer", ""),
            "was_correct": bool(solution.get("_correct", False)),
            "received_valid_critique": received_valid_critique,
            "critiques": critiques_struct,
        }
        result = utils.call_llm_validated(
            model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=utils.Refinement,
            stage="stage3_refine",
            logger=logger,
            response_context=ctx,
        )
        if result is None:
            # Fallback: keep the original solution as the "refined" one.
            logger.warning("stage3 | %s refinement failed; keeping original solution", solver_id)
            refinements[solver_id] = {
                "solver_id": solver_id,
                "model": model_id,
                "critique_responses": [],
                "refined_solution_steps": solution["solution_steps"],
                "refined_final_answer": solution["final_answer"],
                "confidence": solution["confidence"],
                "changes_summary": "Refinement failed; original solution retained.",
                "_correct": solution.get("_correct", False),
            }
            continue
        ref = result.model_dump()
        ref["solver_id"] = solver_id
        ref["model"] = model_id
        ref["_correct"] = utils.answers_match(
            ref["refined_final_answer"], problem.get("correct_answer", "")
        )
        refinements[solver_id] = ref
        logger.info("stage3 | %s refined -> answer=%r conf=%.2f (was_correct=%s now_correct=%s)",
                    solver_id, ref["refined_final_answer"], ref["confidence"],
                    solution.get("_correct"), ref["_correct"])

    return refinements
