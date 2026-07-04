"""Stage 2: Peer Review Round.

Each Solver reviews the solutions of the other Solvers (2 reviews per Solver when
all three are present, for 6 reviews total).
"""
from __future__ import annotations

import json
import logging
from typing import Any

from .. import utils
from ..models import get_model


def run_stage2(
    problem: dict[str, Any],
    assignment: dict[str, str],
    solutions: dict[str, dict],
    logger: logging.Logger,
) -> list[dict]:
    """Return a list of validated review dicts."""
    system_prompt = utils.read_prompt("reviewer_system_prompt.txt")
    reviews: list[dict] = []
    solver_ids = list(solutions.keys())
    seat_of = {v.lower(): k for k, v in assignment.items() if k.startswith("Solver")}

    for reviewer_id in solver_ids:
        reviewer_model = assignment[seat_of[reviewer_id]] if reviewer_id in seat_of \
            else solutions[reviewer_id]["model"]
        model = get_model(reviewer_model)
        reviewer_solution = solutions[reviewer_id]
        for reviewed_id in solver_ids:
            if reviewed_id == reviewer_id:
                continue
            reviewed = solutions[reviewed_id]
            answers_agree = utils.answers_match(
                reviewer_solution["final_answer"], reviewed["final_answer"]
            )
            user_prompt = (
                f"PROBLEM:\n{problem['problem']}\n\n"
                f"You are {reviewer_id}. Here is YOUR OWN independently-derived final "
                f"answer to this same problem, reached before seeing anyone else's work:\n"
                f"{json.dumps(reviewer_solution['final_answer'])}\n\n"
                f"Critically review the following solution written by {reviewed_id}:\n\n"
                f"{json.dumps({k: v for k, v in reviewed.items() if not k.startswith('_')}, indent=2)}\n\n"
                f"Per the SELF-CONSISTENCY RULE: compare {reviewed_id}'s final answer to "
                f"your own final answer above before writing your review.\n\n"
                f"Return the required review JSON. Set reviewer_id to '{reviewer_id}' and "
                f"solution_reviewed to '{reviewed_id}'."
            )
            ctx = {
                "task": "review",
                "problem_id": problem["id"],
                "reviewer_id": reviewer_id,
                "solution_reviewed": reviewed_id,
                "target_correct": bool(reviewed.get("_correct", False)),
            }
            result = utils.call_llm_validated(
                model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=utils.Review,
                stage="stage2_review",
                logger=logger,
                response_context=ctx,
            )
            if result is None:
                logger.warning("stage2 | %s reviewing %s failed; skipping this review",
                               reviewer_id, reviewed_id)
                continue
            rev = result.model_dump()
            rev["reviewer_id"] = reviewer_id
            rev["solution_reviewed"] = reviewed_id
            # Deterministic backstop: a reviewer cannot logically call a solution
            # "fundamentally_wrong" while its own independently-derived answer
            # agrees with it. Downgrade if the model ignored the self-consistency
            # rule above instead of silently trusting a contradictory verdict.
            if answers_agree and rev["overall_assessment"] == "fundamentally_wrong":
                downgraded_to = "promising_but_flawed" if rev["evaluation"]["errors"] else "correct"
                logger.warning(
                    "stage2 | %s -> %s : overridden 'fundamentally_wrong' -> %r "
                    "(reviewer's own answer agrees with the reviewed answer)",
                    reviewer_id, reviewed_id, downgraded_to,
                )
                rev["overall_assessment"] = downgraded_to
            # Internal metric: did this review flag a genuine error?
            rev["_flagged_real_error"] = (
                bool(rev["evaluation"]["errors"]) and not reviewed.get("_correct", False)
            )
            reviews.append(rev)
            logger.info("stage2 | %s -> %s : %s (%d errors flagged)",
                        reviewer_id, reviewed_id, rev["overall_assessment"],
                        len(rev["evaluation"]["errors"]))

    return reviews
