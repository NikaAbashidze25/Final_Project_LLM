"""Stage 0.5: Algorithmic Role Assignment.

A deterministic algorithm (no LLM call) turns the four Stage 0 self-assessments
into a final role mapping: exactly one Judge and three Solvers. Reproducible
given the same inputs, per the README's requirements.
"""
from __future__ import annotations

from typing import Optional

from .. import config, utils


def assign_roles(
    assessments: dict[str, Optional[utils.RoleAssessment]],
) -> dict[str, str]:
    """Deterministically map the 4 canonical models to {Solver_1..3, Judge}.

    Rules (see README Stage 0.5):
      - Exactly 1 model becomes Judge, the other 3 become Solvers.
      - Prefer models that ranked "Judge" first with the highest
        confidence_by_role["Judge"] for the Judge seat.
      - Ties (including missing/failed assessments, treated as lowest
        priority) are broken using `config.MODEL_PRIORITY`.
    """
    candidates = list(config.ALL_MODELS)
    priority_index = {identity: i for i, identity in enumerate(config.MODEL_PRIORITY)}

    def judge_rank_key(identity: str) -> tuple:
        assessment = assessments.get(identity)
        if assessment is None:
            # No valid self-assessment: least eligible for Judge, but still a
            # valid tie-break target via priority order.
            prefers_judge = False
            judge_confidence = -1.0
        else:
            prefers_judge = (
                bool(assessment.role_preferences)
                and assessment.role_preferences[0] == "Judge"
            )
            judge_confidence = assessment.confidence_by_role.get("Judge", 0.0)
        # Sort descending on (prefers_judge, judge_confidence), ascending on
        # priority index for deterministic tie-breaking.
        return (not prefers_judge, -judge_confidence, priority_index.get(identity, len(priority_index)))

    ordered = sorted(candidates, key=judge_rank_key)
    judge = ordered[0]
    solvers = [identity for identity in config.ALL_MODELS if identity != judge]
    # Keep solver ordering deterministic via MODEL_PRIORITY rather than
    # assessment order.
    solvers.sort(key=lambda identity: priority_index.get(identity, len(priority_index)))

    return {
        "Solver_1": solvers[0],
        "Solver_2": solvers[1],
        "Solver_3": solvers[2],
        "Judge": judge,
    }


def save_assignment(problem_id: str, assignment: dict[str, str]) -> None:
    utils.save_json(assignment, config.STAGE_DIRS["stage0"] / f"{problem_id}_role_assignment.json")
