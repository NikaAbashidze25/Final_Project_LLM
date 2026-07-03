"""Stage 0.5: Algorithmic (deterministic) Role Assignment.

A pure, reproducible algorithm -- no LLM -- assigns exactly one Judge and three
Solvers from the four models, based on their Stage 0 self-assessments.

Rules:
  * Exactly 1 Judge, exactly 3 Solvers.
  * The Judge seat goes to the model that listed "Judge" in its preferences and
    has the highest Judge confidence. Models that did not list Judge are
    considered only after those that did.
  * Ties are broken by a fixed model priority order (config.MODEL_PRIORITY).
  * Same inputs always produce the same output.
"""
from __future__ import annotations

import logging
from typing import Any

from .. import config


def assign_roles(assessments: dict[str, dict], logger: logging.Logger) -> dict[str, str]:
    """Return {"Solver_1": model, "Solver_2": model, "Solver_3": model, "Judge": model}."""
    priority_index = {m: i for i, m in enumerate(config.MODEL_PRIORITY)}

    def judge_sort_key(model_id: str):
        a = assessments[model_id]
        listed_judge = "Judge" in a.get("role_preferences", [])
        judge_conf = float(a.get("confidence_by_role", {}).get("Judge", 0.0))
        # Sort so the best Judge candidate is first:
        #  1) prefer those who listed Judge (listed first -> sort key False < True),
        #  2) then higher judge confidence,
        #  3) then fixed priority order (lower index first).
        return (not listed_judge, -judge_conf, priority_index.get(model_id, 99))

    ranked = sorted(config.ALL_MODELS, key=judge_sort_key)
    judge = ranked[0]

    # Remaining three become solvers, ordered deterministically by priority.
    solvers = sorted(
        [m for m in config.ALL_MODELS if m != judge],
        key=lambda m: priority_index.get(m, 99),
    )

    assignment = {
        "Solver_1": solvers[0],
        "Solver_2": solvers[1],
        "Solver_3": solvers[2],
        "Judge": judge,
    }
    logger.info("stage0.5 | role assignment: %s", assignment)
    return assignment
