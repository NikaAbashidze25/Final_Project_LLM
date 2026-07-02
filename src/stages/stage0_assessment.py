"""Stage 0: Role Self-Assessment.

Each of the four LLMs receives the problem and self-assesses which role
(Solver / Judge) it is best suited for, returning a structured JSON assessment.
"""
from __future__ import annotations

import logging
from typing import Any

from .. import config, utils
from ..models import get_model


def run_stage0(problem: dict[str, Any], logger: logging.Logger) -> dict[str, dict]:
    """Return a mapping {model_identity -> validated assessment dict}."""
    system_prompt = utils.read_prompt("role_assessment_prompt.txt")
    assessments: dict[str, dict] = {}

    for model_id in config.ALL_MODELS:
        model = get_model(model_id)
        user_prompt = (
            f"PROBLEM:\n{problem['problem']}\n\n"
            f"Available roles: Solver, Judge.\n"
            f"You are the model identified as '{model_id}'. "
            f"Self-assess and return the required JSON."
        )
        ctx = {"task": "assess", "problem_id": problem["id"], "model": model_id}
        result = utils.call_llm_validated(
            model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=utils.RoleAssessment,
            stage="stage0_assessment",
            logger=logger,
            response_context=ctx,
        )
        if result is None:
            # Fallback assessment so the pipeline can still proceed.
            logger.warning("model=%s | stage0 | using fallback assessment", model_id)
            assessments[model_id] = {
                "role_preferences": ["Solver", "Judge"],
                "confidence_by_role": {"Solver": 0.5, "Judge": 0.5},
                "reasoning": "Fallback assessment (model failed to respond).",
            }
        else:
            assessments[model_id] = result.model_dump()
        logger.info("stage0 | %s assessed: Judge conf=%.2f",
                    model_id, assessments[model_id]["confidence_by_role"]["Judge"])

    return assessments
