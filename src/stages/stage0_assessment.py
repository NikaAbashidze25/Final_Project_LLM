"""Stage 0: Role Self-Assessment.

Every one of the four LLMs receives the problem statement and self-assesses
which role ("Solver" or "Judge") it is best suited for on this problem. All
four assessments must be collected before Stage 0.5 (role assignment) runs.
"""
from __future__ import annotations

import logging
from typing import Optional

from .. import config, utils
from ..models import get_model


def _build_user_prompt(problem: dict) -> str:
    return (
        f"Problem ({problem['category']}, difficulty={problem['difficulty']}):\n"
        f"{problem['problem']}\n\n"
        "Self-assess which role you are best suited for on this specific problem."
    )


def run_stage0(problem: dict, logger: logging.Logger) -> dict[str, Optional[utils.RoleAssessment]]:
    """Collect a role self-assessment from every model in `config.ALL_MODELS`.

    Returns a mapping {model_identity: RoleAssessment | None}. A `None` entry
    means the model failed to produce a valid assessment after retries; the
    caller (Stage 0.5) must handle missing entries gracefully.
    """
    system_prompt = utils.read_prompt("role_assessment_prompt.txt")
    user_prompt = _build_user_prompt(problem)

    assessments: dict[str, Optional[utils.RoleAssessment]] = {}
    for identity in config.ALL_MODELS:
        model = get_model(identity)
        response_context = {
            "task": "assess",
            "problem_id": problem["id"],
        }
        result = utils.call_llm_validated(
            model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=utils.RoleAssessment,
            stage="stage0_assessment",
            logger=logger,
            response_context=response_context,
        )
        assessments[identity] = result
        if result is None:
            utils.log_error(
                logger, identity, "stage0_assessment",
                "no valid self-assessment obtained; will be excluded from role assignment",
            )
    return assessments


def save_stage0(problem_id: str, assessments: dict[str, Optional[utils.RoleAssessment]]) -> None:
    payload = {
        identity: (assessment.model_dump() if assessment is not None else None)
        for identity, assessment in assessments.items()
    }
    utils.save_json(payload, config.STAGE_DIRS["stage0"] / f"{problem_id}.json")
