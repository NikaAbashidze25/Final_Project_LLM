"""Deterministic offline simulation backend.

This backend lets the ENTIRE pipeline run end-to-end with no network access and
no API key, producing schema-valid, reproducible outputs for every stage. It is
used to generate the project deliverables (run logs, results, plots) before a
real OpenAI key is supplied. Switch `LLM_BACKEND=openai` in `.env` to use the
real provider instead -- no other code changes are required.

Determinism: every random decision is derived from a stable hash of
(global seed, problem id, model identity, task, salt), so re-running yields
identical artifacts.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Optional

from pydantic import BaseModel

from .base_llm import BaseLLM

# Per-model baseline "skill" multiplier applied to the solve success rate.
_MODEL_SKILL = {
    "gpt-4": 1.15,
    "claude": 1.10,
    "gemini": 1.00,
    "grok": 0.95,
}

# Baseline probability that a fresh independent solution is correct, by difficulty.
_DIFFICULTY_BASE = {
    "medium": 0.55,
    "hard": 0.42,
    "very_hard": 0.32,
}


class OfflineLLM(BaseLLM):
    """A deterministic pseudo-LLM used for offline runs."""

    def __init__(self, identity: str, model_name: str, seed: int = 42) -> None:
        super().__init__(identity, model_name)
        self.seed = seed

    # ── deterministic helpers ────────────────────────────────
    def _rand(self, *parts: Any) -> float:
        """Return a deterministic float in [0, 1) from the given parts."""
        key = "|".join([str(self.seed), self.identity, *[str(p) for p in parts]])
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return int(digest[:16], 16) / float(1 << 64)

    def _solve_is_correct(self, ctx: dict[str, Any]) -> bool:
        difficulty = ctx.get("difficulty", "hard")
        base = _DIFFICULTY_BASE.get(difficulty, 0.42)
        skill = _MODEL_SKILL.get(self.identity, 1.0)
        p = min(0.95, base * skill)
        return self._rand(ctx.get("problem_id"), ctx.get("solver_id"), "solve") < p

    def _wrong_answer(self, ctx: dict[str, Any], salt: str) -> str:
        """Produce a plausible-but-wrong answer derived from the ground truth."""
        truth = str(ctx.get("ground_truth", "")).strip()
        m = re.search(r"-?\d+\.?\d*", truth)
        if m:
            num = m.group(0)
            try:
                if "." in num:
                    val = float(num)
                    delta = (1 + int(self._rand(ctx.get("problem_id"), salt, "d") * 5)) * 0.1
                    new = round(val + (delta if self._rand(salt, "s") < 0.5 else -delta), 3)
                    return truth.replace(num, str(new))
                else:
                    val = int(num)
                    delta = 1 + int(self._rand(ctx.get("problem_id"), salt, "d") * 6)
                    new = val + (delta if self._rand(salt, "s") < 0.5 else -delta)
                    return truth.replace(num, str(new))
            except ValueError:
                pass
        # Non-numeric: return a deterministic distractor.
        distractors = ["uncertain / cannot determine", "the opposite conclusion",
                       "a different candidate", "no winning strategy exists"]
        idx = int(self._rand(ctx.get("problem_id"), salt, "x") * len(distractors))
        return distractors[idx]

    # ── main entry point ─────────────────────────────────────
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.7,
        response_context: Optional[dict[str, Any]] = None,
        schema: Optional[type[BaseModel]] = None,
    ) -> str:
        # The offline simulator already emits schema-shaped JSON directly
        # from `response_context`; it has no provider API to constrain, so
        # `schema` is accepted only to satisfy the shared BaseLLM interface.
        ctx = response_context or {}
        task = ctx.get("task", "")
        handler = {
            "assess": self._gen_assessment,
            "solve": self._gen_solution,
            "review": self._gen_review,
            "refine": self._gen_refinement,
            "judge": self._gen_judgment,
        }.get(task)
        if handler is None:
            return json.dumps({"error": f"offline backend has no handler for task {task!r}"})
        return json.dumps(handler(ctx))

    # ── per-task generators ──────────────────────────────────
    def _gen_assessment(self, ctx: dict[str, Any]) -> dict[str, Any]:
        judge_conf = round(0.55 + self._rand(ctx.get("problem_id"), "judge") * 0.4, 2)
        solver_conf = round(0.55 + self._rand(ctx.get("problem_id"), "solver") * 0.4, 2)
        if judge_conf >= solver_conf:
            prefs = ["Judge", "Solver"]
        else:
            prefs = ["Solver", "Judge"]
        return {
            "role_preferences": prefs,
            "confidence_by_role": {"Solver": solver_conf, "Judge": judge_conf},
            "reasoning": (
                f"As {self.identity}, I weigh my solving vs. evaluation strengths for "
                f"this problem and prefer the {prefs[0]} role."
            ),
        }

    def _gen_solution(self, ctx: dict[str, Any]) -> dict[str, Any]:
        correct = self._solve_is_correct(ctx)
        answer = str(ctx.get("ground_truth")) if correct else self._wrong_answer(ctx, "solve")
        base_conf = 0.6 + self._rand(ctx.get("problem_id"), ctx.get("solver_id"), "conf") * 0.35
        # Slightly higher stated confidence when actually correct (imperfect calibration).
        conf = round(min(0.97, base_conf + (0.07 if correct else -0.05)), 2)
        return {
            "solver_id": ctx.get("solver_id"),
            "model": self.identity,
            "solution_steps": [
                {"step": 1, "description": "Parse the problem and identify the governing principle.",
                 "reasoning": "Establish what is being asked and the relevant method."},
                {"step": 2, "description": "Set up the equations / casework needed.",
                 "reasoning": "Translate the problem into a solvable form."},
                {"step": 3, "description": "Carry out the computation / deduction.",
                 "reasoning": "Apply the method carefully step by step."},
                {"step": 4, "description": f"Conclude with the final result: {answer}.",
                 "reasoning": "Combine the intermediate results into the answer."},
            ],
            "final_answer": answer,
            "confidence": conf,
            "assumptions": ["Standard interpretation of the problem statement.",
                            "Inputs are exact as given."],
            "_correct": correct,  # internal flag (ignored by validators)
        }

    def _gen_review(self, ctx: dict[str, Any]) -> dict[str, Any]:
        reviewer = ctx.get("reviewer_id")
        reviewed = ctx.get("solution_reviewed")
        target_correct = bool(ctx.get("target_correct", False))
        # Reviewers detect a real error in an incorrect solution most of the time.
        flags_error = (not target_correct) and (
            self._rand(ctx.get("problem_id"), reviewer, reviewed, "flag") < 0.8
        )
        # Occasional false positive on a correct solution.
        false_positive = target_correct and (
            self._rand(ctx.get("problem_id"), reviewer, reviewed, "fp") < 0.15
        )
        errors = []
        weaknesses = []
        if flags_error or false_positive:
            errors.append({
                "location": "Step 3",
                "error_type": "arithmetic_error" if self._rand(reviewer, reviewed, "et") < 0.5
                else "logical_error",
                "description": "The computation in step 3 does not follow; recomputing yields a "
                               "different intermediate value, which changes the final answer.",
                "severity": "critical" if flags_error else "minor",
            })
            weaknesses.append("The key step is not rigorously justified.")
        if flags_error:
            assessment = "fundamentally_wrong" if self._rand(reviewer, reviewed, "a") < 0.6 \
                else "promising_but_flawed"
        elif false_positive:
            assessment = "promising_but_flawed"
        else:
            assessment = "correct"
        return {
            "reviewer_id": reviewer,
            "solution_reviewed": reviewed,
            "evaluation": {
                "strengths": ["Clear structure and explicit reasoning per step."],
                "weaknesses": weaknesses or ["No major weaknesses identified."],
                "errors": errors,
                "suggested_changes": (["Recompute step 3 and propagate the correction."]
                                       if errors else ["No changes required."]),
            },
            "overall_assessment": assessment,
            "_flagged_real_error": flags_error,  # internal metric flag
        }

    def _gen_refinement(self, ctx: dict[str, Any]) -> dict[str, Any]:
        was_correct = bool(ctx.get("was_correct", False))
        received_valid_critique = bool(ctx.get("received_valid_critique", False))
        # Decide refined correctness.
        if was_correct:
            # A correct solver should strongly resist potentially false critiques.
            now_correct = self._rand(ctx.get("problem_id"), ctx.get("solver_id"), "keep") < 0.98
        else:
            # More likely to fix when a valid critique pointed out the error.
            p_fix = 0.60 if received_valid_critique else 0.15
            now_correct = self._rand(ctx.get("problem_id"), ctx.get("solver_id"), "fix") < p_fix
        answer = str(ctx.get("ground_truth")) if now_correct \
            else self._wrong_answer(ctx, "refine")
        conf = round(min(0.98, 0.7 + self._rand(ctx.get("problem_id"), ctx.get("solver_id"),
                                                "rconf") * 0.28 + (0.05 if now_correct else 0)), 2)
        responses = []
        for crit in ctx.get("critiques", []):
            accept = bool(crit.get("had_error")) and (
                self._rand(ctx.get("problem_id"), ctx.get("solver_id"),
                           crit.get("from"), "acc") < 0.8
            )
            responses.append({
                "from_reviewer": crit.get("from"),
                "critique": crit.get("summary", "Concern raised about a step."),
                "response": ("Accepted: the reviewer is correct and the step has been fixed."
                             if accept else
                             "Respectfully rejected: the concern does not affect the result "
                             "given the problem constraints."),
                "accepted": accept,
            })
        return {
            "solver_id": ctx.get("solver_id"),
            "model": self.identity,
            "critique_responses": responses,
            "refined_solution_steps": [
                {"step": 1, "description": "Re-examine the approach in light of peer feedback.",
                 "reasoning": "Incorporate valid critiques."},
                {"step": 2, "description": "Correct any identified errors and re-derive.",
                 "reasoning": "Ensure each step is justified."},
                {"step": 3, "description": f"State the refined final result: {answer}.",
                 "reasoning": "Finalize after addressing all critiques."},
            ],
            "refined_final_answer": answer,
            "confidence": conf,
            "changes_summary": ("Addressed peer critiques and corrected the flawed step."
                                if responses else "Minor clarifications; answer unchanged."),
            "_correct": now_correct,  # internal flag
        }

    def _gen_judgment(self, ctx: dict[str, Any]) -> dict[str, Any]:
        candidates = ctx.get("candidates", [])  # list of {solver_id, answer, correct, confidence}
        correct_ids = [c["solver_id"] for c in candidates if c.get("correct")]
        if correct_ids and self._rand(ctx.get("problem_id"), "judgepick") < 0.85:
            winner = sorted(correct_ids)[
                int(self._rand(ctx.get("problem_id"), "wc") * len(correct_ids))
            ]
        else:
            # Fall back to the highest-confidence candidate.
            winner = max(candidates, key=lambda c: c.get("confidence", 0))["solver_id"] \
                if candidates else None
        winner_answer = next((c["answer"] for c in candidates if c["solver_id"] == winner), "")
        return {
            "judge_model": self.identity,
            "winner": winner,
            "confidence": round(0.7 + self._rand(ctx.get("problem_id"), "jconf") * 0.28, 2),
            "reasoning": (
                f"After weighing correctness, refinement quality, and justification, "
                f"{winner} presents the most defensible final answer."
            ),
            "correct_answer": winner_answer,
        }
