"""End-to-end orchestrator for the Multi-LLM Debate System.

Runs all stages for one or more problems, persists every intermediate output to
disk (so interrupted runs can be resumed/inspected), derives the two baselines
(single-LLM and majority voting) from the same data, and writes a consolidated
`final_answers.json`.

Usage:
    python src/pipeline.py --problems data/problems.json --output data/results/
    python src/pipeline.py --problem-id prob_001 --problems data/problems.json --output data/results/
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Allow running both as `python src/pipeline.py` and `python -m src.pipeline`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config, utils  # type: ignore
    from src.models import get_model  # type: ignore
    from src.stages.stage0_assessment import run_stage0  # type: ignore
    from src.stages.stage0_5_assignment import assign_roles  # type: ignore
    from src.stages.stage1_solve import run_stage1  # type: ignore
    from src.stages.stage2_review import run_stage2  # type: ignore
    from src.stages.stage3_refine import run_stage3  # type: ignore
    from src.stages.stage4_judge import run_stage4  # type: ignore
else:
    from . import config, utils
    from .models import get_model
    from .stages.stage0_assessment import run_stage0
    from .stages.stage0_5_assignment import assign_roles
    from .stages.stage1_solve import run_stage1
    from .stages.stage2_review import run_stage2
    from .stages.stage3_refine import run_stage3
    from .stages.stage4_judge import run_stage4

SINGLE_BASELINE_MODEL = "gpt-4"  # fixed model for the single-LLM baseline


def _majority_answer(answers: list[str]) -> Optional[str]:
    """Return the majority answer (by normalized form), or None if no majority."""
    if not answers:
        return None
    norm = [utils.normalize_answer(a) for a in answers]
    counts = Counter(norm)
    top_norm, top_count = counts.most_common(1)[0]
    if top_count < 2:  # no agreement among 3 -> no majority
        return None
    # Return an original (non-normalized) answer matching the winning form.
    for original, n in zip(answers, norm):
        if n == top_norm:
            return original
    return None


def run_single_baseline(problem: dict[str, Any], logger) -> dict[str, Any]:
    """Single-LLM baseline: one fixed model solves the problem once, no debate."""
    model = get_model(SINGLE_BASELINE_MODEL)
    system_prompt = utils.read_prompt("solver_system_prompt.txt")
    user_prompt = (
        f"PROBLEM:\n{problem['problem']}\n\n"
        f"Solve this problem and return the required JSON. Your solver_id is "
        f"'baseline' and your model is '{SINGLE_BASELINE_MODEL}'."
    )
    ctx = {
        "task": "solve",
        "problem_id": problem["id"],
        "solver_id": "baseline_single",
        "difficulty": problem.get("difficulty", "hard"),
        "ground_truth": problem.get("correct_answer", ""),
    }
    result = utils.call_llm_validated(
        model, system_prompt=system_prompt, user_prompt=user_prompt,
        schema=utils.Solution, stage="baseline_single", logger=logger,
        response_context=ctx,
    )
    answer = result.final_answer if result else ""
    return {
        "model": SINGLE_BASELINE_MODEL,
        "answer": answer,
        "is_correct": utils.answers_match(answer, problem.get("correct_answer", "")),
    }


def process_problem(problem: dict[str, Any], output_dir: Path, logger) -> dict[str, Any]:
    """Run the full pipeline for a single problem and return its consolidated record."""
    pid = problem["id"]
    logger.info("=" * 70)
    logger.info("PROBLEM %s [%s / %s]", pid, problem["category"], problem["difficulty"])

    # Stage 0 + 0.5
    assessments = run_stage0(problem, logger)
    utils.save_json(assessments, config.STAGE_DIRS["stage0"] / f"{pid}.json")
    assignment = assign_roles(assessments, logger)

    # Stage 1
    solutions = run_stage1(problem, assignment, logger)
    utils.save_json(list(solutions.values()), config.STAGE_DIRS["stage1"] / f"{pid}.json")
    if len(solutions) < 2:
        logger.error("stage1 | fewer than 2 valid solutions for %s; aborting problem", pid)
        return _abort_record(problem, assignment, solutions)

    # Stage 2
    reviews = run_stage2(problem, assignment, solutions, logger)
    utils.save_json(reviews, config.STAGE_DIRS["stage2"] / f"{pid}.json")

    # Stage 3
    refinements = run_stage3(problem, assignment, solutions, reviews, logger)
    utils.save_json(list(refinements.values()), config.STAGE_DIRS["stage3"] / f"{pid}.json")

    # Stage 4
    judgment = run_stage4(problem, assignment, solutions, reviews, refinements, logger)
    utils.save_json(judgment, config.STAGE_DIRS["stage4"] / f"{pid}.json")

    # Baselines (derived without extra debate)
    single = run_single_baseline(problem, logger)
    original_answers = [s["final_answer"] for s in solutions.values()]
    refined_answers = [r["refined_final_answer"] for r in refinements.values()]
    vote_answer = _majority_answer(original_answers)
    voting = {
        "answer": vote_answer,
        "is_correct": utils.answers_match(vote_answer or "", problem.get("correct_answer", "")),
        "had_majority": vote_answer is not None,
    }

    # Full-system final answer = winner's refined answer.
    winner = judgment["winner"]
    final_answer = refinements[winner]["refined_final_answer"]
    is_correct = utils.answers_match(final_answer, problem.get("correct_answer", ""))

    # Derived metric flags.
    consensus = len({utils.normalize_answer(a) for a in refined_answers}) == 1 and len(refined_answers) == 3
    improvement_occurred = any(
        (not solutions[sid].get("_correct", False)) and refinements[sid].get("_correct", False)
        for sid in refinements
    )
    solvers_disagree = len({utils.normalize_answer(a) for a in refined_answers}) > 1

    record = {
        "problem_id": pid,
        "category": problem["category"],
        "difficulty": problem["difficulty"],
        "winner_solver": winner,
        "final_answer": final_answer,
        "correct_answer": problem.get("correct_answer", ""),
        "is_correct": is_correct,
        "judge_confidence": judgment["confidence"],
        "judge_fallback": judgment.get("_fallback", False),
        "consensus": consensus,
        "improvement_occurred": improvement_occurred,
        "solvers_disagree": solvers_disagree,
        "role_assignment": assignment,
        "baseline_single": single,
        "baseline_voting": voting,
        "solvers": {
            sid: {
                "model": solutions[sid]["model"],
                "original_answer": solutions[sid]["final_answer"],
                "original_correct": solutions[sid].get("_correct", False),
                "original_confidence": solutions[sid]["confidence"],
                "refined_answer": refinements[sid]["refined_final_answer"],
                "refined_correct": refinements[sid].get("_correct", False),
                "refined_confidence": refinements[sid]["confidence"],
            }
            for sid in solutions
        },
        "reviews_summary": [
            {
                "reviewer": r["reviewer_id"],
                "reviewed": r["solution_reviewed"],
                "assessment": r["overall_assessment"],
                "flagged_real_error": r.get("_flagged_real_error", False),
                "num_errors": len(r["evaluation"]["errors"]),
            }
            for r in reviews
        ],
    }

    # Save a full per-problem record for the evaluator.
    by_problem_dir = output_dir / "by_problem"
    utils.save_json(
        {
            "problem": problem,
            "assessments": assessments,
            "assignment": assignment,
            "solutions": solutions,
            "reviews": reviews,
            "refinements": refinements,
            "judgment": judgment,
            "record": record,
        },
        by_problem_dir / f"{pid}.json",
    )

    logger.info("DONE %s | final=%r correct=%s | single=%s voting=%s",
                pid, final_answer, is_correct, single["is_correct"], voting["is_correct"])
    return record


def _abort_record(problem, assignment, solutions) -> dict[str, Any]:
    return {
        "problem_id": problem["id"],
        "category": problem["category"],
        "difficulty": problem["difficulty"],
        "winner_solver": None,
        "final_answer": "",
        "correct_answer": problem.get("correct_answer", ""),
        "is_correct": False,
        "judge_confidence": 0.0,
        "judge_fallback": True,
        "consensus": False,
        "improvement_occurred": False,
        "solvers_disagree": False,
        "role_assignment": assignment,
        "baseline_single": {"model": SINGLE_BASELINE_MODEL, "answer": "", "is_correct": False},
        "baseline_voting": {"answer": None, "is_correct": False, "had_majority": False},
        "solvers": {},
        "reviews_summary": [],
        "aborted": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Multi-LLM Debate pipeline.")
    parser.add_argument("--problems", default=str(config.PROBLEMS_PATH),
                        help="Path to problems.json")
    parser.add_argument("--output", default=str(config.RESULTS_DIR),
                        help="Directory for result outputs")
    parser.add_argument("--problem-id", default=None,
                        help="Run only this problem id (e.g. prob_001)")
    args = parser.parse_args()

    config.ensure_directories()
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = utils.setup_logger(run_tag)
    logger.info("Backend=%s | free_mode=%s | provider=%s | seed=%d",
                config.LLM_BACKEND, config.FREE_MODEL_MODE,
                config.FREE_MODEL_PROVIDER, config.RANDOM_SEED)

    problems = utils.load_json(Path(args.problems))
    if args.problem_id:
        problems = [p for p in problems if p["id"] == args.problem_id]
        if not problems:
            logger.error("No problem with id %s", args.problem_id)
            sys.exit(1)

    output_dir = Path(args.output)
    records = []
    for problem in problems:
        try:
            records.append(process_problem(problem, output_dir, logger))
        except Exception as err:  # never let one problem kill the whole run
            utils.log_error(logger, "pipeline", "process_problem",
                            f"unhandled error on {problem['id']}: {err}")

    utils.save_json(records, output_dir / "final_answers.json")
    n_correct = sum(1 for r in records if r["is_correct"])
    logger.info("=" * 70)
    logger.info("RUN COMPLETE | %d problems | full-system accuracy = %d/%d = %.1f%%",
                len(records), n_correct, len(records),
                100.0 * n_correct / max(1, len(records)))
    logger.info("Results written to %s", output_dir / "final_answers.json")


if __name__ == "__main__":
    main()
