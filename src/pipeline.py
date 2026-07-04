"""End-to-end orchestrator for the Multi-LLM Debate System.

Runs all stages for one or more problems, persists every intermediate output to
disk (so interrupted runs can be resumed/inspected), derives the two baselines
(single-LLM and majority voting) from the same data, and writes a consolidated
`final_answers.json`.

Usage:
    python src/pipeline.py --problems data/problems.json --output data/results/
    python src/pipeline.py --problem-id prob_001 --problems data/problems.json --output data/results/
    python src/pipeline.py --start-id prob_010 --end-id prob_015 --problems data/problems.json --output data/results/

`--start-id`/`--end-id` resume a run over a range of problems (e.g. after
hitting an API rate limit) without rerunning problems already completed.
Results are merged into the existing `final_answers.json` rather than
overwriting it, so problems outside the range are preserved.
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


def process_problem(
    problem: dict[str, Any], output_dir: Path, logger, reuse_stages: bool = False,
) -> dict[str, Any]:
    """Run the full pipeline for a single problem and return its consolidated record.

    If `reuse_stages` is set and stage0-4 output files already exist for this
    problem, they are loaded from disk instead of re-querying the LLMs (only the
    single-model baseline is still called fresh). This makes it cheap to rebuild
    `by_problem/`/`final_answers.json` after a crash in the final save step, or to
    re-score existing results after a change to answer-matching logic, without
    burning API calls on stages that already succeeded.
    """
    pid = problem["id"]
    logger.info("=" * 70)
    logger.info("PROBLEM %s [%s / %s]", pid, problem["category"], problem["difficulty"])

    stage_files = {k: config.STAGE_DIRS[k] / f"{pid}.json" for k in ("stage0", "stage1", "stage2", "stage3", "stage4")}

    if reuse_stages and all(p.exists() for p in stage_files.values()):
        logger.info("stage0-4 | reusing existing results for %s (no LLM calls)", pid)
        assessments = utils.load_json(stage_files["stage0"])
        assignment = assign_roles(assessments, logger)
        solutions = {s["solver_id"]: s for s in utils.load_json(stage_files["stage1"])}
        reviews = utils.load_json(stage_files["stage2"])
        refinements = {r["solver_id"]: r for r in utils.load_json(stage_files["stage3"])}
        judgment = utils.load_json(stage_files["stage4"])
        # Re-derive correctness with the CURRENT matching logic, in case it has
        # changed since these files were written (e.g. an answers_match fix) --
        # don't trust the "_correct" flags baked in at save time.
        for s in solutions.values():
            s["_correct"] = utils.answers_match(s["final_answer"], problem.get("correct_answer", ""))
        for r in refinements.values():
            r["_correct"] = utils.answers_match(r["refined_final_answer"], problem.get("correct_answer", ""))
        if len(solutions) < 2:
            logger.error("stage1 | fewer than 2 valid solutions for %s; aborting problem", pid)
            return _abort_record(problem, assignment, solutions)
    else:
        # Stage 0 + 0.5
        assessments = run_stage0(problem, logger)
        utils.save_json(assessments, stage_files["stage0"])
        assignment = assign_roles(assessments, logger)

        # Stage 1
        solutions = run_stage1(problem, assignment, logger)
        utils.save_json(list(solutions.values()), stage_files["stage1"])
        if len(solutions) < 2:
            logger.error("stage1 | fewer than 2 valid solutions for %s; aborting problem", pid)
            return _abort_record(problem, assignment, solutions)

        # Stage 2
        reviews = run_stage2(problem, assignment, solutions, logger)
        utils.save_json(reviews, stage_files["stage2"])

        # Stage 3
        refinements = run_stage3(problem, assignment, solutions, reviews, logger)
        utils.save_json(list(refinements.values()), stage_files["stage3"])

        # Stage 4
        judgment = run_stage4(problem, assignment, solutions, reviews, refinements, logger)
        utils.save_json(judgment, stage_files["stage4"])

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
    parser.add_argument("--start-id", default=None,
                        help="Resume from this problem id, inclusive (e.g. prob_010). "
                             "Combine with --end-id to run a range; useful for continuing "
                             "after a rate limit without rerunning earlier problems.")
    parser.add_argument("--end-id", default=None,
                        help="Stop at this problem id, inclusive (e.g. prob_015). "
                             "Defaults to the last problem in --problems if omitted.")
    parser.add_argument("--reuse-stages", action="store_true",
                        help="If stage0-4 results already exist on disk for a problem, load "
                             "them instead of calling the LLMs again (only the single-model "
                             "baseline is re-run). Use this to cheaply rebuild by_problem/ and "
                             "final_answers.json after a crash in the final save step, or to "
                             "re-score existing results after a change to answer-matching logic "
                             "-- without spending API calls on stages that already succeeded.")
    args = parser.parse_args()

    config.ensure_directories()
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = utils.setup_logger(run_tag)
    logger.info("Backend=%s | free_mode=%s | provider=%s | seed=%d",
                config.LLM_BACKEND, config.FREE_MODEL_MODE,
                config.FREE_MODEL_PROVIDER, config.RANDOM_SEED)

    # Validate --output up front: a malformed path (e.g. an accidental paste
    # that embeds an absolute path mid-string) should fail immediately with a
    # clear message, not after burning a full round of API calls per problem
    # only to crash on the final save.
    output_dir = Path(args.output)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "by_problem").mkdir(parents=True, exist_ok=True)
    except OSError as err:
        logger.error("Invalid --output path %r: %s", args.output, err)
        sys.exit(1)

    all_problems = utils.load_json(Path(args.problems))
    all_ids = [p["id"] for p in all_problems]
    problems = all_problems

    if args.problem_id:
        problems = [p for p in problems if p["id"] == args.problem_id]
        if not problems:
            logger.error("No problem with id %s", args.problem_id)
            sys.exit(1)
    elif args.start_id or args.end_id:
        start_idx = all_ids.index(args.start_id) if args.start_id else 0
        if args.start_id and args.start_id not in all_ids:
            logger.error("No problem with id %s", args.start_id)
            sys.exit(1)
        end_idx = all_ids.index(args.end_id) if args.end_id else len(all_problems) - 1
        if args.end_id and args.end_id not in all_ids:
            logger.error("No problem with id %s", args.end_id)
            sys.exit(1)
        problems = all_problems[start_idx:end_idx + 1]
        if not problems:
            logger.error("Empty problem range: %s to %s", args.start_id, args.end_id)
            sys.exit(1)
        logger.info("Resuming run over %d problem(s): %s .. %s",
                    len(problems), problems[0]["id"], problems[-1]["id"])

    records = []
    for problem in problems:
        try:
            records.append(process_problem(problem, output_dir, logger, reuse_stages=args.reuse_stages))
        except Exception as err:  # never let one problem kill the whole run
            utils.log_error(logger, "pipeline", "process_problem",
                            f"unhandled error on {problem['id']}: {err}")

    # Merge into any existing final_answers.json instead of overwriting it, so
    # a partial/resumed run doesn't wipe out results for problems outside this range.
    final_path = output_dir / "final_answers.json"
    existing_records: list[dict] = []
    if final_path.exists():
        try:
            existing_records = utils.load_json(final_path)
        except Exception:
            existing_records = []
    by_id = {r["problem_id"]: r for r in existing_records}
    by_id.update({r["problem_id"]: r for r in records})
    merged_records = [by_id[pid] for pid in all_ids if pid in by_id]
    utils.save_json(merged_records, final_path)

    n_correct = sum(1 for r in records if r["is_correct"])
    n_total_correct = sum(1 for r in merged_records if r["is_correct"])
    logger.info("=" * 70)
    logger.info("RUN COMPLETE | %d problem(s) this run | accuracy = %d/%d = %.1f%%",
                len(records), n_correct, len(records),
                100.0 * n_correct / max(1, len(records)))
    logger.info("OVERALL | %d/%d problems recorded | accuracy = %d/%d = %.1f%%",
                len(merged_records), len(all_ids), n_total_correct, len(merged_records),
                100.0 * n_total_correct / max(1, len(merged_records)))
    logger.info("Results written to %s", final_path)


if __name__ == "__main__":
    main()
