"""Evaluation & analysis: metrics, baseline comparisons, and all 7 required plots.

Loads the consolidated `final_answers.json` written by the pipeline together
with the per-problem detail files, then computes accuracy, improvement rate,
consensus rate, confidence calibration, and per-category/model breakdowns.
All 7 plots are saved as PNG files under the configured plots directory.

Usage:
    python src/evaluator.py --results data/results/ --output plots/
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless / file-only output, no display needed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config, utils  # type: ignore
else:
    from . import config, utils

sns.set_theme(style="whitegrid")

# Short display labels for category names used in plots.
CATEGORY_LABELS = {
    "mathematical_reasoning": "Math",
    "physics_reasoning": "Physics",
    "logic_puzzle": "Logic",
    "game_theory": "Game Theory",
}

# Consistent colour palette across every plot that compares the three systems.
SYSTEM_COLORS = {
    "Single-LLM": "#888888",
    "Voting": "#4C8BE2",
    "Full Debate": "#2BB673",
}


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def load_records(results_dir: Path) -> list[dict]:
    """Load the top-level summary records written by the pipeline."""
    return utils.load_json(results_dir / "final_answers.json")


def load_details(results_dir: Path) -> list[dict]:
    """Load the per-problem detail files (one JSON per problem)."""
    detail_dir = results_dir / "by_problem"
    details = []
    for path in sorted(detail_dir.glob("*.json")):
        details.append(utils.load_json(path))
    return details


# ─────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────

def compute_metrics(records: list[dict], details: list[dict]) -> dict[str, Any]:
    """Compute all metrics from pipeline output. Returns a metrics dict."""
    n = max(len(records), 1)  # guard against division by zero on empty runs

    full_correct = sum(1 for r in records if r["is_correct"])
    single_correct = sum(1 for r in records if r["baseline_single"]["is_correct"])
    voting_correct = sum(1 for r in records if r["baseline_voting"]["is_correct"])

    improvement = sum(1 for r in records if r["improvement_occurred"])
    consensus = sum(1 for r in records if r["consensus"])

    # Judge accuracy is only meaningful on problems where solvers disagreed.
    disagree = [r for r in records if r["solvers_disagree"]]
    judge_correct_on_disagree = sum(1 for r in disagree if r["is_correct"])
    judge_accuracy = judge_correct_on_disagree / len(disagree) if disagree else float("nan")

    metrics: dict[str, Any] = {
        "num_problems": len(records),
        "overall_accuracy_full": full_correct / n,
        "overall_accuracy_single": single_correct / n,
        "overall_accuracy_voting": voting_correct / n,
        "improvement_rate": improvement / n,
        "consensus_rate": consensus / n,
        "judge_accuracy_on_disagreement": judge_accuracy,
        "num_disagreements": len(disagree),
        "per_category": _per_category(records),
        "per_model": _per_model(details),
        "calibration": _calibration_points(details),
        "improvement_breakdown": _improvement_breakdown(details),
        "judge_acc_agree_vs_disagree": _judge_acc_split(records),
    }
    return metrics


def _per_category(records: list[dict]) -> dict[str, dict]:
    """Aggregate accuracy and consensus counts per problem category."""
    cats: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "full": 0, "single": 0, "voting": 0, "consensus": 0}
    )
    for r in records:
        c = cats[r["category"]]
        c["n"] += 1
        c["full"] += int(r["is_correct"])
        c["single"] += int(r["baseline_single"]["is_correct"])
        c["voting"] += int(r["baseline_voting"]["is_correct"])
        c["consensus"] += int(r["consensus"])
    return {k: dict(v) for k, v in cats.items()}


def _per_model(details: list[dict]) -> dict[str, dict]:
    """Per-model accuracy as solver, as judge, and peer-review recall."""
    stats: dict[str, dict] = defaultdict(lambda: {
        "solver_total": 0, "solver_correct": 0,
        "judge_total": 0, "judge_correct": 0,
        "reviews_of_wrong": 0, "reviews_flagged_real": 0,
    })
    for d in details:
        rec = d["record"]
        # Solver accuracy measured on the *refined* final answer.
        for sid, sdata in rec.get("solvers", {}).items():
            m = sdata["model"]
            stats[m]["solver_total"] += 1
            stats[m]["solver_correct"] += int(sdata["refined_correct"])

        # Judge accuracy: did this model's judgment pick the correct answer?
        judge_model = rec["role_assignment"]["Judge"]
        stats[judge_model]["judge_total"] += 1
        stats[judge_model]["judge_correct"] += int(rec["is_correct"])

        # Review recall: of all incorrect solutions, how many did each reviewer flag?
        id_to_model = {sid: sdata["model"] for sid, sdata in rec.get("solvers", {}).items()}
        solver_original_correct = {
            sid: sdata["original_correct"]
            for sid, sdata in rec.get("solvers", {}).items()
        }
        for rv in rec.get("reviews_summary", []):
            reviewer_model = id_to_model.get(rv["reviewer"])
            if reviewer_model is None:
                continue
            reviewed_was_wrong = not solver_original_correct.get(rv["reviewed"], True)
            if reviewed_was_wrong:
                stats[reviewer_model]["reviews_of_wrong"] += 1
                if rv["flagged_real_error"]:
                    stats[reviewer_model]["reviews_flagged_real"] += 1

    out: dict[str, dict] = {}
    for m, s in stats.items():
        out[m] = {
            **s,
            "solver_accuracy": (
                s["solver_correct"] / s["solver_total"] if s["solver_total"] else float("nan")
            ),
            "judge_accuracy": (
                s["judge_correct"] / s["judge_total"] if s["judge_total"] else float("nan")
            ),
            "review_recall": (
                s["reviews_flagged_real"] / s["reviews_of_wrong"]
                if s["reviews_of_wrong"] else float("nan")
            ),
        }
    return out


def _calibration_points(details: list[dict]) -> list[dict]:
    """Collect (confidence, correctness) pairs for all solver answers."""
    pts = []
    for d in details:
        for sid, sdata in d["record"].get("solvers", {}).items():
            pts.append({"confidence": sdata["original_confidence"],
                        "correct": int(sdata["original_correct"])})
            pts.append({"confidence": sdata["refined_confidence"],
                        "correct": int(sdata["refined_correct"])})
    return pts


def _improvement_breakdown(details: list[dict]) -> dict[str, int]:
    """Count how many problems were helped, hurt, or unchanged by refinement."""
    out = {"helped": 0, "hurt": 0, "no_effect": 0}
    for d in details:
        solvers = d["record"].get("solvers", {})
        if not solvers:
            continue
        before = sum(int(s["original_correct"]) for s in solvers.values())
        after = sum(int(s["refined_correct"]) for s in solvers.values())
        if after > before:
            out["helped"] += 1
        elif after < before:
            out["hurt"] += 1
        else:
            out["no_effect"] += 1
    return out


def _judge_acc_split(records: list[dict]) -> dict[str, float]:
    """Final-answer accuracy split by whether solvers agreed or disagreed."""
    agree = [r for r in records if not r["solvers_disagree"]]
    disagree = [r for r in records if r["solvers_disagree"]]

    def acc(rs: list[dict]) -> float:
        return sum(int(r["is_correct"]) for r in rs) / len(rs) if rs else float("nan")

    return {
        "agree_accuracy": acc(agree),
        "disagree_accuracy": acc(disagree),
        "n_agree": len(agree),
        "n_disagree": len(disagree),
    }


# ─────────────────────────────────────────────────────────────
# Plot 1 — Overall accuracy comparison
# ─────────────────────────────────────────────────────────────

def plot_overall_accuracy(metrics: dict, out: Path) -> None:
    """Bar chart: Single-LLM vs Voting vs Full Debate overall accuracy."""
    systems = ["Single-LLM", "Voting", "Full Debate"]
    vals = [
        metrics["overall_accuracy_single"],
        metrics["overall_accuracy_voting"],
        metrics["overall_accuracy_full"],
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(systems, [v * 100 for v in vals],
                  color=[SYSTEM_COLORS[s] for s in systems])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Overall Accuracy: Baselines vs. Full Debate System")
    ax.set_ylim(0, 100)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1.5, f"{v * 100:.1f}%",
                ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "overall_accuracy.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Plot 2 — Per-category accuracy (grouped bars)
# ─────────────────────────────────────────────────────────────

def plot_per_category(metrics: dict, out: Path) -> None:
    """Grouped bar chart: accuracy per category for each of the three systems."""
    cats = list(metrics["per_category"].keys())
    labels = [CATEGORY_LABELS.get(c, c) for c in cats]
    systems = [("single", "Single-LLM"), ("voting", "Voting"), ("full", "Full Debate")]
    x = np.arange(len(cats))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (key, label) in enumerate(systems):
        vals = [
            100 * metrics["per_category"][c][key] / metrics["per_category"][c]["n"]
            for c in cats
        ]
        ax.bar(x + (i - 1) * width, vals, width, label=label, color=SYSTEM_COLORS[label])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Per-Category Accuracy by System")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "per_category_accuracy.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Plot 3 — Effect of refinement across problems
# ─────────────────────────────────────────────────────────────

def plot_improvement_rate(metrics: dict, out: Path) -> None:
    """Bar chart showing how many problems were helped/unchanged/hurt by refinement."""
    b = metrics["improvement_breakdown"]
    labels = ["Helped", "No effect", "Hurt"]
    vals = [b["helped"], b["no_effect"], b["hurt"]]
    colors = ["#2BB673", "#AAAAAA", "#E2574C"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("Number of problems")
    ax.set_title("Effect of Refinement Across Problems")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.2, str(v),
                ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "improvement_rate.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Plot 4 — Solver consensus rate per category
# ─────────────────────────────────────────────────────────────

def plot_consensus_rate(metrics: dict, out: Path) -> None:
    """Bar chart: fraction of problems where all 3 solvers agreed after refinement."""
    cats = list(metrics["per_category"].keys())
    labels = [CATEGORY_LABELS.get(c, c) for c in cats]
    vals = [
        100 * metrics["per_category"][c]["consensus"] / metrics["per_category"][c]["n"]
        for c in cats
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, vals, color="#7E57C2")
    ax.set_ylabel("Consensus rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Solver Consensus Rate by Category")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.0f}%",
                ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "consensus_rate.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Plot 5 — Judge / system accuracy vs solver disagreement
# ─────────────────────────────────────────────────────────────

def plot_judge_accuracy(metrics: dict, out: Path) -> None:
    """Bar chart comparing final-answer accuracy when solvers agree vs disagree."""
    s = metrics["judge_acc_agree_vs_disagree"]
    labels = [
        f"Solvers agree\n(n={s['n_agree']})",
        f"Solvers disagree\n(n={s['n_disagree']})",
    ]
    vals = [
        100 * (s["agree_accuracy"] if not np.isnan(s["agree_accuracy"]) else 0),
        100 * (s["disagree_accuracy"] if not np.isnan(s["disagree_accuracy"]) else 0),
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, vals, color=["#4C8BE2", "#E2914C"])
    ax.set_ylabel("Final-answer accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Judge / System Accuracy vs. Solver Disagreement")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.0f}%",
                ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "judge_accuracy.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Plot 6 — Per-model accuracy as solver and as judge
# ─────────────────────────────────────────────────────────────

def plot_per_model_accuracy(metrics: dict, out: Path) -> None:
    """Grouped bar chart: each model's accuracy when acting as solver vs judge."""
    models = sorted(metrics["per_model"].keys())
    solver_acc = [
        100 * metrics["per_model"][m]["solver_accuracy"]
        if not np.isnan(metrics["per_model"][m]["solver_accuracy"]) else 0
        for m in models
    ]
    judge_acc = [
        100 * metrics["per_model"][m]["judge_accuracy"]
        if not np.isnan(metrics["per_model"][m]["judge_accuracy"]) else 0
        for m in models
    ]
    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, solver_acc, width, label="As Solver", color="#2BB673")
    ax.bar(x + width / 2, judge_acc, width, label="As Judge", color="#E2574C")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Per-Model Accuracy (Solver vs. Judge)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "per_model_accuracy.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Plot 7 — Confidence calibration
# ─────────────────────────────────────────────────────────────

def plot_confidence_calibration(metrics: dict, out: Path) -> None:
    """Scatter + binned-accuracy curve showing how well confidence predicts correctness."""
    pts = metrics["calibration"]
    conf = np.array([p["confidence"] for p in pts])
    correct = np.array([p["correct"] for p in pts])
    fig, ax = plt.subplots(figsize=(7, 6))

    # Scatter with jitter so overlapping points are visible.
    jitter = (np.random.RandomState(0).rand(len(correct)) - 0.5) * 0.05
    ax.scatter(conf, correct + jitter, alpha=0.35, s=25, color="#4C8BE2",
               label="Solver answers")

    # Binned observed accuracy curve to visualise calibration quality.
    bins = np.linspace(0, 1, 6)
    idx = np.digitize(conf, bins) - 1
    xs, ys = [], []
    for b in range(len(bins) - 1):
        mask = idx == b
        if mask.sum() > 0:
            xs.append((bins[b] + bins[b + 1]) / 2)
            ys.append(correct[mask].mean())
    ax.plot(xs, ys, "o-", color="#E2574C", linewidth=2, label="Observed accuracy (binned)")

    # Perfect-calibration diagonal reference.
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    ax.set_xlabel("Reported confidence")
    ax.set_ylabel("Actual correctness")
    ax.set_title("Confidence Calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc="center left")
    fig.tight_layout()
    fig.savefig(out / "confidence_calibration.png", dpi=150)
    plt.close(fig)


ALL_PLOTS = [
    plot_overall_accuracy,
    plot_per_category,
    plot_improvement_rate,
    plot_consensus_rate,
    plot_judge_accuracy,
    plot_per_model_accuracy,
    plot_confidence_calibration,
]


# ─────────────────────────────────────────────────────────────
# Human-readable summary
# ─────────────────────────────────────────────────────────────

def print_summary(metrics: dict) -> None:
    """Print a compact evaluation summary to stdout."""
    print("\n" + "=" * 60)
    print("MULTI-LLM DEBATE SYSTEM — EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Problems evaluated         : {metrics['num_problems']}")
    print(f"Single-LLM accuracy        : {metrics['overall_accuracy_single'] * 100:5.1f}%")
    print(f"Voting baseline accuracy   : {metrics['overall_accuracy_voting'] * 100:5.1f}%")
    print(f"Full debate accuracy       : {metrics['overall_accuracy_full'] * 100:5.1f}%")
    print(f"Improvement rate           : {metrics['improvement_rate'] * 100:5.1f}%")
    print(f"Consensus rate             : {metrics['consensus_rate'] * 100:5.1f}%")
    ja = metrics["judge_accuracy_on_disagreement"]
    ja_s = f"{ja * 100:5.1f}%" if not np.isnan(ja) else "  n/a"
    print(f"Judge accuracy (disagree)  : {ja_s} (n={metrics['num_disagreements']})")
    print("-" * 60)
    print("Per-model breakdown:")
    for m, s in sorted(metrics["per_model"].items()):
        sa = s["solver_accuracy"]
        ja = s["judge_accuracy"]
        rr = s["review_recall"]

        def f(x: float) -> str:
            return f"{x * 100:5.1f}%" if not np.isnan(x) else "  n/a"

        print(f"  {m:8s} | solver {f(sa)} | judge {f(ja)} | review-recall {f(rr)}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate results and generate plots.")
    parser.add_argument("--results", default=str(config.RESULTS_DIR),
                        help="Directory containing final_answers.json and by_problem/")
    parser.add_argument("--output", default=str(config.PLOTS_DIR),
                        help="Directory to write PNG plots into")
    args = parser.parse_args()

    results_dir = Path(args.results)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    records = load_records(results_dir)
    details = load_details(results_dir)
    metrics = compute_metrics(records, details)

    utils.save_json(metrics, results_dir / "metrics.json")

    for plot_fn in ALL_PLOTS:
        plot_fn(metrics, out)
        print(f"  generated {plot_fn.__name__}")

    print_summary(metrics)
    print(f"Plots saved to    : {out}")
    print(f"Metrics saved to  : {results_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
