from src.stages import stage0_5_assignment as s05
from src.stages import stage0_assessment as s0
from src.stages import stage1_solve as s1
from src.stages import stage2_review as s2
from src.stages import stage3_refine as s3
from src.stages import stage4_judge as s4


def _refined(problem, logger):
    assessments = s0.run_stage0(problem, logger)
    assignment = s05.assign_roles(assessments)
    solutions = s1.run_stage1(problem, assignment, logger)
    reviews = s2.run_stage2(problem, assignment, solutions, logger)
    refinements = s3.run_stage3(problem, assignment, solutions, reviews, logger)
    return assignment, solutions, reviews, refinements


def test_judge_picks_a_present_solver_with_full_ranking(problem, logger):
    assignment, solutions, reviews, refinements = _refined(problem, logger)

    judgment = s4.run_stage4(problem, assignment, solutions, reviews, refinements, logger)

    assert judgment.winner in solutions
    assert set(judgment.ranking.keys()) == set(solutions.keys())
    assert sorted(judgment.ranking.values()) == list(range(1, len(solutions) + 1))
    assert 0.0 <= judgment.confidence <= 1.0


def test_fallback_used_when_judge_call_fails(problem, logger, monkeypatch):
    """If the Judge fails after retries, README requires falling back to the
    refined solution with the highest self-reported confidence."""
    assignment, solutions, reviews, refinements = _refined(problem, logger)
    monkeypatch.setattr(s4.utils, "call_llm_validated", lambda *args, **kwargs: None)

    judgment = s4.run_stage4(problem, assignment, solutions, reviews, refinements, logger)

    expected_winner = max(
        solutions.keys(),
        key=lambda sid: refinements[sid].confidence if refinements[sid] else solutions[sid].confidence,
    )
    assert judgment.winner == expected_winner
    assert "fallback" in judgment.notes.lower()
