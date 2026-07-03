from src.stages import stage0_5_assignment as s05
from src.stages import stage0_assessment as s0
from src.stages import stage1_solve as s1
from src.stages import stage2_review as s2


def _solved(problem, logger):
    assessments = s0.run_stage0(problem, logger)
    assignment = s05.assign_roles(assessments, logger)
    solutions = s1.run_stage1(problem, assignment, logger)
    return assignment, solutions


def test_six_reviews_when_all_solvers_survive(problem, logger):
    assignment, solutions = _solved(problem, logger)

    reviews = s2.run_stage2(problem, assignment, solutions, logger)

    assert len(reviews) == 6


def test_each_solver_receives_exactly_two_reviews(problem, logger):
    assignment, solutions = _solved(problem, logger)
    reviews = s2.run_stage2(problem, assignment, solutions, logger)

    for solver_id in solutions:
        received = [r for r in reviews if r["solution_reviewed"] == solver_id]
        assert len(received) == 2
        reviewers = {r["reviewer_id"] for r in received}
        assert reviewers == set(solutions.keys()) - {solver_id}


def test_excluded_solver_is_skipped_as_reviewer_and_target(problem, logger):
    assignment, solutions = _solved(problem, logger)
    solutions = dict(solutions)
    del solutions["solver_1"]  # simulate a Stage 1 failure

    reviews = s2.run_stage2(problem, assignment, solutions, logger)

    assert len(reviews) == 2  # only the solver_2 <-> solver_3 pair remains
    assert all(r["reviewer_id"] != "solver_1" and r["solution_reviewed"] != "solver_1"
               for r in reviews)
