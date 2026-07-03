from src.stages import stage0_5_assignment as s05
from src.stages import stage0_assessment as s0
from src.stages import stage1_solve as s1
from src.stages import stage2_review as s2
from src.stages import stage3_refine as s3


def _reviewed(problem, logger):
    assessments = s0.run_stage0(problem, logger)
    assignment = s05.assign_roles(assessments, logger)
    solutions = s1.run_stage1(problem, assignment, logger)
    reviews = s2.run_stage2(problem, assignment, solutions, logger)
    return assignment, solutions, reviews


def test_refinement_addresses_every_received_critique(problem, logger):
    assignment, solutions, reviews = _reviewed(problem, logger)

    refinements = s3.run_stage3(problem, assignment, solutions, reviews, logger)

    assert set(refinements.keys()) == set(solutions.keys())
    for solver_id, refinement in refinements.items():
        received = [r for r in reviews if r["solution_reviewed"] == solver_id]
        assert len(refinement["critique_responses"]) == len(received)
        assert len(refinement["refined_solution_steps"]) >= 1
        assert 0.0 <= refinement["confidence"] <= 1.0


def test_excluded_solver_has_no_refinement(problem, logger):
    assignment, solutions, reviews = _reviewed(problem, logger)
    solutions = dict(solutions)
    del solutions["solver_1"]
    reviews = s2.run_stage2(problem, assignment, solutions, logger)

    refinements = s3.run_stage3(problem, assignment, solutions, reviews, logger)

    assert "solver_1" not in refinements
    assert set(refinements.keys()) == {"solver_2", "solver_3"}
