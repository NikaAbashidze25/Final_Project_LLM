from src.models import get_model
from src.stages import stage0_5_assignment as s05
from src.stages import stage0_assessment as s0
from src.stages import stage1_solve as s1


def _assignment(problem, logger) -> dict:
    assessments = s0.run_stage0(problem, logger)
    return s05.assign_roles(assessments, logger)


def test_three_solutions_produced(problem, logger):
    assignment = _assignment(problem, logger)

    solutions = s1.run_stage1(problem, assignment, logger)

    assert set(solutions.keys()) == {"solver_1", "solver_2", "solver_3"}
    for solver_id, solution in solutions.items():
        assert solution["solver_id"] == solver_id
        assert solution["model"] == assignment[f"Solver_{solver_id[-1]}"]
        assert len(solution["solution_steps"]) >= 1
        assert 0.0 <= solution["confidence"] <= 1.0
        assert solution["final_answer"]


def test_failed_solver_is_excluded_not_raised(problem, logger, monkeypatch):
    """A Solver whose provider call always errors must be logged and excluded,
    not crash the stage (README error-handling requirement)."""
    assignment = _assignment(problem, logger)
    failing_identity = assignment["Solver_2"]
    failing_model = get_model(failing_identity)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated provider outage")

    monkeypatch.setattr(failing_model, "generate", boom)

    solutions = s1.run_stage1(problem, assignment, logger)

    assert "solver_2" not in solutions
    assert "solver_1" in solutions
    assert "solver_3" in solutions
