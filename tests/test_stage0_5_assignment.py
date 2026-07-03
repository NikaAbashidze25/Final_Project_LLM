from src import config, utils
from src.stages import stage0_5_assignment as s05
from src.stages import stage0_assessment as s0


def test_assignment_has_one_judge_three_distinct_solvers(problem, logger):
    assessments = s0.run_stage0(problem, logger)
    assignment = s05.assign_roles(assessments)

    assert set(assignment.keys()) == {"Solver_1", "Solver_2", "Solver_3", "Judge"}
    assert set(assignment.values()) == set(config.ALL_MODELS)


def test_assignment_is_deterministic(problem, logger):
    assessments = s0.run_stage0(problem, logger)

    first = s05.assign_roles(assessments)
    second = s05.assign_roles(assessments)

    assert first == second


def test_missing_assessments_fall_back_to_priority_order():
    # No valid self-assessments at all (e.g. all 4 models failed Stage 0).
    assessments = {identity: None for identity in config.ALL_MODELS}

    assignment = s05.assign_roles(assessments)

    assert assignment["Judge"] == config.MODEL_PRIORITY[0]
    assert set(assignment.values()) == set(config.ALL_MODELS)


def test_explicit_judge_preference_wins_the_judge_seat():
    # Give the lowest-priority model a clear, confident Judge preference; it
    # should still win the Judge seat over the priority-order tie-break.
    preferred = config.MODEL_PRIORITY[-1]
    assessments = {identity: None for identity in config.ALL_MODELS}
    assessments[preferred] = utils.RoleAssessment(
        role_preferences=["Judge", "Solver"],
        confidence_by_role={"Solver": 0.5, "Judge": 0.99},
        reasoning="test fixture: strong judge preference",
    )

    assignment = s05.assign_roles(assessments)

    assert assignment["Judge"] == preferred
