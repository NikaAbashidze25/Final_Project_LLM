from src import config
from src.stages import stage0_5_assignment as s05
from src.stages import stage0_assessment as s0


def test_assignment_has_one_judge_three_distinct_solvers(problem, logger):
    assessments = s0.run_stage0(problem, logger)
    assignment = s05.assign_roles(assessments, logger)

    assert set(assignment.keys()) == {"Solver_1", "Solver_2", "Solver_3", "Judge"}
    assert set(assignment.values()) == set(config.ALL_MODELS)


def test_assignment_is_deterministic(problem, logger):
    assessments = s0.run_stage0(problem, logger)

    first = s05.assign_roles(assessments, logger)
    second = s05.assign_roles(assessments, logger)

    assert first == second


def test_neutral_assessments_fall_back_to_priority_order(logger):
    # All four models equally undecided (e.g. every Stage 0 call failed and
    # used the fallback assessment) -- ties must break by MODEL_PRIORITY.
    neutral = {
        identity: {
            "role_preferences": ["Solver", "Judge"],
            "confidence_by_role": {"Solver": 0.5, "Judge": 0.5},
            "reasoning": "neutral",
        }
        for identity in config.ALL_MODELS
    }

    assignment = s05.assign_roles(neutral, logger)

    assert assignment["Judge"] == config.MODEL_PRIORITY[0]
    assert set(assignment.values()) == set(config.ALL_MODELS)


def test_explicit_judge_preference_wins_the_judge_seat(logger):
    # Give the lowest-priority model a clear, confident Judge preference; it
    # should still win the Judge seat over the priority-order tie-break.
    preferred = config.MODEL_PRIORITY[-1]
    assessments = {
        identity: {
            "role_preferences": ["Solver", "Judge"],
            "confidence_by_role": {"Solver": 0.5, "Judge": 0.5},
            "reasoning": "neutral",
        }
        for identity in config.ALL_MODELS
    }
    assessments[preferred] = {
        "role_preferences": ["Judge", "Solver"],
        "confidence_by_role": {"Solver": 0.5, "Judge": 0.99},
        "reasoning": "test fixture: strong judge preference",
    }

    assignment = s05.assign_roles(assessments, logger)

    assert assignment["Judge"] == preferred
