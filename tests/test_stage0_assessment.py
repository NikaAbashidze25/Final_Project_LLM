from src import config
from src.stages import stage0_assessment as s0


def test_all_models_produce_valid_assessment(problem, logger):
    assessments = s0.run_stage0(problem, logger)

    assert set(assessments.keys()) == set(config.ALL_MODELS)
    for identity, assessment in assessments.items():
        assert assessment is not None, f"{identity} failed to produce an assessment"
        assert set(assessment.role_preferences) == {"Solver", "Judge"}
        assert 0.0 <= assessment.confidence_by_role["Solver"] <= 1.0
        assert 0.0 <= assessment.confidence_by_role["Judge"] <= 1.0


def test_deterministic_across_runs(problem, logger):
    first = s0.run_stage0(problem, logger)
    second = s0.run_stage0(problem, logger)

    for identity in config.ALL_MODELS:
        assert first[identity].model_dump() == second[identity].model_dump()
