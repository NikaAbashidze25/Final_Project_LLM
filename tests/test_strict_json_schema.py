"""Unit tests for the pydantic -> strict-mode JSON Schema conversion used by
the OpenAI/Grok/Claude structured-output code paths. Pure function, no
network access or API keys required.
"""
from __future__ import annotations

from src import utils
from src.models.base_llm import strict_json_schema, supports_strict_mode


def _assert_strict(node: dict) -> None:
    """Recursively assert every object node is `additionalProperties: false`
    with every property listed as required (OpenAI strict-mode rules)."""
    if not isinstance(node, dict):
        return
    if "properties" in node:
        assert node.get("additionalProperties") is False
        assert set(node["required"]) == set(node["properties"].keys())
        for sub_schema in node["properties"].values():
            _assert_strict(sub_schema)
    if "items" in node:
        _assert_strict(node["items"])
    for combinator in ("anyOf", "oneOf", "allOf"):
        for sub_schema in node.get(combinator, []):
            _assert_strict(sub_schema)


def test_flat_schema_is_strict():
    schema = strict_json_schema(utils.RoleAssessment)
    _assert_strict(schema)


def test_nested_schema_is_strict_including_defs():
    # Solution nests SolutionStep via a $ref into $defs -- make sure the
    # nested definition is tightened too, not just the top-level object.
    schema = strict_json_schema(utils.Solution)
    _assert_strict(schema)
    for definition in schema.get("$defs", {}).values():
        _assert_strict(definition)


def test_deeply_nested_schema_is_strict():
    # Review nests ReviewEvaluation -> list[ReviewError] two levels deep.
    schema = strict_json_schema(utils.Review)
    _assert_strict(schema)
    for definition in schema.get("$defs", {}).values():
        _assert_strict(definition)


def test_all_stage_schemas_convert_without_error():
    for schema_cls in (
        utils.RoleAssessment, utils.Solution, utils.Review,
        utils.Refinement, utils.Judgment,
    ):
        result = strict_json_schema(schema_cls)
        assert isinstance(result, dict)
        assert result.get("additionalProperties") is False


def test_role_assessment_is_not_strict_mode_compatible():
    # confidence_by_role is a dict[str, X] field with runtime-only keys --
    # OpenAI/Grok strict mode can't represent an open-ended map, so callers
    # must detect this and fall back to plain JSON mode.
    assert supports_strict_mode(strict_json_schema(utils.RoleAssessment)) is False


def test_fixed_shape_schemas_are_strict_mode_compatible():
    # Solution/Review/Refinement/Judgment have no dict[str, X] fields, so
    # they should all be fully representable in strict mode.
    for schema_cls in (utils.Solution, utils.Review, utils.Refinement, utils.Judgment):
        assert supports_strict_mode(strict_json_schema(schema_cls)) is True
