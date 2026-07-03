"""Abstract LLM interface shared by every backend.

Every concrete client implements `generate()`, which takes a system prompt and a
user prompt and returns the raw text response. Real API clients ignore the
optional `response_context`; the offline simulator uses it to synthesize
deterministic, schema-valid responses without any network access.

Clients that support provider-native structured outputs (OpenAI/Grok's
`json_schema` strict mode, Gemini's `response_schema`, Anthropic's forced
tool-use) use the optional `schema` argument to constrain generation to the
pydantic model's shape, instead of relying solely on the JSON shape described
in the prompt text. `strict_json_schema()` is the single place that turns a
pydantic model into the "all fields required, no extra properties" JSON
Schema that strict-mode providers require.
"""
from __future__ import annotations

import abc
from typing import Any, Optional

from pydantic import BaseModel


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Convert a pydantic model to a JSON Schema suitable for strict-mode
    structured outputs (e.g. OpenAI's `json_schema` response format).

    Strict mode requires every object in the schema (including nested
    `$defs`) to set `additionalProperties: false` and list every one of its
    properties as `required` -- pydantic's own `model_json_schema()` doesn't
    do this by default, since it separately tracks which fields have
    defaults.
    """
    schema = model.model_json_schema()

    def _tighten(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
            for sub_schema in node["properties"].values():
                _tighten(sub_schema)
        if "items" in node:
            _tighten(node["items"])
        for combinator in ("anyOf", "oneOf", "allOf"):
            for sub_schema in node.get(combinator, []):
                _tighten(sub_schema)

    _tighten(schema)
    for definition in schema.get("$defs", {}).values():
        _tighten(definition)
    return schema


def _has_open_map(node: Any) -> bool:
    """True if `node` (or anything nested under it) is a true open-ended map
    -- i.e. a pydantic `dict[str, X]` field, which JSON-Schemas as an object
    with `additionalProperties` set to a sub-schema and no fixed `properties`
    list."""
    if not isinstance(node, dict):
        return False
    if "properties" not in node and isinstance(node.get("additionalProperties"), dict):
        return True
    if any(_has_open_map(sub) for sub in node.get("properties", {}).values()):
        return True
    if _has_open_map(node.get("items")):
        return True
    for combinator in ("anyOf", "oneOf", "allOf"):
        if any(_has_open_map(sub) for sub in node.get(combinator, [])):
            return True
    return False


def supports_strict_mode(schema: dict[str, Any]) -> bool:
    """Whether a schema from `strict_json_schema()` is actually usable in
    OpenAI/Grok strict mode.

    Strict mode requires every object to have a fixed, enumerable set of
    properties. A pydantic `dict[str, X]` field (used for e.g.
    `confidence_by_role`, whose keys are only known at runtime) has no fixed
    key set and is not representable in strict mode at any nesting level,
    including inside `$defs`. Callers should fall back to plain JSON mode
    for schemas where this returns False.
    """
    if _has_open_map(schema):
        return False
    return not any(_has_open_map(d) for d in schema.get("$defs", {}).values())


class BaseLLM(abc.ABC):
    """Common interface for all language-model backends."""

    def __init__(self, identity: str, model_name: str) -> None:
        #: Canonical identity used across the system (e.g. "gpt-4", "claude").
        self.identity = identity
        #: Concrete provider model name actually called (e.g. "gpt-4o-mini").
        self.model_name = model_name

    @abc.abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.7,
        response_context: Optional[dict[str, Any]] = None,
        schema: Optional[type[BaseModel]] = None,
    ) -> str:
        """Return the model's raw text response for the given prompts.

        `schema`, when given, is the pydantic model the response must match.
        Clients that support provider-native structured outputs should use
        it to constrain generation; clients that don't may simply ignore it
        and rely on the prompt text, since the caller always validates the
        parsed response against the same schema afterwards regardless.
        """

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(identity={self.identity!r}, model={self.model_name!r})"
