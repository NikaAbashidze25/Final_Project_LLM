"""Abstract LLM interface shared by every backend.

Every concrete client implements `generate()`, which takes a system prompt and a
user prompt and returns the raw text response. Real API clients ignore the
optional `response_context`; the offline simulator uses it to synthesize
deterministic, schema-valid responses without any network access.
"""
from __future__ import annotations

import abc
from typing import Any, Optional


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
    ) -> str:
        """Return the model's raw text response for the given prompts."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(identity={self.identity!r}, model={self.model_name!r})"
