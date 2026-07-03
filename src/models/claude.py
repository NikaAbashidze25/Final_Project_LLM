"""Anthropic (Claude) client."""
from __future__ import annotations

import json
import time
from typing import Any, Optional

from pydantic import BaseModel

from .base_llm import BaseLLM, strict_json_schema


class ClaudeLLM(BaseLLM):
    def __init__(self, identity: str, model_name: str, api_key: str) -> None:
        super().__init__(identity, model_name)
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for the Claude backend.")
        import anthropic  # lazy import

        self._client = anthropic.Anthropic(api_key=api_key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.7,
        response_context: Optional[dict[str, Any]] = None,
        schema: Optional[type[BaseModel]] = None,
    ) -> str:
        # Anthropic has no `response_format`-style structured output; the
        # standard way to constrain generation to a schema is to give the
        # model exactly one tool (shaped like the schema) and force its use,
        # so `tool_use.input` comes back already matching it.
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": 4096,
            "temperature": temperature,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if schema is not None:
            kwargs["system"] = system_prompt
            kwargs["tools"] = [{
                "name": schema.__name__,
                "description": f"Record the {schema.__name__} result.",
                "input_schema": strict_json_schema(schema),
            }]
            kwargs["tool_choice"] = {"type": "tool", "name": schema.__name__}
        else:
            kwargs["system"] = system_prompt + "\nReturn ONLY valid JSON."

        delay = 2.0
        last_err: Optional[Exception] = None
        for _ in range(5):
            try:
                resp = self._client.messages.create(**kwargs)
                if schema is not None:
                    for block in resp.content:
                        if getattr(block, "type", "") == "tool_use":
                            return json.dumps(block.input)
                    return ""
                return "".join(
                    block.text for block in resp.content if getattr(block, "type", "") == "text"
                )
            except Exception as err:
                last_err = err
                if "429" in str(err) or "rate" in str(err).lower() or "overloaded" in str(err).lower():
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
        raise RuntimeError(f"Claude call failed after backoff: {last_err}")
