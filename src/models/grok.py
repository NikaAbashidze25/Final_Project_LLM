"""xAI (Grok) client. Grok exposes an OpenAI-compatible API."""
from __future__ import annotations

import time
from typing import Any, Optional

from pydantic import BaseModel

from .base_llm import BaseLLM, strict_json_schema, supports_strict_mode


class GrokLLM(BaseLLM):
    BASE_URL = "https://api.x.ai/v1"

    def __init__(self, identity: str, model_name: str, api_key: str) -> None:
        super().__init__(identity, model_name)
        if not api_key:
            raise ValueError("GROK_API_KEY is required for the Grok backend.")
        from openai import OpenAI  # Grok is OpenAI-compatible

        self._client = OpenAI(api_key=api_key, base_url=self.BASE_URL)

    def _response_format(self, schema: Optional[type[BaseModel]]) -> dict[str, Any]:
        if schema is None:
            return {"type": "json_object"}
        tightened = strict_json_schema(schema)
        if not supports_strict_mode(tightened):
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": tightened,
                "strict": True,
            },
        }

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.7,
        response_context: Optional[dict[str, Any]] = None,
        schema: Optional[type[BaseModel]] = None,
    ) -> str:
        response_format = self._response_format(schema)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        delay = 2.0
        last_err: Optional[Exception] = None
        for _ in range(5):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model_name,
                    temperature=temperature,
                    response_format=response_format,
                    messages=messages,
                )
                return resp.choices[0].message.content or ""
            except Exception as err:
                last_err = err
                msg = str(err).lower()
                if "429" in msg or "rate" in msg:
                    time.sleep(delay)
                    delay *= 2
                    continue
                # xAI's json_schema strict-mode support is unconfirmed; if the
                # schema-constrained request itself is rejected as invalid,
                # fall back to plain json_object mode instead of failing the
                # whole call (the caller validates the parsed JSON either way).
                if schema is not None and response_format["type"] == "json_schema" and (
                    "400" in msg or "invalid" in msg or "unsupported" in msg
                ):
                    response_format = {"type": "json_object"}
                    continue
                raise
        raise RuntimeError(f"Grok call failed after backoff: {last_err}")
