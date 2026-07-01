"""OpenAI (GPT-4 family) client."""
from __future__ import annotations

import time
from typing import Any, Optional

from .base_llm import BaseLLM


class GPT4LLM(BaseLLM):
    def __init__(self, identity: str, model_name: str, api_key: str) -> None:
        super().__init__(identity, model_name)
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI backend.")
        from openai import OpenAI  # lazy import so offline mode needs no SDK

        self._client = OpenAI(api_key=api_key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.7,
        response_context: Optional[dict[str, Any]] = None,
    ) -> str:
        delay = 2.0
        last_err: Optional[Exception] = None
        for _ in range(5):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model_name,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return resp.choices[0].message.content or ""
            except Exception as err:  # broad: handle rate limits + transient errors
                last_err = err
                if "429" in str(err) or "rate" in str(err).lower():
                    time.sleep(delay)
                    delay *= 2  # exponential backoff
                    continue
                raise
        raise RuntimeError(f"OpenAI call failed after backoff: {last_err}")
