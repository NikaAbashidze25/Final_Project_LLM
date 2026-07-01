"""Anthropic (Claude) client."""
from __future__ import annotations

import time
from typing import Any, Optional

from .base_llm import BaseLLM


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
    ) -> str:
        delay = 2.0
        last_err: Optional[Exception] = None
        for _ in range(5):
            try:
                resp = self._client.messages.create(
                    model=self.model_name,
                    max_tokens=4096,
                    temperature=temperature,
                    system=system_prompt + "\nReturn ONLY valid JSON.",
                    messages=[{"role": "user", "content": user_prompt}],
                )
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
