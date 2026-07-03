"""Google Gemini client."""
from __future__ import annotations

import time
from typing import Any, Optional

from pydantic import BaseModel

from .base_llm import BaseLLM


class GeminiLLM(BaseLLM):
    def __init__(self, identity: str, model_name: str, api_key: str) -> None:
        super().__init__(identity, model_name)
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for the Gemini backend.")
        import google.generativeai as genai  # lazy import

        genai.configure(api_key=api_key)
        self._genai = genai

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.7,
        response_context: Optional[dict[str, Any]] = None,
        schema: Optional[type[BaseModel]] = None,
    ) -> str:
        generation_config: dict[str, Any] = {
            "temperature": temperature,
            "response_mime_type": "application/json",
        }
        if schema is not None:
            # The SDK accepts a pydantic model class directly and converts it
            # to Gemini's (OpenAPI-subset) response_schema internally.
            generation_config["response_schema"] = schema

        model = self._genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt,
            generation_config=generation_config,
        )
        delay = 2.0
        last_err: Optional[Exception] = None
        for _ in range(5):
            try:
                resp = model.generate_content(user_prompt)
                return resp.text or ""
            except Exception as err:
                last_err = err
                if "429" in str(err) or "rate" in str(err).lower() or "quota" in str(err).lower():
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
        raise RuntimeError(f"Gemini call failed after backoff: {last_err}")
