"""Model factory.

`get_model(identity)` returns a ready-to-use client for the given canonical
model identity ("gpt-4", "claude", "gemini", "grok"), honoring the configured
backend (offline simulation vs. a real provider) and free-model mode.
"""
from __future__ import annotations

from functools import lru_cache

from .. import config
from .base_llm import BaseLLM
from .offline_backend import OfflineLLM

__all__ = ["BaseLLM", "OfflineLLM", "get_model"]


def _real_client(identity: str) -> BaseLLM:
    """Instantiate a real provider client for `identity`."""
    # In free-model mode, every identity is served by one underlying provider.
    provider = config.FREE_MODEL_PROVIDER if config.FREE_MODEL_MODE else identity
    if provider in ("gpt-4", "openai"):
        from .gpt4 import GPT4LLM
        return GPT4LLM(identity, config.MODEL_NAMES["openai"], config.API_KEYS["openai"])
    if provider == "claude":
        from .claude import ClaudeLLM
        return ClaudeLLM(identity, config.MODEL_NAMES["claude"], config.API_KEYS["claude"])
    if provider == "gemini":
        from .gemini import GeminiLLM
        return GeminiLLM(identity, config.MODEL_NAMES["gemini"], config.API_KEYS["gemini"])
    if provider == "grok":
        from .grok import GrokLLM
        return GrokLLM(identity, config.MODEL_NAMES["grok"], config.API_KEYS["grok"])
    raise ValueError(f"Unknown provider for identity {identity!r}: {provider!r}")


@lru_cache(maxsize=None)
def get_model(identity: str) -> BaseLLM:
    """Return a (cached) client for the given canonical model identity."""
    if config.LLM_BACKEND == "offline":
        return OfflineLLM(identity, f"offline-sim::{identity}", seed=config.RANDOM_SEED)
    return _real_client(identity)
