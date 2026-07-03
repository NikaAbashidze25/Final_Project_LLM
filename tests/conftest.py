"""Shared fixtures for the src/stages/ test suite.

Every test runs against the deterministic offline backend, so the suite
needs no network access or API keys and produces reproducible results.
"""
from __future__ import annotations

import pytest

from src import config, utils


@pytest.fixture(autouse=True)
def offline_backend(monkeypatch):
    """Force the offline simulation backend regardless of local .env settings."""
    monkeypatch.setattr(config, "LLM_BACKEND", "offline")


@pytest.fixture
def problem() -> dict:
    """A minimal, schema-valid problem independent of data/problems.json."""
    return {
        "id": "test_prob_001",
        "category": "mathematical_reasoning",
        "problem": "What is the sum of the first 10 positive integers?",
        "correct_answer": "55",
        "difficulty": "medium",
        "source_notes": "synthetic fixture for stages/ unit tests",
    }


@pytest.fixture
def logger(tmp_path, monkeypatch):
    """A real logger writing to a throwaway directory (no repo-tree side effects)."""
    monkeypatch.setattr(config, "ensure_directories", lambda: None)
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path)
    return utils.setup_logger("pytest")
