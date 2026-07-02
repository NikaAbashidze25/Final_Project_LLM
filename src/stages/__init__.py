"""Stage orchestration modules for the Multi-LLM Debate pipeline.

Each module implements exactly one stage from the README workflow and is a
thin orchestration layer: it builds prompts, calls `utils.call_llm_validated`
for each participating model, and returns validated pydantic objects (plus
raw dicts, ready to be persisted with `utils.save_json`). The heavy lifting
(JSON parsing, retries, schema validation, logging) lives in `src/utils.py`;
these modules only decide *who* gets called with *what context*.
"""
