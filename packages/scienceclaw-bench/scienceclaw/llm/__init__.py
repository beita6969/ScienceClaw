"""LLM access layer (DESIGN.md 8.1): real client, fake client, JSON extraction."""
from scienceclaw.llm.client import (
    ROLES,
    USAGE_KEYS,
    LLMClient,
    LLMError,
    LLMResponse,
    ResponseCache,
    UsageTracker,
    add_usage,
    build_request_body,
    cache_key,
    empty_usage,
    extract_json,
    is_reasoning_model,
    logical_usage,
)
from scienceclaw.llm.fake import FakeLLM, FakeLLMExhausted

__all__ = [
    "ROLES", "USAGE_KEYS", "LLMClient", "LLMError", "LLMResponse", "ResponseCache", "UsageTracker",
    "add_usage", "build_request_body", "cache_key", "empty_usage", "extract_json", "is_reasoning_model",
    "logical_usage", "FakeLLM", "FakeLLMExhausted",
]
