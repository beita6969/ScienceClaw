"""LLM access layer (DESIGN.md 8.1): the ChatModel interface, the OpenAI-compatible client and JSON extraction."""
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
from scienceclaw.llm.interface import ChatModel, build_chat_model, register_backend

__all__ = [
    "ChatModel", "build_chat_model", "register_backend",
    "ROLES", "USAGE_KEYS", "LLMClient", "LLMError", "LLMResponse", "ResponseCache", "UsageTracker",
    "add_usage", "build_request_body", "cache_key", "empty_usage", "extract_json", "is_reasoning_model",
    "logical_usage",
]
