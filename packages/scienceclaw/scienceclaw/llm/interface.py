"""The language model as an interface.

ScienceClaw never depends on a particular model family. Everything that needs a language model (the canvas policy when it
runs inside the engine, ``llm`` nodes, Skill patching, Operator documentation) talks to a :class:`ChatModel`:

* ``backend: openai`` (default) -> :class:`scienceclaw.llm.client.LLMClient`, an OpenAI-compatible HTTP client for a hosted
  gateway or for self-hosted servers (``llm.endpoints``). Model names come from the config or from
  ``SCIENCECLAW_MODEL`` / ``SCIENCECLAW_<ROLE>_MODEL``; credentials from ``SCIENCECLAW_API_BASE_URL`` and
  ``SCIENCECLAW_API_KEY`` or the credentials file.
* ``backend: command`` -> :class:`scienceclaw.llm.command.CommandChatModel`, completions from a local program
  (``llm.command`` / ``SCIENCECLAW_LLM_COMMAND``).
* ``backend: package.module:factory`` -> any callable ``factory(cfg: LLMConfig, **kw) -> ChatModel``. This is the extension
  point for a host application that already owns a model connection.
* :func:`register_backend` adds a named backend at run time.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable, Protocol, runtime_checkable

from scienceclaw.config import LLMConfig, ModelRole
from scienceclaw.llm.client import LLMClient, LLMResponse
from scienceclaw.llm.command import CommandChatModel


@runtime_checkable
class ChatModel(Protocol):
    """What the engine needs from a language model."""

    cfg: LLMConfig

    def role_config(self, role: str) -> ModelRole: ...

    def chat(self, role: str, messages: list[dict], *, json_mode: bool | None = None, max_tokens: int | None = None,
             temperature: float | None = None, cache_salt: str = "", tag: str = "") -> LLMResponse: ...

    def usage(self) -> dict: ...

    def close(self) -> None: ...


Factory = Callable[..., ChatModel]
_BACKENDS: dict[str, Factory] = {"openai": LLMClient, "command": CommandChatModel}


def register_backend(name: str, factory: Factory) -> None:
    """Make ``backend: <name>`` available in configs."""
    if not name or ":" in name:
        raise ValueError("backend names must be non-empty and must not contain ':'")
    _BACKENDS[name] = factory


def _import_factory(spec: str) -> Factory:
    module, _, attr = spec.partition(":")
    if not module or not attr:
        raise ValueError(f"backend {spec!r}: expected 'package.module:factory'")
    return getattr(importlib.import_module(module), attr)


def build_chat_model(cfg: LLMConfig, **kw: Any) -> ChatModel:
    """Instantiate the chat model selected by ``cfg.backend``."""
    name = (cfg.backend or "openai").strip()
    factory = _BACKENDS.get(name) or (_import_factory(name) if ":" in name else None)
    if factory is None:
        raise ValueError(f"unknown LLM backend {name!r}; registered: {sorted(_BACKENDS)}")
    return factory(cfg, **kw)
