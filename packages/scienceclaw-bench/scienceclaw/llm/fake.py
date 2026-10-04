"""FakeLLM: offline stand-in with the public API of :class:`LLMClient` (for tests).

``FakeLLM(responder)`` where ``responder`` is either

* a list of items consumed in call order (raises :class:`FakeLLMExhausted` when
  empty), or
* a callable ``(role, messages) -> item``.

An item is a ``str`` (the response text), an :class:`LLMResponse` (its ``text`` and
``finish_reason`` are used), or an exception instance, which is raised (handy for
testing error paths, e.g. ``LLMError("boom", status=500)``).

Every call is recorded in ``.calls`` (a list of dicts with role, messages, keyword
arguments, index and response text). Usage is counted as ``len(text) // 4`` tokens
(prompt tokens from the concatenated message contents), reported in the same format
as :meth:`LLMClient.usage`.
"""
from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Union

from scienceclaw.config import LLMConfig
from scienceclaw.llm.client import (
    ROLES,
    LLMError,
    LLMResponse,
    UsageTracker,
    _approx_tokens,
    _collect,
    _error_response,
    empty_usage,
)

FakeItem = Union[str, LLMResponse, BaseException]
Responder = Union[Sequence[FakeItem], Callable[[str, list[dict]], FakeItem]]


class FakeLLMExhausted(RuntimeError):
    """Raised when a list responder has no responses left."""


class FakeLLM:
    """Deterministic fake LLM with the same public API as ``LLMClient``.

    Args:
        responder: list of items (consumed in order) or ``callable(role, messages)``.
        cfg: optional ``LLMConfig`` exposed as ``.cfg`` (defaults to ``LLMConfig()``),
            so code that reads role settings from the client keeps working.
        model: model name reported in responses.
        concurrent: if True, ``chat_many`` uses a thread pool like the real client;
            by default it runs sequentially so list responders are consumed in batch
            order (deterministic).
    """

    def __init__(self, responder: Responder, cfg: LLMConfig | None = None, *, model: str = "fake-llm",
                 concurrent: bool = False) -> None:
        if isinstance(responder, (str, bytes)):
            raise TypeError("responder must be a list of responses or a callable, not a single string")
        if callable(responder):
            self._fn: Callable[[str, list[dict]], FakeItem] | None = responder
            self._queue: list[FakeItem] | None = None
        elif isinstance(responder, Sequence):
            self._fn = None
            self._queue = list(responder)
        else:
            raise TypeError(f"unsupported responder type {type(responder).__name__}")
        self.cfg = cfg or LLMConfig()
        self.model = model
        self.concurrent = concurrent
        self.calls: list[dict] = []
        self._lock = threading.Lock()
        self._tracker = UsageTracker()

    @property
    def remaining(self) -> int | None:
        """Responses left for a list responder (``None`` for a callable responder)."""
        with self._lock:
            return None if self._queue is None else len(self._queue)

    def chat(self, role: str, messages: list[dict], *, json_mode: bool | None = None,
             max_tokens: int | None = None, temperature: float | None = None,
             cache_salt: str = "", tag: str = "") -> LLMResponse:
        if role not in ROLES:
            raise ValueError(f"unknown LLM role {role!r}; expected one of {ROLES}")
        record: dict[str, Any] = {"role": role, "messages": messages, "json_mode": json_mode,
                                  "max_tokens": max_tokens, "temperature": temperature,
                                  "cache_salt": cache_salt, "tag": tag, "text": None}
        with self._lock:
            record["index"] = len(self.calls)
            self.calls.append(record)
            if self._queue is not None:
                if not self._queue:
                    raise FakeLLMExhausted(
                        f"FakeLLM responses exhausted at call #{record['index']} (role={role!r}, tag={tag!r})")
                item: FakeItem = self._queue.pop(0)
        if self._fn is not None:
            item = self._fn(role, messages)

        if isinstance(item, BaseException):
            record["error"] = f"{type(item).__name__}: {item}"
            self._tracker.record(role, tag, self.model, empty_usage(), 0.0, {"errors": 1})
            raise item
        if isinstance(item, LLMResponse):
            text, finish = item.text, item.finish_reason
        elif isinstance(item, str):
            text, finish = item, "stop"
        else:
            raise TypeError(f"FakeLLM responder returned {type(item).__name__}; expected str, "
                            "LLMResponse or an exception instance")
        record["text"] = text
        usage = empty_usage()
        usage["prompt_tokens"] = _approx_tokens(messages)
        usage["completion_tokens"] = len(text) // 4
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        usage["calls"] = 1
        self._tracker.record(role, tag, self.model, usage, 0.0)
        return LLMResponse(text=text, usage=usage, cached=False, latency_s=0.0, model=self.model,
                           finish_reason=finish, role=role, tag=tag)

    def chat_many(self, role: str, batch: list[list[dict]], *, return_exceptions: bool = False,
                  max_workers: int | None = None, **kw: Any) -> list[LLMResponse]:
        """Order-preserving batch; same error semantics as ``LLMClient.chat_many``."""
        if not batch:
            return []
        tag = kw.get("tag", "")
        if not self.concurrent:
            out: list[LLMResponse] = []
            for msgs in batch:
                try:
                    out.append(self.chat(role, msgs, **kw))
                except LLMError as e:
                    if not return_exceptions:
                        raise
                    out.append(_error_response(role, tag, e))
            return out
        workers = max(1, min(len(batch), int(max_workers or self.cfg.concurrency)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"fake-llm-{role}") as pool:
            futures = [pool.submit(self.chat, role, msgs, **kw) for msgs in batch]
            return _collect(role, tag, futures, return_exceptions)

    def usage(self) -> dict:
        return self._tracker.snapshot()

    def reset_usage(self) -> None:
        self._tracker.reset()

    def close(self) -> None:
        """No-op (API parity with ``LLMClient``)."""

    def __enter__(self) -> "FakeLLM":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
