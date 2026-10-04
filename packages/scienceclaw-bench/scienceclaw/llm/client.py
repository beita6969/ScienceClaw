"""LLM access layer: OpenAI-compatible chat client with cache, retries and accounting.

This module implements DESIGN.md section 8.1. The foundation model Θ0 is only ever
*queried* through :class:`LLMClient`; its parameters are never touched (paper Eq. 3,
"Θ fixed").

Key properties
--------------
* **Roles.** ``role`` ∈ {"policy", "executor", "patch"} selects ``cfg.<role>``
  (:class:`scienceclaw.config.ModelRole`).
* **Request shape.** Reasoning models (name contains ``gpt-5``, ``o1``, ``o3`` or
  ``o4``) receive ``max_completion_tokens`` (+ ``reasoning={"effort": ...}`` when set and
  ``temperature`` only when not ``None``); all other models receive ``max_tokens``
  and ``temperature``. ``json_mode`` adds ``response_format={"type": "json_object"}``.
* **Length retry.** Reasoning tokens count against the completion limit, so an
  *empty* answer with ``finish_reason == "length"`` is retried exactly once with the
  limit doubled. If it is still empty the empty text is returned with
  ``finish_reason="length"`` so callers can see why.
* **Transport retries.** Exponential backoff with jitter (honouring ``Retry-After``)
  on HTTP 408/409/425/429/5xx, timeouts, connection resets/refusals and malformed
  gateway payloads, up to ``cfg.max_retries`` retries. Other 4xx (400/401/403/...)
  raise :class:`LLMError` immediately, except a non-auth 4xx whose JSON error code
  is transient (the lab gateway sends ``400 {"code": "upstream_error"}`` for upstream
  failures), which gets at most :data:`GATEWAY_ERROR_RETRY_LIMIT` retries.
* **Gateway quirks.** The lab gateway reports ``finish_reason="stop"`` even when the
  completion limit was hit, so ``finish_reason`` is normalised to ``"length"`` when
  ``completion_tokens >= limit``; limits below :data:`MIN_COMPLETION_TOKENS` are
  raised to it (the gateway rejects them).
* **Concurrency.** A single semaphore of ``cfg.concurrency`` slots per client
  bounds in-flight HTTP requests across *all* threads that share the client (the
  semaphore is not held while sleeping between retries).
* **Cache.** SQLite (WAL) keyed by ``sha256(model, messages, params, cache_salt)``.
  Hits return ``cached=True``; their tokens are reported under ``cached_*`` keys and
  are *not* counted as spent (see :func:`logical_usage` for budget-style counting).
* **Secrets.** The API key is read from ``cfg.credentials_file`` for each request
  and is never stored on the client, logged, written to the cache, pickled or
  included in exception messages.
"""
from __future__ import annotations

import ast
import hashlib
import http.client
import json
import logging
import random
import re
import socket
import sqlite3
import ssl
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urlparse

from scienceclaw.config import LLMConfig, ModelRole

log = logging.getLogger(__name__)

ROLES: tuple[str, ...] = ("policy", "executor", "patch")
RETRYABLE_STATUS: frozenset[int] = frozenset({408, 409, 425, 429})
# Gateway error codes that are transient even when sent with a 4xx status (never for 401/403).
TRANSIENT_ERROR_CODES: frozenset[str] = frozenset({
    "upstream_error", "server_error", "rate_limit_exceeded", "overloaded", "overloaded_error", "timeout"})
GATEWAY_ERROR_RETRY_LIMIT = 2        # retries for a 4xx carrying a transient gateway error code
ENDPOINT_COOLDOWN_S = 20.0           # a self-hosted endpoint that failed is skipped for this long
MIN_COMPLETION_TOKENS = 16           # the gateway rejects smaller limits (HTTP 400 upstream_error)
DEFAULT_USER_AGENT = "scienceclaw-rebuild/0.1"
MAX_COMPLETION_LIMIT = 65_536        # cap for the doubled limit of the single length retry
_ERROR_BODY_CHARS = 500

# Per-call usage keys (LLMResponse.usage). "calls"/token keys count only *spent*
# (non-cached) work; "cached_*" keys report work served from the cache.
USAGE_KEYS: tuple[str, ...] = (
    "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens",
    "calls", "cached_calls", "cached_prompt_tokens", "cached_completion_tokens",
    "cached_reasoning_tokens",
)
# Extra keys that only appear in the aggregates returned by ``usage()``.
AGGREGATE_EXTRA_KEYS: tuple[str, ...] = ("latency_s", "retries", "length_retries", "length_inferred", "errors",
                                         "usage_estimated")


# --------------------------------------------------------------------------- data

@dataclass
class LLMResponse:
    """One chat completion.

    ``usage`` always contains every key of :data:`USAGE_KEYS`. For a cache hit the
    spent keys (``prompt_tokens``, ``completion_tokens``, ``reasoning_tokens``,
    ``total_tokens``, ``calls``) are 0 and the original counts are under ``cached_*``.
    ``calls`` can be 2 when the single length retry fired (both requests are billed).
    """

    text: str
    usage: dict
    cached: bool
    latency_s: float
    model: str
    finish_reason: str | None
    role: str = ""
    tag: str = ""
    error: str | None = None          # only set by chat_many(..., return_exceptions=True)

    def to_dict(self) -> dict:
        return asdict(self)


class LLMError(RuntimeError):
    """Failure talking to the model gateway. Never contains the API key."""

    def __init__(self, message: str, *, status: int | None = None, body: str = "",
                 retryable: bool = False, retry_after: float | None = None,
                 model: str = "", role: str = "", retry_limit: int | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body
        self.retryable = retryable
        self.retry_after = retry_after
        self.model = model
        self.role = role
        self.retry_limit = retry_limit          # max retries for this kind of error (None = cfg.max_retries)
        self.attempts = 0                       # HTTP attempts made before giving up
        self.spent_usage: dict | None = None   # billed usage of requests completed before the failure

    def __str__(self) -> str:
        base = super().__str__()
        parts = [base]
        if self.status is not None and f"HTTP {self.status}" not in base:
            parts.append(f"status={self.status}")
        if self.body:
            parts.append(f"body={self.body!r}")
        return " | ".join(parts)


# ------------------------------------------------------------------- usage helpers

def empty_usage() -> dict:
    """A per-call usage dict with every :data:`USAGE_KEYS` entry set to 0."""
    return {k: 0 for k in USAGE_KEYS}


def add_usage(dst: dict, src: dict) -> dict:
    """Add the numeric entries of ``src`` into ``dst`` (in place) and return ``dst``."""
    for k, v in src.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        dst[k] = dst.get(k, 0) + v
    return dst


def logical_usage(usage: dict) -> dict:
    """Token counts irrespective of caching (spent + cached).

    Use this for *budget* checks that must behave identically whether or not a
    response came from the cache (e.g. ``Budget.max_policy_tokens``); use the plain
    spent keys for *cost* accounting.
    """
    p = usage.get("prompt_tokens", 0) + usage.get("cached_prompt_tokens", 0)
    c = usage.get("completion_tokens", 0) + usage.get("cached_completion_tokens", 0)
    r = usage.get("reasoning_tokens", 0) + usage.get("cached_reasoning_tokens", 0)
    return {"prompt_tokens": p, "completion_tokens": c, "reasoning_tokens": r, "total_tokens": p + c,
            "calls": usage.get("calls", 0) + usage.get("cached_calls", 0)}


def _as_cached_usage(spent: dict) -> dict:
    """Convert the stored (spent) usage of a cache entry into the usage of a cache hit."""
    u = empty_usage()
    u["cached_calls"] = 1
    u["cached_prompt_tokens"] = int(spent.get("prompt_tokens", 0))
    u["cached_completion_tokens"] = int(spent.get("completion_tokens", 0))
    u["cached_reasoning_tokens"] = int(spent.get("reasoning_tokens", 0))
    return u


class UsageTracker:
    """Thread-safe usage aggregation: total, by_role, by_tag, by_model."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    @staticmethod
    def _zero() -> dict:
        z: dict[str, float] = {k: 0 for k in USAGE_KEYS}
        z.update({k: 0 for k in AGGREGATE_EXTRA_KEYS})
        z["latency_s"] = 0.0
        return z

    def reset(self) -> None:
        with self._lock:
            self._total = self._zero()
            self._by_role: dict[str, dict] = {}
            self._by_tag: dict[str, dict] = {}
            self._by_model: dict[str, dict] = {}

    def record(self, role: str, tag: str, model: str, usage: dict, latency_s: float = 0.0,
               extra: dict | None = None) -> None:
        delta = dict(usage)
        delta["latency_s"] = float(latency_s)
        if extra:
            add_usage(delta, extra)
        with self._lock:
            for bucket in (self._total,
                           self._by_role.setdefault(role, self._zero()),
                           self._by_tag.setdefault(tag or "untagged", self._zero()),
                           self._by_model.setdefault(model, self._zero())):
                add_usage(bucket, delta)

    def snapshot(self) -> dict:
        with self._lock:
            return {"total": dict(self._total),
                    "by_role": {k: dict(v) for k, v in self._by_role.items()},
                    "by_tag": {k: dict(v) for k, v in self._by_tag.items()},
                    "by_model": {k: dict(v) for k, v in self._by_model.items()}}


# ------------------------------------------------------------------ request shape

_O_SERIES = re.compile(r"(?<![a-z0-9])o[134](?![0-9])")


def is_reasoning_model(model: str) -> bool:
    """True for reasoning models (names containing ``gpt-5`` or an ``o1``/``o3``/``o4`` token)."""
    m = model.lower()
    return "gpt-5" in m or _O_SERIES.search(m) is not None


def build_request_body(role_cfg: ModelRole, messages: list[dict], *, json_mode: bool | None = None,
                       max_tokens: int | None = None, temperature: float | None = None) -> dict:
    """OpenAI-compatible ``/chat/completions`` body for one call.

    Call-level ``json_mode``/``max_tokens``/``temperature`` override the role config
    when not ``None``. The completion limit is raised to at least
    :data:`MIN_COMPLETION_TOKENS` (the gateway rejects smaller limits).
    """
    limit = int(max_tokens if max_tokens is not None else role_cfg.max_tokens)
    if limit <= 0:
        raise ValueError(f"max_tokens must be positive, got {limit}")
    limit = max(limit, MIN_COMPLETION_TOKENS)
    temp = temperature if temperature is not None else role_cfg.temperature
    jm = role_cfg.json_mode if json_mode is None else json_mode
    body: dict[str, Any] = {"model": role_cfg.model, "messages": messages}
    if is_reasoning_model(role_cfg.model):
        body["max_completion_tokens"] = limit
        if role_cfg.reasoning_effort is not None:
            # The flowsteer gateway ignores the chat-completions "reasoning_effort" field (it silently
            # uses its default, ~medium) but honours the Responses-style "reasoning": {"effort": ...}
            # object (verified 2026-09-28: low -> ~90-220 reasoning tokens, "reasoning_effort": "low"
            # -> 500-2800). "none"/"minimal" are rejected upstream (HTTP 400), so they map to "low".
            effort = role_cfg.reasoning_effort
            if effort in ("none", "minimal"):
                effort = "low"
            body["reasoning"] = {"effort": effort}
        if temp is not None:
            body["temperature"] = float(temp)
    else:
        body["max_tokens"] = limit
        if temp is not None:
            body["temperature"] = float(temp)
    if jm:
        body["response_format"] = {"type": "json_object"}
    if role_cfg.extra_body:
        body.update(json.loads(json.dumps(role_cfg.extra_body)))
    return body


def cache_key(body: dict, cache_salt: str = "") -> str:
    """sha256 over (model, messages, params, cache_salt) with canonical JSON."""
    params = {k: v for k, v in body.items() if k not in ("model", "messages")}
    blob = json.dumps({"model": body["model"], "messages": body["messages"], "params": params,
                       "cache_salt": cache_salt}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _limit_key(body: dict) -> str:
    return "max_completion_tokens" if "max_completion_tokens" in body else "max_tokens"


def _check_messages(messages: list[dict]) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list of {'role','content'} dicts")
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise ValueError(f"messages[{i}] must be a dict with 'role' and 'content'")


def _approx_tokens(messages: list[dict]) -> int:
    chars = 0
    for m in messages:
        c = m.get("content")
        chars += len(c) if isinstance(c, str) else len(json.dumps(c, ensure_ascii=False))
    return chars // 4


# ---------------------------------------------------------------------- redaction

_BEARER_RE = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=\-]+")
_SK_RE = re.compile(r"\bsk-[A-Za-z0-9_\-]{6,}")


def _redact(text: str, secret: str | None = None) -> str:
    if secret:
        text = text.replace(secret, "[REDACTED]")
    text = _BEARER_RE.sub(r"\1[REDACTED]", text)
    return _SK_RE.sub("sk-[REDACTED]", text)


def _truncate(text: str, n: int = _ERROR_BODY_CHARS) -> str:
    return text if len(text) <= n else text[:n] + f"...[+{len(text) - n} chars]"


# -------------------------------------------------------------------------- cache

class ResponseCache:
    """SQLite response cache (WAL; one shared connection guarded by a lock).

    Stores only the key hash, model name, response text, spent usage and finish
    reason — never request headers or the API key. Safe to share between threads,
    and between processes via SQLite's own locking.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), timeout=60.0, check_same_thread=False, isolation_level=None)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=60000")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS responses ("
                " key TEXT PRIMARY KEY, model TEXT NOT NULL, text TEXT NOT NULL, usage TEXT NOT NULL,"
                " finish_reason TEXT, latency_s REAL, created REAL NOT NULL)")

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT text, usage, finish_reason, latency_s, model FROM responses WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return {"text": row[0], "usage": json.loads(row[1]), "finish_reason": row[2],
                "latency_s": row[3], "model": row[4]}

    def put(self, key: str, *, model: str, text: str, usage: dict, finish_reason: str | None,
            latency_s: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses (key, model, text, usage, finish_reason, latency_s, created)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (key, model, text, json.dumps(usage, sort_keys=True), finish_reason, float(latency_s), time.time()))

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ------------------------------------------------------------------------- client

class LLMClient:
    """Thread-safe OpenAI-compatible chat client (DESIGN.md 8.1).

    Args:
        cfg: LLM configuration (roles, credentials file, concurrency, retries, cache).
        cache_dir: if given, the cache lives at ``cache_dir/llm_cache.sqlite``;
            otherwise at ``cfg.cache_path``. No cache when ``cfg.use_cache`` is False
            (or ``cfg.cache_path`` is empty and no ``cache_dir`` is given).
        user_agent: User-Agent header (REQUIRED by the gateway; 403 without it).
        backoff_base_s / backoff_max_s: exponential backoff parameters.
        sleep: sleep function used between retries (injectable for tests).
        seed: seed of the jitter RNG (``None`` = nondeterministic jitter).
    """

    def __init__(self, cfg: LLMConfig, cache_dir: str | Path | None = None, *,
                 user_agent: str = DEFAULT_USER_AGENT, backoff_base_s: float = 1.0,
                 backoff_max_s: float = 60.0, sleep: Callable[[float], None] = time.sleep,
                 seed: int | None = None) -> None:
        if not user_agent:
            raise ValueError("user_agent must be non-empty (the gateway rejects requests without one)")
        self.cfg = cfg
        self._user_agent = user_agent
        self._backoff_base = float(backoff_base_s)
        self._backoff_max = float(backoff_max_s)
        self._sleep = sleep
        self._rng = random.Random(seed)
        self._rng_lock = threading.Lock()
        self._sem = threading.BoundedSemaphore(max(1, int(cfg.concurrency)))
        self._eps = [e.strip().rstrip("/") for e in cfg.endpoints if e.strip()]
        self._ep_lock = threading.Lock()
        self._ep_inflight = {e: 0 for e in self._eps}
        self._ep_down_until = {e: 0.0 for e in self._eps}
        self._tracker = UsageTracker()
        self._cache: ResponseCache | None = None
        if cfg.use_cache:
            if cache_dir is not None:
                self._cache = ResponseCache(Path(cache_dir).expanduser() / "llm_cache.sqlite")
            elif cfg.cache_path:
                self._cache = ResponseCache(cfg.cache_path)

    # -- public API -------------------------------------------------------------

    @property
    def cache_path(self) -> Path | None:
        return self._cache.path if self._cache is not None else None

    def role_config(self, role: str) -> ModelRole:
        if role not in ROLES:
            raise ValueError(f"unknown LLM role {role!r}; expected one of {ROLES}")
        return getattr(self.cfg, role)

    def chat(self, role: str, messages: list[dict], *, json_mode: bool | None = None,
             max_tokens: int | None = None, temperature: float | None = None,
             cache_salt: str = "", tag: str = "") -> LLMResponse:
        """One chat completion for ``role`` (cached, retried, accounted)."""
        role_cfg = self.role_config(role)
        _check_messages(messages)
        body = build_request_body(role_cfg, messages, json_mode=json_mode, max_tokens=max_tokens,
                                  temperature=temperature)
        key = cache_key(body, cache_salt)
        t0 = time.monotonic()

        if self._cache is not None:
            hit = self._cache.get(key)
            if hit is not None:
                usage = _as_cached_usage(hit["usage"])
                latency = time.monotonic() - t0
                self._tracker.record(role, tag, role_cfg.model, usage, latency)
                return LLMResponse(text=hit["text"], usage=usage, cached=True, latency_s=latency,
                                   model=role_cfg.model, finish_reason=hit["finish_reason"], role=role, tag=tag)

        try:
            text, usage, finish, stats = self._complete(body, role)
        except LLMError as e:
            e.role = e.role or role
            e.model = e.model or role_cfg.model
            self._tracker.record(role, tag, role_cfg.model, e.spent_usage or empty_usage(),
                                 time.monotonic() - t0, {"errors": 1, "retries": max(0, e.attempts - 1)})
            raise
        latency = time.monotonic() - t0
        self._tracker.record(role, tag, role_cfg.model, usage, latency, stats)
        if self._cache is not None and text.strip():
            # Empty answers are not cached so that a later run gets another chance.
            self._cache.put(key, model=role_cfg.model, text=text, usage=usage, finish_reason=finish,
                            latency_s=latency)
        return LLMResponse(text=text, usage=usage, cached=False, latency_s=latency, model=role_cfg.model,
                           finish_reason=finish, role=role, tag=tag)

    def chat_many(self, role: str, batch: list[list[dict]], *, return_exceptions: bool = False,
                  max_workers: int | None = None, **kw: Any) -> list[LLMResponse]:
        """Run ``chat(role, messages, **kw)`` for every conversation concurrently; order-preserving.

        With ``return_exceptions=True`` an item whose call raised :class:`LLMError`
        becomes an ``LLMResponse`` with empty text, ``finish_reason="error"`` and
        ``error`` set; otherwise the first error (in batch order) is re-raised after
        cancelling not-yet-started items. Other exceptions always propagate.
        """
        if not batch:
            return []
        tag = kw.get("tag", "")
        workers = max(1, min(len(batch), int(max_workers or self.cfg.concurrency)))
        if workers == 1:
            out: list[LLMResponse] = []
            for msgs in batch:
                try:
                    out.append(self.chat(role, msgs, **kw))
                except LLMError as e:
                    if not return_exceptions:
                        raise
                    out.append(_error_response(role, tag, e))
            return out
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"llm-{role}") as pool:
            futures = [pool.submit(self.chat, role, msgs, **kw) for msgs in batch]
            try:
                return _collect(role, tag, futures, return_exceptions)
            except BaseException:
                for f in futures:
                    f.cancel()
                raise

    def usage(self) -> dict:
        """Thread-safe snapshot ``{"total", "by_role", "by_tag", "by_model"}``.

        Token keys and ``calls`` count only non-cached (spent) work; cache hits are
        counted in ``cached_calls`` and ``cached_*_tokens``. ``latency_s`` sums the
        wall time of all ``chat`` calls (cache lookups included, they are ~ms).
        """
        return self._tracker.snapshot()

    def reset_usage(self) -> None:
        self._tracker.reset()

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()
            self._cache = None

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        roles = ", ".join(f"{r}={getattr(self.cfg, r).model}" for r in ROLES)
        return f"LLMClient({roles}, cache={self.cache_path})"

    def __getstate__(self) -> dict:
        raise TypeError("LLMClient is not picklable (holds locks, a DB handle and access to credentials)")

    # -- internals --------------------------------------------------------------

    def _complete(self, body: dict, role: str) -> tuple[str, dict, str | None, dict]:
        """POST (with retries) and apply the single empty-``length`` retry."""
        data, retries = self._post_with_retries(body, role)
        text, finish, usage, pstats = self._parse(data, body)
        stats = {"retries": retries, **pstats}
        if not text.strip() and finish == "length":
            lk = _limit_key(body)
            body2 = dict(body)
            body2[lk] = min(int(body[lk]) * 2, MAX_COMPLETION_LIMIT)
            log.warning("LLM %s/%s returned empty content with finish_reason=length at %s=%d; "
                        "retrying once with %d", role, body["model"], lk, body[lk], body2[lk])
            try:
                data2, retries2 = self._post_with_retries(body2, role)
            except LLMError as e:
                e.spent_usage = usage       # the first (billed) request still counts as spent
                raise
            text, finish, usage2, pstats2 = self._parse(data2, body2)
            add_usage(usage, usage2)
            add_usage(stats, pstats2)
            stats["retries"] += retries2
            stats["length_retries"] = 1
            if not text.strip():
                log.warning("LLM %s/%s still empty after length retry (finish_reason=%s)",
                            role, body["model"], finish)
        return text, usage, finish, stats

    @staticmethod
    def _parse(data: dict, body: dict) -> tuple[str, str | None, dict, dict]:
        """(text, finish_reason, spent usage, stats) of one completion payload.

        The lab gateway translates to a Responses-style upstream and reports
        ``finish_reason="stop"`` even when the completion limit was hit; when the
        reported completion tokens reach the requested limit the finish reason is
        therefore normalised to ``"length"`` (counted as ``length_inferred``).
        """
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):   # some gateways return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        text = content if isinstance(content, str) else ""
        finish = choice.get("finish_reason")
        u = data.get("usage") or {}
        usage = empty_usage()
        usage["calls"] = 1
        stats = {"usage_estimated": 0, "length_inferred": 0}
        if u:
            usage["prompt_tokens"] = int(u.get("prompt_tokens") or 0)
            usage["completion_tokens"] = int(u.get("completion_tokens") or 0)
            details = u.get("completion_tokens_details") or {}
            usage["reasoning_tokens"] = int(details.get("reasoning_tokens") or 0)
            limit = int(body[_limit_key(body)])
            if finish in (None, "stop") and usage["completion_tokens"] >= limit:
                finish = "length"
                stats["length_inferred"] = 1
        else:
            stats["usage_estimated"] = 1
            usage["prompt_tokens"] = _approx_tokens(body["messages"])
            usage["completion_tokens"] = len(text) // 4
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        return text, finish, usage, stats

    def _post_with_retries(self, body: dict, role: str) -> tuple[dict, int]:
        attempts = int(self.cfg.max_retries) + 1
        for attempt in range(attempts):
            try:
                with self._sem:
                    return self._post_once(body), attempt
            except LLMError as e:
                e.attempts = attempt + 1
                limit_hit = e.retry_limit is not None and attempt >= e.retry_limit
                if not e.retryable or attempt == attempts - 1 or limit_hit:
                    if e.retryable:
                        e.args = (f"{e.args[0]} (gave up after {attempt + 1} attempts)",)
                    raise
                delay = self._backoff(attempt, e.retry_after)
                log.warning("LLM %s/%s attempt %d/%d failed (%s); retrying in %.2fs",
                            role, body["model"], attempt + 1, attempts, e.args[0], delay)
                self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        base = min(self._backoff_max, self._backoff_base * (2 ** attempt))
        with self._rng_lock:
            delay = base * (0.5 + 0.5 * self._rng.random())
        if retry_after is not None and retry_after > 0:
            delay = max(delay, min(retry_after, self._backoff_max))
        return delay

    def _credentials(self) -> tuple[str, str]:
        """(base_url, api_key) read fresh from the credentials file; never cached on self."""
        path = Path(self.cfg.credentials_file).expanduser()
        try:
            creds = json.loads(path.read_text())
        except FileNotFoundError:
            raise LLMError(f"credentials file not found: {path}") from None
        except (OSError, json.JSONDecodeError) as e:
            raise LLMError(f"cannot read credentials file {path}: {type(e).__name__}") from None
        base_url = (self.cfg.base_url or creds.get("base_url") or "").strip().rstrip("/")
        api_key = creds.get("api_key") or ""
        if not base_url:
            raise LLMError(f"no base_url in cfg.base_url or {path}")
        if not api_key:
            raise LLMError(f"no api_key in {path}")
        return base_url, api_key

    def _acquire_endpoint(self) -> str:
        now = time.monotonic()
        with self._ep_lock:
            up = [e for e in self._eps if self._ep_down_until[e] <= now] or [min(self._eps, key=self._ep_down_until.get)]
            ep = min(up, key=lambda e: self._ep_inflight[e])
            self._ep_inflight[ep] += 1
            return ep

    def _release_endpoint(self, ep: str, failed: bool) -> None:
        with self._ep_lock:
            self._ep_inflight[ep] -= 1
            if failed:
                self._ep_down_until[ep] = time.monotonic() + ENDPOINT_COOLDOWN_S

    def _build_request(self, body: dict, endpoint: str | None = None) -> tuple[urllib.request.Request, str]:
        if endpoint is not None:
            base_url, api_key = endpoint, "EMPTY"           # self-hosted server: the gateway key is never sent
        else:
            base_url, api_key = self._credentials()
        url = base_url if base_url.endswith("/chat/completions") else base_url + "/chat/completions"
        req = urllib.request.Request(
            url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), method="POST",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                     "Accept": "application/json", "User-Agent": self._user_agent})
        return req, urlparse(url).netloc

    def _redact_with_key(self, text: str) -> str:
        if self._eps:
            return _redact(text, None)
        try:
            secret = self._credentials()[1]
        except LLMError:
            secret = None
        return _redact(text, secret)

    def _post_once(self, body: dict) -> dict:
        if not self._eps:
            return self._post_to(body, None)
        ep = self._acquire_endpoint()
        failed = True
        try:
            data = self._post_to(body, ep)
            failed = False
            return data
        except LLMError as e:
            failed = e.retryable and (e.status is None or e.status >= 500)
            raise
        finally:
            self._release_endpoint(ep, failed)

    def _post_to(self, body: dict, endpoint: str | None) -> dict:
        req, host = self._build_request(body, endpoint)
        model = body["model"]
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            status = int(e.code)
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except (OSError, http.client.HTTPException) as read_err:
                err_body = f"<unreadable error body: {type(read_err).__name__}>"
            retry_after = _parse_retry_after(e.headers.get("Retry-After") if e.headers else None)
            retryable = status in RETRYABLE_STATUS or 500 <= status < 600
            retry_limit = None
            if not retryable and status not in (401, 403) and _is_transient_gateway_error(err_body):
                # The gateway also uses 4xx + "upstream_error" for some deterministic upstream
                # rejections, so these get only a couple of retries.
                retryable, retry_limit = True, GATEWAY_ERROR_RETRY_LIMIT
            raise LLMError(f"HTTP {status} from {host} (model={model})", status=status,
                           body=_truncate(self._redact_with_key(err_body)), retryable=retryable,
                           retry_after=retry_after, model=model, retry_limit=retry_limit) from None
        except urllib.error.URLError as e:
            reason = e.reason
            retryable = not isinstance(reason, ssl.SSLCertVerificationError)
            raise LLMError(f"connection error to {host}: {type(reason).__name__}: {reason}",
                           retryable=retryable, model=model) from None
        except (TimeoutError, socket.timeout) as e:
            raise LLMError(f"timeout after {self.cfg.timeout_s}s talking to {host}: {type(e).__name__}",
                           retryable=True, model=model) from None
        except (ConnectionError, http.client.HTTPException, OSError) as e:
            raise LLMError(f"connection error to {host}: {type(e).__name__}: {e}",
                           retryable=True, model=model) from None
        try:
            data = json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            raise LLMError(f"non-JSON response from {host}", body=_truncate(self._redact_with_key(
                raw.decode("utf-8", errors="replace"))), retryable=True, model=model) from None
        if (not isinstance(data, dict) or not isinstance(data.get("choices"), list) or not data["choices"]
                or not isinstance(data["choices"][0], dict)):
            raise LLMError(f"response from {host} has no choices", body=_truncate(self._redact_with_key(
                json.dumps(data, ensure_ascii=False)[:2000])), retryable=True, model=model)
        return data


def _error_response(role: str, tag: str, exc: LLMError) -> LLMResponse:
    """Placeholder response for a failed item of ``chat_many(return_exceptions=True)``."""
    return LLMResponse(text="", usage=empty_usage(), cached=False, latency_s=0.0, model=exc.model,
                       finish_reason="error", role=role, tag=tag, error=f"{type(exc).__name__}: {exc}")


def _collect(role: str, tag: str, futures: list[Future], return_exceptions: bool) -> list[LLMResponse]:
    out: list[LLMResponse] = []
    for f in futures:
        exc = f.exception()
        if exc is None:
            out.append(f.result())
        elif return_exceptions and isinstance(exc, LLMError):
            out.append(_error_response(role, tag, exc))
        else:
            raise exc
    return out


def _is_transient_gateway_error(body: str) -> bool:
    """True if an error body carries a transient gateway error code.

    The lab gateway reports upstream-provider failures as HTTP 400 with
    ``{"error": {"code": "upstream_error", ...}}``; those are transient and retried.
    """
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return False
    err = data.get("error") if isinstance(data, dict) else None
    if not isinstance(err, dict):
        return False
    return str(err.get("code") or "").lower() in TRANSIENT_ERROR_CODES


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None   # HTTP-date form is not worth parsing here; exponential backoff applies


# ------------------------------------------------------------------ extract_json

_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+\-]*)[ \t]*\r?\n?(.*?)```", re.DOTALL)
_PY_CONSTS = {"True": "true", "False": "false", "None": "null"}
_MAX_CANDIDATE_STARTS = 200


def _balanced_objects(text: str) -> Iterator[str]:
    """Lazily yield balanced ``{...}`` spans, in order of their opening brace (strings respected)."""
    tried = 0
    for start, first in enumerate(text):
        if first != "{":
            continue
        tried += 1
        if tried > _MAX_CANDIDATE_STARTS:
            return
        depth = 0
        quote: str | None = None
        escape = False
        for j in range(start, len(text)):
            ch = text[j]
            if quote is not None:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    quote = None
                continue
            if ch in ('"', "'"):
                quote = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[start:j + 1]
                    break


def _repair(candidate: str) -> str:
    """Outside string literals: drop trailing commas, map True/False/None to JSON."""
    out: list[str] = []
    i, n = 0, len(candidate)
    quote: str | None = None
    while i < n:
        ch = candidate[i]
        if quote is not None:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(candidate[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch == '"':
            quote = ch
            out.append(ch)
            i += 1
            continue
        if ch == ",":
            k = i + 1
            while k < n and candidate[k] in " \t\r\n":
                k += 1
            if k < n and candidate[k] in "}]":
                i += 1          # trailing comma
                continue
        if ch.isalpha() or ch == "_":
            k = i
            while k < n and (candidate[k].isalnum() or candidate[k] == "_"):
                k += 1
            word = candidate[i:k]
            out.append(_PY_CONSTS.get(word, word))
            i = k
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _try_parse(candidate: str) -> dict | None:
    s = candidate.strip()
    if not s:
        return None
    try:
        obj = json.loads(s, strict=False)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict):
        return obj
    try:
        obj = json.loads(_repair(s), strict=False)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict):
        return obj
    try:                                   # python-style dict literal (single quotes, True/None)
        obj = ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        obj = None
    return obj if isinstance(obj, dict) else None


def extract_json(text: str | None) -> dict | None:
    """Tolerantly extract the first JSON object from model output.

    Tries, in order: the whole text; the contents of ```json / ``` code fences;
    every balanced ``{...}`` span (by position of its opening brace). Each candidate
    is parsed strictly, then after removing trailing commas and mapping Python
    ``True/False/None`` to JSON, then as a Python dict literal. Control characters
    inside strings (raw newlines in code) are accepted. Returns ``None`` if nothing
    parses to a dict.
    """
    if not text or not isinstance(text, str):
        return None
    obj = _try_parse(text)
    if obj is not None:
        return obj
    for m in _FENCE_RE.finditer(text):
        body = m.group(2)
        obj = _try_parse(body)
        if obj is not None:
            return obj
        for span in _balanced_objects(body):
            obj = _try_parse(span)
            if obj is not None:
                return obj
    for span in _balanced_objects(text):
        obj = _try_parse(span)
        if obj is not None:
            return obj
    return None
