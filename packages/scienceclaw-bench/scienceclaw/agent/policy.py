"""Policy π_Θ0 (paper Eq. 6): samples (a_{t,k}, ν_{t,k}) from the fixed foundation model.

The policy is a thin, stateless wrapper around ``llm.chat(role="policy", ...)`` plus the action parser
(:func:`scienceclaw.core.actions.parse_action`). If the reply cannot be parsed, the model is re-asked once
*within the same step* with the parse error appended; the usage of both calls is summed. The foundation
model is never trained or otherwise modified here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from .prompts import PARSE_RETRY_TEMPLATE

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import SolverConfig
    from ..core.actions import Action

__all__ = ["Policy", "empty_usage", "add_usage", "response_usage", "logical_tokens", "USAGE_KEYS"]

ParseFn = Callable[[str], "tuple[Action | None, str | None]"]

# Same convention as ``llm.client.LLMResponse.usage``: the plain keys count *spent* (non-cached) work,
# ``cached_*`` keys count work served from the response cache.
USAGE_KEYS = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "calls",
              "cached_calls", "cached_prompt_tokens", "cached_completion_tokens", "cached_reasoning_tokens")


def empty_usage() -> dict:
    """Zero usage record: spent tokens/calls, cached tokens/calls, latency and number of attempts."""
    u: dict[str, Any] = {k: 0 for k in USAGE_KEYS}
    u["latency_s"] = 0.0
    u["attempts"] = 0
    return u


def response_usage(resp: Any) -> dict:
    """Usage record of one LLM response.

    Follows the ``LLMClient`` convention (spent keys + ``cached_*`` keys). A client that flags
    ``cached=True`` but reports its tokens only under the plain keys is normalized: the tokens are moved
    to the ``cached_*`` keys so that cost accounting never counts cache hits as spent work.
    """
    raw = dict(getattr(resp, "usage", None) or {})
    cached = bool(getattr(resp, "cached", False))
    u = empty_usage()
    for k in USAGE_KEYS:
        u[k] = int(raw.get(k, 0) or 0)
    if "calls" not in raw and "cached_calls" not in raw:
        u["calls"] = 1
    if cached and not any(k.startswith("cached_") for k in raw):
        for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "calls"):
            u["cached_" + k] += u[k]
            u[k] = 0
    u["latency_s"] = float(getattr(resp, "latency_s", 0.0) or 0.0)
    u["attempts"] = 1
    return u


def add_usage(a: dict, b: dict) -> dict:
    """Element-wise sum of two usage records (unknown numeric keys are summed too)."""
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = out.get(k, 0) + v
    return out


def logical_tokens(usage: dict) -> int:
    """Prompt + completion tokens irrespective of caching (spent + cached).

    Used for the policy-token *budget*, so that a run served from the response cache stops at exactly the
    same step as the original run.
    """
    return int(usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
               + usage.get("cached_prompt_tokens", 0) + usage.get("cached_completion_tokens", 0))


def _default_parser(allow_batch: bool) -> ParseFn:
    from ..core.actions import parse_action  # lazy: core.actions is owned by another module

    def parse(text: str) -> "tuple[Action | None, str | None]":
        return parse_action(text, allow_batch=allow_batch)

    return parse


class Policy:
    """π_Θ0(· | D_t, G_{t,k}, H_{t,k}, R_{t,r}) — one proposal per step.

    Args:
        llm: an ``LLMClient`` or ``FakeLLM`` (anything with ``chat(role, messages, *, json_mode=..., tag=...)``).
        cfg: the :class:`~scienceclaw.config.SolverConfig`. JSON mode follows the LLM role config
            (``LLMConfig.policy.json_mode``) unless ``cfg`` has an explicit ``policy_json_mode`` attribute.
        parser: optional ``text -> (Action | None, error | None)``; defaults to ``core.actions.parse_action``.
        max_reasks: number of re-asks after a parse error within the same step (paper setup: 1).
    """

    def __init__(self, llm: Any, cfg: "SolverConfig | None" = None, parser: ParseFn | None = None,
                 *, max_reasks: int = 1) -> None:
        self.llm = llm
        self.cfg = cfg
        self._parser = parser
        self.max_reasks = max(0, int(max_reasks))
        self.json_mode: bool | None = getattr(cfg, "policy_json_mode", None) if cfg is not None else None
        # "batch" actions are part of the action language only in single-turn orchestration; elsewhere a
        # batch is a parse error that is re-asked within the step.
        self.allow_batch: bool = getattr(cfg, "orchestration", "canvas") == "single_turn"

    def _parse(self, text: str) -> tuple["Action | None", str | None]:
        parser = self._parser or _default_parser(self.allow_batch)
        try:
            action, err = parser(text)
        except Exception as ex:  # a parser crash is reported to the policy like any parse error
            return None, f"action parser raised {type(ex).__name__}: {ex}"
        if action is None and not err:
            err = "no action found in the reply"
        return action, err

    def propose(self, system: str, messages: list[dict], *, tag: str = "policy",
                cache_salt: str = "") -> tuple["Action | None", str | None, str, dict]:
        """Ask the policy model for one action.

        Returns ``(action, parse_error, raw_text, usage)``; ``action`` is None iff ``parse_error`` is set.
        ``raw_text`` is the text of the last attempt; ``usage`` sums all attempts (see :func:`empty_usage`).
        LLM transport errors (after the client's own retries) propagate to the caller.
        """
        convo = [{"role": "system", "content": system}, *messages]
        usage = empty_usage()
        raw = ""
        action = None
        err: str | None = None
        for attempt in range(self.max_reasks + 1):
            resp = self.llm.chat("policy", convo, json_mode=self.json_mode, cache_salt=cache_salt, tag=tag)
            usage = add_usage(usage, response_usage(resp))
            raw = str(getattr(resp, "text", "") or "")
            action, err = self._parse(raw)
            if action is not None:
                return action, None, raw, usage
            if attempt < self.max_reasks:
                convo = convo + [
                    {"role": "assistant", "content": raw or "(empty reply)"},
                    {"role": "user", "content": PARSE_RETRY_TEMPLATE.format(error=err)},
                ]
        return None, err, raw, usage
