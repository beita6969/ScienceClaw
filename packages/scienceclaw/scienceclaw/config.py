"""Run configuration (YAML <-> dataclasses)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelRole:
    # Empty -> $SCIENCECLAW_<ROLE>_MODEL, then $SCIENCECLAW_MODEL (resolved when the first request is built).
    model: str = ""
    max_tokens: int = 6000                 # sent as max_completion_tokens for reasoning models
    temperature: float | None = 0.0
    reasoning_effort: str | None = "low"   # None for non-reasoning models
    json_mode: bool = False
    extra_body: dict = field(default_factory=dict)   # merged into the request body (e.g. vLLM chat_template_kwargs)


@dataclass
class LLMConfig:
    # Chat-model backend: "openai" (OpenAI-compatible HTTP, the default), "command" (a local program) or
    # "package.module:factory" for a custom implementation of scienceclaw.llm.interface.ChatModel.
    backend: str = "openai"
    command: str = ""                       # backend "command": program that reads the prompt on stdin (see llm/command.py)
    base_url: str = ""                      # empty -> $SCIENCECLAW_API_BASE_URL, then the credentials file
    # Self-hosted OpenAI-compatible servers (vLLM). When non-empty they REPLACE the gateway: requests are spread over
    # these base URLs (least in-flight, failing ones cooled down) and the credentials file is never read or sent.
    endpoints: list[str] = field(default_factory=list)
    # {"base_url","api_key"}; never copied anywhere. $SCIENCECLAW_API_BASE_URL / $SCIENCECLAW_API_KEY take precedence.
    credentials_file: str = "~/.config/scienceclaw/credentials.json"
    policy: ModelRole = field(default_factory=lambda: ModelRole(json_mode=True, max_tokens=8000))
    executor: ModelRole = field(default_factory=lambda: ModelRole(max_tokens=2000))
    patch: ModelRole = field(default_factory=lambda: ModelRole(max_tokens=4000))
    concurrency: int = 16
    timeout_s: float = 240.0
    max_retries: int = 6
    cache_path: str = "cache/llm_cache.sqlite"
    use_cache: bool = True


@dataclass
class SolverConfig:
    max_steps: int = 24
    history_window: int = 6                 # last k steps shown in full; older ones summarized
    orchestration: str = "canvas"          # canvas | single_turn | single_operator | fixed_workflow
    replay_on_new_submit: bool = True
    show_dev_score: bool = True
    retrieve_skills_k: int = 4
    retrieve_ops_k: int = 6
    stop_on_first_pass: bool = True         # source mode: stop at the first replay-verified pass


@dataclass
class EvolutionConfig:
    variant: str = "full"                  # frozen | workflow_only | skill_only | operator_only | unlinked | full
    update_schedule: str = "per_candidate" # per_candidate | per_round_argmax
    qval: str = "macrosr_then_score"       # macrosr | macrosr_then_score | score
    qval_eps: float = 0.02                 # min normalized-score gain when MacroSR ties (≥ LLM re-solve noise)
    hval_mode: str = "no_regression"       # no_regression | absolute
    pass_requires_acceptance: bool = True
    instances_per_episode: int = 1
    breplay_repeats: int = 1
    # B: budget on the D_val cost vector (DESIGN decision 11). Tokens are LOGICAL (spent + cached). B_abs is
    # `budget_tokens` if > 0, else `budget_tokens_per_val_episode` x |D_val|; a candidate must ALSO satisfy
    # C_val(cand) <= (1 + budget_beta) x C_val(incumbent) (budget_beta < 0 disables the relative rule).
    budget_tokens: float = 0.0             # explicit absolute cap on D_val logical tokens (0 = use the per-episode budget)
    budget_tokens_per_val_episode: float = 250_000.0
    budget_beta: float = 0.5
    budget_wall_s: float = 0.0             # explicit absolute cap on summed D_val wall time (0 = per-episode budget)
    budget_wall_s_per_val_episode: float = 1_800.0
    lazy_revalidation: bool = True
    # Noise guard on Eq. 3 (DESIGN decision 12): the Q_val gain must be supported by >= min_improved_episodes val
    # episodes that individually improved, and by at most max_regressed_episodes regressed ones (-1 = no cap).
    # min_improved_episodes: 1 restores the literal "any Q_val gain" rule (and disables the regression cap).
    min_improved_episodes: int = 2
    max_regressed_episodes: int = 1
    # Gateway outages (DESIGN decision 13, SolveResult.infra_error): bounded retries of a val / source solve, doubling backoff.
    infra_retries: int = 2
    infra_backoff_s: float = 30.0


@dataclass
class RunConfig:
    name: str = "dev"
    runs_root: str = "runs"
    llm: LLMConfig = field(default_factory=LLMConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True))


def _build(cls, data: dict | None, base=None):
    """Build ``cls`` from a (possibly partial) mapping.

    Missing fields keep the value of ``base`` (default: ``cls()``), so a partial nested mapping such as
    ``llm: {policy: {model: x}}`` only overrides ``model`` and keeps the other fields of the *default
    instance* (e.g. ``policy.json_mode = True``), not the bare nested dataclass defaults.
    """
    data = data or {}
    base = cls() if base is None else base
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        v = data[f.name]
        ft = f.type if not isinstance(f.type, str) else eval(f.type, globals())  # noqa: S307 - local dataclass names only
        if is_dataclass(ft) and isinstance(v, dict):
            kwargs[f.name] = _build(ft, v, getattr(base, f.name))
        else:
            kwargs[f.name] = v
    return replace(base, **kwargs)


_ENV_FIELDS = (("runs_root",), ("llm", "cache_path"), ("llm", "base_url"),
               ("llm", "policy", "model"), ("llm", "executor", "model"), ("llm", "patch", "model"))


def _expand_env(data: dict) -> None:
    """Expand ``$VAR`` / ``${VAR}`` in paths, endpoints and model names so a config stays machine independent."""
    for keys in _ENV_FIELDS:
        cur: Any = data
        for k in keys[:-1]:
            cur = cur.get(k) if isinstance(cur, dict) else None
        if isinstance(cur, dict) and isinstance(cur.get(keys[-1]), str):
            cur[keys[-1]] = os.path.expandvars(cur[keys[-1]])
    llm = data.get("llm")
    if isinstance(llm, dict) and isinstance(llm.get("endpoints"), list):
        llm["endpoints"] = [os.path.expandvars(e) if isinstance(e, str) else e for e in llm["endpoints"]]


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> RunConfig:
    data: dict = {}
    if path:
        data = yaml.safe_load(Path(path).read_text()) or {}
    for dotted, value in (overrides or {}).items():
        cur = data
        parts = dotted.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value
    _expand_env(data)
    return _build(RunConfig, data)
