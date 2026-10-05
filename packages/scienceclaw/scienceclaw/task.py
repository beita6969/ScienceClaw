"""Task specification D_t = (D_T, D_V, D_E) as executable episodes, plus the adapter interface."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable

from .core.schema import PortSchema
from .core.trace import Trace

SPLITS = ("src", "val", "id", "ood", "rep")


@dataclass
class Budget:
    max_steps: int = 12
    max_policy_tokens: int = 200_000
    max_wall_s: float = 1800.0
    max_node_s: float = 300.0
    max_llm_items: int = 256

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToolSpec:
    """A D_E tool. Runs in the parent process (trusted adapter code) and returns visible data only."""
    name: str
    description: str
    inputs: dict[str, PortSchema]
    outputs: dict[str, PortSchema]
    fn: Callable[[dict, dict], dict]
    config_doc: str = ""

    def signature(self) -> str:
        ins = ", ".join(f"{k}: {v.render()}" for k, v in self.inputs.items()) or ""
        outs = ", ".join(f"{k}: {v.render()}" for k, v in self.outputs.items())
        cfg = f"  config: {self.config_doc}" if self.config_doc else ""
        ports = "".join(f"\n    {side} {k}: {' '.join(v.description.split())}"
                        for side, group in (("in", self.inputs), ("out", self.outputs))
                        for k, v in group.items() if v.description)
        return f"tool:{self.name}({ins}) -> ({outs})\n    {self.description}{cfg}{ports}"


@dataclass
class ConstraintSpec:
    """A D_V hard constraint checked on the final output y (and its trace)."""
    name: str
    description: str
    check: Callable[[Any, Trace | None], tuple[bool, str]]
    visible: bool = True
    grade: Callable[[Any], float] | None = None      # optional graded share in [0, 1] of the criterion (live tasks)


@dataclass
class EvalResult:
    metrics: dict[str, float] = field(default_factory=dict)
    primary: float | None = None
    direction: str = "max"
    h: dict[str, bool] = field(default_factory=dict)
    h_msgs: dict[str, str] = field(default_factory=dict)
    cost: dict[str, float] = field(default_factory=dict)
    accepted: bool = False
    z: int = 0
    completed: bool = False
    reproducible: bool | None = None
    within_budget: bool = True
    details: dict = field(default_factory=dict)

    def hard_ok(self) -> bool:
        return all(self.h.values())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EvalResult":
        return cls(**d)


@dataclass
class Probe:
    """A hidden re-run of the submitted graph on a derived episode (inputs perturbed by the adapter).

    Used for properties that cannot be read off ``y`` alone, e.g. FoR42 causality: the same graph is run on stays cut at
    hidden hours and ``check`` compares the prefix predictions with the full-stay predictions it captured.
    ``overrides`` are the ``Episode`` fields replaced in the derived episode (typically ``tools``, ``constraints``,
    ``required_output``); ``check(y')`` returns ``(ok, message)`` for the output of the re-run.
    """
    name: str
    overrides: dict[str, Any]
    check: Callable[[Any], tuple[bool, str]]


@dataclass
class Episode:
    id: str
    discipline: str
    family: str
    split: str
    task_type: str
    objective: str
    required_output: PortSchema
    tools: list[ToolSpec]
    constraints: list[ConstraintSpec]
    budget: Budget = field(default_factory=Budget)
    lineage: dict = field(default_factory=dict)
    acceptance: str = ""
    tolerance: dict = field(default_factory=lambda: {"rtol": 1e-6, "atol": 1e-8})
    tags: list[str] = field(default_factory=list)
    metric: str = ""
    direction: str = "max"
    n_items: int = 0
    _evaluate: Callable[[Any, Trace | None], EvalResult] | None = field(default=None, repr=False)
    _dev_evaluate: Callable[[Any], dict] | None = field(default=None, repr=False)
    _probes: Callable[[Any], list[Probe]] | None = field(default=None, repr=False)     # y -> hidden graph re-runs

    def tool(self, name: str) -> ToolSpec | None:
        return next((t for t in self.tools if t.name == name), None)

    def probe_specs(self, y: Any) -> list[Probe]:
        """The adapter's hidden re-runs for output ``y`` (empty when the episode has none or ``y`` is unusable)."""
        if self._probes is None or y is None:
            return []
        try:
            return list(self._probes(y))
        except Exception:
            return []

    def run_probes(self, y: Any, trace: Trace | None,
                   runner: Callable[[Episode, str], tuple[Any, Any]]) -> dict[str, dict]:
        """Run every hidden probe of ``y`` and record ``{name: {"ok", "msg"}}`` in ``trace.probes``.

        ``runner(derived_episode, name)`` must execute the *same graph* that produced ``y`` on the derived episode and
        return ``(y', trace')`` (e.g. ``lambda ep, name: replay(graph, ep, program, llm, run_dir / name)``). A graph
        that fails or returns nothing on the derived inputs fails the probe. The verdicts are read by the adapter's
        hidden constraints in :meth:`evaluate`, so call this *before* :meth:`evaluate`. Without a call those
        constraints report "not probed" and pass (episodes without probes are unaffected).
        """
        out: dict[str, dict] = {}
        for p in self.probe_specs(y):
            derived = replace(self, id=f"{self.id}#{p.name}", _evaluate=None, _dev_evaluate=None, _probes=None,
                              **p.overrides)
            try:
                y_p, _ = runner(derived, p.name)
                ok, msg = (False, "no output on the derived inputs") if y_p is None else p.check(y_p)
            except Exception as ex:
                ok, msg = False, f"graph failed on the derived inputs: {type(ex).__name__}: {ex}"
            out[p.name] = {"ok": bool(ok), "msg": str(msg)}
        if trace is not None and out:
            trace.probes = {**(getattr(trace, "probes", None) or {}), **out}
        return out

    def check_constraints(self, y: Any, trace: Trace | None, visible_only: bool = False) -> tuple[dict[str, bool], dict[str, str]]:
        h: dict[str, bool] = {}
        msgs: dict[str, str] = {}
        for c in self.constraints:
            if visible_only and not c.visible:
                continue
            try:
                ok, msg = c.check(y, trace)
            except Exception as ex:
                ok, msg = False, f"constraint check raised {type(ex).__name__}: {ex}"
            h[c.name] = bool(ok)
            msgs[c.name] = msg
        return h, msgs

    def _failure_probe(self, trace: Trace | None) -> EvalResult | None:
        """The adapter's own verdict on *no output* (``_evaluate(None, trace)``), or None if it has none/raises.

        Adapters answer ``y=None`` with their uniform failure payload (reference-level or worse; DESIGN 7 "failure
        semantics"), so an episode that produced nothing still contributes exactly one payload to the pooled metric.
        """
        if self._evaluate is None:
            return None
        try:
            return self._evaluate(None, trace)
        except Exception:
            return None

    def failure_result(self, msg: str = "no output", trace: Trace | None = None) -> EvalResult:
        """EvalResult of an episode with no (usable) output: z=0, ``details['failed']`` and the failure payload.

        A held-out episode must never silently drop out of the pooled metric. The payload is the adapter's
        failure payload (its reference/inaction payload), so pooling stays over the full episode set and a method that
        fails cannot look better than the reference on the remaining episodes.
        """
        h = {c.name: False for c in self.constraints}
        msgs = {c.name: msg for c in self.constraints}
        res = EvalResult(primary=None, direction=self.direction, h=h, h_msgs=msgs, completed=False)
        probe = self._failure_probe(trace)
        if probe is not None:
            res.metrics = dict(probe.metrics or {})
            res.details = dict(probe.details or {})
            res.primary = probe.primary
        self._mark_failed(res, msg)
        return res

    @staticmethod
    def _mark_failed(res: EvalResult, msg: str) -> None:
        d = res.details
        d["failed"] = True
        d.setdefault("invalid", msg)
        d["norm_score"] = 0.0
        if d.get("pooled_payload") is None and d.get("reference_payload") is not None:
            d["pooled_payload"] = d["reference_payload"]
        res.accepted, res.z = False, 0

    def evaluate(self, y: Any, trace: Trace | None) -> EvalResult:
        """Hidden-label evaluation Eval_{D_V,t}(y, τ). Hard constraints are always included in h.

        Every call returns a result that carries ``details['pooled_payload']`` whenever the adapter has one, including
        for ``y is None`` and for malformed outputs (``details['failed'] = True``, see :meth:`failure_result`).
        """
        if y is None:
            return self.failure_result("no output", trace)
        if self._evaluate is None:
            raise RuntimeError(f"episode {self.id} has no evaluator")
        res = self._evaluate(y, trace)
        h, msgs = self.check_constraints(y, trace)
        res.h = {**h, **res.h}
        res.h_msgs = {**msgs, **res.h_msgs}
        res.direction = self.direction
        res.completed = True
        d = res.details
        no_score = res.primary is None or math.isnan(float(res.primary))     # +-inf is a legitimate (perfect) score
        if no_score:
            # an evaluator that produced no usable score: pool the failure payload, never drop the episode
            if d.get("pooled_payload") is None:
                probe = self._failure_probe(trace)
                pd = (probe.details or {}) if probe is not None else {}
                d["pooled_payload"] = next((v for v in (pd.get("pooled_payload"), pd.get("reference_payload"),
                                                        d.get("reference_payload")) if v is not None), None)
            d.setdefault("invalid", "no usable score")
            self._mark_failed(res, "no usable score")
        res.z = int(res.completed and res.hard_ok() and res.accepted and res.within_budget)
        return res

    def dev_evaluate(self, y: Any) -> dict | None:
        """Visible dev scoring (only on visible data). Returned dict is shown to the policy."""
        if self._dev_evaluate is None or y is None:
            return None
        try:
            return self._dev_evaluate(y)
        except Exception as ex:
            return {"error": f"{type(ex).__name__}: {ex}"}

    def public_view(self) -> dict:
        """What the policy may see about the episode (no evaluator, no hidden data)."""
        return {"discipline": self.discipline, "task_type": self.task_type, "objective": self.objective,
                "required_output": self.required_output.to_dict(), "tools": [t.signature() for t in self.tools],
                "constraints": [f"{c.name}: {c.description}" for c in self.constraints if c.visible],
                "budget": self.budget.to_dict(), "tags": self.tags}


def passes(ev: EvalResult, require_acceptance: bool = True) -> bool:
    """Pass_t(e) — convergence/validity/reproducibility within budget (+ acceptance if required)."""
    ok = ev.completed and ev.hard_ok() and ev.within_budget and (ev.reproducible is not False)
    return bool(ok and (ev.accepted or not require_acceptance))
