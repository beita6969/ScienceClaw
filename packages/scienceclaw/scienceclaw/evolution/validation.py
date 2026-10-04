"""Source-task replay R_src (Eq. 13) and the program-level validation gate (Eq. 2-3).

* ``source_replay_check``: a fresh reset solve of the source episode with the candidate program A~_i
  (mode "source"); R_src = Pass(e_src) and Use(omega_i). Both conjuncts are evaluated against the *same*
  evidence e_src, the passing replay-verified evidence (the solver's final one, else the first): an
  Operator is used iff an operator node with that ref is in e_src's graph and ran ok in e_src's trace; a
  Skill is used iff it is in ``uses`` of an *effective* step (action applied OK, step <= e_src.step).
  Uses of rejected / failed actions, of failing evidence graphs and of later steps do not count.
* ``ValidationGate.evaluate``: solves every D_val episode in mode "val" (concurrently), with lazy
  re-validation (DESIGN decision 8): an episode whose retrieval-visible program slice is unchanged w.r.t.
  the reference report reuses its stored entry. A solve cut short by a gateway outage
  (``SolveResult.infra_error``, DESIGN decision 13) is retried (bounded, with backoff) and, if it still fails, becomes an
  *error entry* (never a silent z=0), which blocks H_val["no_solver_error"] and is re-solved by the
  evolver's incumbent refresh.
* ``ValidationGate.admit``: Eq. 2 feasibility (H_val all true, C_val within the budget B; R_src is checked
  upstream) and Eq. 3 strict improvement of Q_val (DESIGN decision 2), plus the noise guard of DESIGN
  decision 12 (``min_improved_episodes``).

Budget B (DESIGN decision 11): the token cost is *logical* (spent + cached, cache independent) and must
satisfy BOTH (a) C_val <= B_abs, where B_abs = ``budget_tokens`` if > 0 else
``budget_tokens_per_val_episode`` x |D_val|, and (b) the relative rule C_val(cand) <= (1 + ``budget_beta``)
x C_val(incumbent) (skipped when ``budget_beta`` < 0 or the incumbent cost is 0). Wall time is checked
against ``budget_wall_s`` (> 0) or ``budget_wall_s_per_val_episode`` x |D_val|.

H_val (DESIGN decision 3, see ``hval_vector``): (a) integrity = no sandbox/leakage violation (static scan
of the code nodes of every final val graph + evaluator-reported violations), (b) schema = completed outputs
pass the output-schema constraints (``SCHEMA_CONSTRAINT_PATTERN``), (c) hard constraints, plus
``no_solver_error``. In ``hval_mode="absolute"`` (a)-(c) must hold on every *completed* val episode
(unsolved episodes only lower MacroSR through z=0); in the default ``"no_regression"`` mode none of (a)-(c)
may *newly* fail relative to the incumbent, episode by episode (the literal absolute values are always
reported as ``h_absolute``).

Report fields (``ValReport.per_episode[id]``): z, norm_score, h_ok, hard_violations, completed,
schema_ok, integrity_violations, error, cost, slice_hash, reused, discipline, run_dir.
"""
from __future__ import annotations

import copy
import logging
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from ..core.program import AgentProgram

__all__ = ["ValReport", "ValidationGate", "source_replay_check", "source_pass", "source_evidence", "use_check",
           "used_refs", "infra_error_of", "logical_cost", "entry_tokens", "improved_regressed",
           "norm_ref", "episode_entry", "aggregate_report", "hval_vector", "absolute_h", "new_hard_violations",
           "new_integrity_violations", "new_schema_failures", "qval_key", "QVAL_MODES", "HVAL_MODES",
           "SCHEMA_CONSTRAINT_PATTERN"]

log = logging.getLogger(__name__)

QVAL_MODES = ("macrosr", "macrosr_then_score", "score")
HVAL_MODES = ("no_regression", "absolute")
_TIE = 1e-12


# ------------------------------------------------------------------------------------ Eq. 13 R_src
def norm_ref(ref: Any) -> str:
    """"skill:x@v2" -> "skill:x"; "op:y@v1" -> "op:y" (version ids are unique per program)."""
    return str(ref).split("@v", 1)[0]


def _graph_nodes(g: Any) -> Iterable[Any]:
    if g is None:
        return []
    if isinstance(g, dict):
        from ..core.graph import WorkflowGraph

        g = WorkflowGraph.from_dict(g)
    return list(getattr(g, "nodes", {}).values())


def source_evidence(result: Any) -> Any | None:
    """e_src: the passing replay-verified evidence of a source solve (the solver's final one, else the first)."""
    passing = [e for e in (getattr(result, "evidence", None) or []) if bool(getattr(e, "passed", False))]
    if not passing:
        return None
    fs = getattr(result, "final_step", None)
    if fs is not None and getattr(result, "final_source", "pass") == "pass":
        for e in passing:
            if getattr(e, "step", None) == fs:
                return e
    return passing[0]


def _node_ran_ok(trace: Any, nid: str) -> bool:
    recs = getattr(trace, "records", None)
    if recs is None and isinstance(trace, dict):
        recs = trace.get("records")
    rec = (recs or {}).get(nid)
    if rec is None:
        return False
    status = rec.get("status") if isinstance(rec, dict) else getattr(rec, "status", None)
    return status == "ok"


def used_refs(result: Any) -> set[str]:
    """Refs used in the passing evidence e_src of a solve (version suffix stripped), see module docstring.

    * ``op:<id>``    - an operator node with that ref is in e_src's graph and ran ok in e_src's trace;
    * ``skill:<id>`` - listed in ``uses`` of an effective step (``feedback.action_ok`` is True) with
      ``step <= e_src.step``.

    Without any passing evidence the final graph / trace / step of the result stand in for e_src.
    """
    ev = source_evidence(result)
    if ev is not None:
        graph, trace, last = getattr(ev, "graph_dict", None), getattr(ev, "trace", None), getattr(ev, "step", None)
    else:
        graph, trace = getattr(result, "final_graph", None), getattr(result, "trace", None)
        last = getattr(result, "final_step", None)
    refs: set[str] = set()
    for rec in getattr(result, "steps", None) or []:
        fb = getattr(rec, "feedback", None) or {}
        fb = fb if isinstance(fb, dict) else getattr(fb, "__dict__", {})
        if fb.get("action_ok") is not True or fb.get("finish"):
            continue
        if last is not None and int(getattr(rec, "step", 0)) > int(last):
            continue
        refs |= {norm_ref(u) for u in (getattr(rec, "uses", None) or []) if str(u).startswith("skill:")}
    for n in _graph_nodes(graph):
        if getattr(n, "kind", None) == "operator" and getattr(n, "ref", None) and _node_ran_ok(trace, n.id):
            r = str(n.ref)
            refs.add(norm_ref(r if r.startswith("op:") else f"op:{r}"))
    return refs


def use_check(omega: list[str], result: Any) -> tuple[bool, list[str]]:
    """Use(omega; e_src): every new version id of the bundle is used in the passing evidence e_src.

    Returns (ok, missing).
    """
    used = used_refs(result)
    missing = [w for w in omega if norm_ref(w) not in used]
    return not missing, missing


def source_pass(result: Any, require_acceptance: bool = True) -> bool:
    """Pass(e_src): some replay-verified evidence passed, or the final evaluation passes."""
    if any(bool(getattr(e, "passed", False)) for e in (getattr(result, "evidence", None) or [])):
        return True
    ev = getattr(result, "eval", None)
    if ev is None:
        return False
    from ..bench.task import passes

    return passes(ev, require_acceptance)


def source_replay_check(candidate: AgentProgram, omega: list[str], episode: Any, solver: Any,
                        run_dir: str | Path, *, retries: int = 0, backoff_s: float = 0.0,
                        sleep: Callable[[float], None] = time.sleep) -> tuple[bool, Any]:
    """Eq. 13: R_src(A~) = Pass(e_src) and Use(omega) on a fresh reset solve of the source episode.

    A solve cut short by a gateway outage (``SolveResult.infra_error``) is repeated up to ``retries`` times
    (doubling ``backoff_s``) so that an outage does not reject a candidate; the last result is returned
    (its ``infra_error`` stays set if the outage persisted).
    """
    result = None
    for attempt in range(max(0, int(retries)) + 1):
        d = Path(run_dir) if attempt == 0 else _fresh_dir(Path(run_dir).with_name(Path(run_dir).name + "_retry"))
        d.mkdir(parents=True, exist_ok=True)
        result = solver.solve(episode, candidate, mode="source", run_dir=str(d))
        err = infra_error_of(result)
        if err is None:
            break
        log.warning("source replay of %s hit an infrastructure error (attempt %d): %s", episode.id, attempt + 1, err)
        if attempt < int(retries):
            sleep(float(backoff_s) * (2 ** attempt))
    evo = getattr(solver, "evo_cfg", None)
    req = bool(getattr(evo, "pass_requires_acceptance", True)) if evo is not None else True
    ok_use, _ = use_check(omega, result)
    return bool(source_pass(result, req) and ok_use), result


# ------------------------------------------------------------------------------------ val reports
@dataclass
class ValReport:
    program_version: str
    per_episode: dict[str, dict]
    macro_sr: float
    norm_score: float
    h_val: dict[str, bool]
    cost: dict
    # ---- additive fields
    incurred: dict = field(default_factory=dict)     # cost actually spent (re-solved episodes only)
    n_resolved: int = 0
    n_reused: int = 0
    wall_s: float = 0.0
    h_absolute: dict = field(default_factory=dict)   # literal (absolute) integrity / schema / all-hard checks
    budget: dict = field(default_factory=dict)       # B in force when the report was made (see ValidationGate)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ValReport":
        return cls(**d)

    def summary(self) -> dict:
        return {"program_version": self.program_version, "macro_sr": self.macro_sr, "norm_score": self.norm_score,
                "h_val": self.h_val, "cost": self.cost, "incurred": self.incurred, "n_resolved": self.n_resolved,
                "n_reused": self.n_reused}


def _num(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _fin(x: Any) -> Any:
    """JSON-safe number: +inf (no limit) becomes None."""
    return None if isinstance(x, float) and math.isinf(x) else x


def _finite(d: dict) -> dict:
    return {k: _fin(v) for k, v in d.items()}


def _numeric_usage(usage: Any) -> dict[str, float]:
    return {str(k): float(v) for k, v in dict(usage or {}).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))}


def _sum_costs(costs: Iterable[dict]) -> dict[str, float]:
    tot: dict[str, float] = {}
    for c in costs:
        for k, v in c.items():
            tot[k] = tot.get(k, 0.0) + float(v)
    return tot


def _integrity_violations(result: Any) -> list[str]:
    """Sandbox/leakage violations of the final val graph (static scan of code nodes) + evaluator-reported."""
    out: list[str] = []
    ev = getattr(result, "eval", None)
    out += [str(v) for v in (getattr(ev, "details", {}) or {}).get("integrity_violations", []) or []]
    try:
        from ..runtime.integrity import scan_code
    except ImportError:
        scan_code = None
    if scan_code is not None:
        for n in _graph_nodes(getattr(result, "final_graph", None)):
            if getattr(n, "kind", None) == "code" and getattr(n, "code", None):
                out += [f"{n.id}: {v}" for v in scan_code(n.code)]
    else:
        steps = getattr(result, "steps", None) or []
        if steps:
            fb = getattr(steps[-1], "feedback", None) or {}
            fb = fb if isinstance(fb, dict) else getattr(fb, "__dict__", {})
            out += [str(v) for v in fb.get("integrity_violations", []) or []]
    return out


# Hard constraints that check the *form* of the deliverable y (the submit port schema Gamma: type, shape,
# unit). Adapters name them e.g. "output_shape", "output_structure", "output_length", "declared_unit".
SCHEMA_CONSTRAINT_PATTERN = re.compile(
    r"(schema|^output_(shape|structure|length|format|type)$|^declared_unit$)", re.I)


def episode_entry(result: Any, episode: Any, slice_hash: str, run_dir: str = "") -> dict:
    """Per-episode val entry (JSON-safe) from a val-mode SolveResult."""
    ev = result.eval
    completed = bool(getattr(ev, "completed", False))
    h = {str(k): bool(v) for k, v in (getattr(ev, "h", {}) or {}).items()}
    details = getattr(ev, "details", {}) or {}
    return {
        "discipline": str(getattr(episode, "discipline", "")),
        "z": int(getattr(ev, "z", 0) or 0),
        "norm_score": _num(details.get("norm_score")) if completed else 0.0,
        "completed": completed,
        "h_ok": bool(completed and all(h.values())),
        "hard_violations": sorted(k for k, v in h.items() if not v) if completed else [],
        "schema_ok": (not completed) or all(v for k, v in h.items() if SCHEMA_CONSTRAINT_PATTERN.search(k)),
        "integrity_violations": _integrity_violations(result),
        "error": None,
        "cost": _numeric_usage(getattr(result, "usage", {})),
        "slice_hash": slice_hash,
        "reused": False,
        "run_dir": run_dir,
    }


def _failed_entry(episode: Any, slice_hash: str, error: str, run_dir: str, wall_s: float,
                  cost: dict | None = None) -> dict:
    return {"discipline": str(getattr(episode, "discipline", "")), "z": 0, "norm_score": 0.0, "completed": False,
            "h_ok": False, "hard_violations": [], "schema_ok": True, "integrity_violations": [], "error": error,
            "cost": {**(cost or {}), "wall_s": wall_s}, "slice_hash": slice_hash, "reused": False,
            "run_dir": run_dir}


def infra_error_of(result: Any) -> str | None:
    """The infrastructure error (gateway / LLM outage) that cut a solve short, or None."""
    err = getattr(result, "infra_error", None)
    return err if isinstance(err, str) and err else None


def entry_tokens(entry: dict) -> float:
    """Logical (spent + cached) tokens of one val entry; falls back to spent tokens for old entries."""
    c = entry.get("cost") or {}
    return _num(c["logical_tokens"]) if "logical_tokens" in c else _num(c.get("total_tokens"))


def logical_cost(rep: "ValReport") -> float:
    """C_val token component of a report: sum of the logical tokens of its entries (cache independent)."""
    return float(sum(entry_tokens(e) for e in rep.per_episode.values()))


def _macro(per_episode: dict[str, dict], key: str) -> float:
    by_d: dict[str, list[float]] = {}
    for e in per_episode.values():
        by_d.setdefault(e.get("discipline", ""), []).append(_num(e.get(key)))
    if not by_d:
        return 0.0
    return float(sum(sum(v) / len(v) for v in by_d.values()) / len(by_d))


def new_hard_violations(cand: ValReport, inc: ValReport) -> dict[str, list[str]]:
    """Per episode: hard constraints violated under ``cand`` but not under ``inc``."""
    out: dict[str, list[str]] = {}
    for eid, e in cand.per_episode.items():
        before = set((inc.per_episode.get(eid) or {}).get("hard_violations", []) or [])
        new = sorted(set(e.get("hard_violations", []) or []) - before)
        if new:
            out[eid] = new
    return out


def new_integrity_violations(cand: ValReport, inc: ValReport) -> dict[str, list[str]]:
    """Episodes with integrity violations under ``cand`` whose incumbent entry had none."""
    return {eid: list(e.get("integrity_violations") or []) for eid, e in cand.per_episode.items()
            if e.get("integrity_violations") and not (inc.per_episode.get(eid) or {}).get("integrity_violations")}


def new_schema_failures(cand: ValReport, inc: ValReport) -> list[str]:
    """Episodes whose completed output fails the output-schema constraint under ``cand`` but not ``inc``."""
    out = []
    for eid, e in cand.per_episode.items():
        if e.get("completed") and not e.get("schema_ok", True):
            before = inc.per_episode.get(eid) or {}
            if not (before.get("completed") and not before.get("schema_ok", True)):
                out.append(eid)
    return out


def absolute_h(per_episode: dict[str, dict]) -> dict[str, bool]:
    """Literal H_val checks over the val episodes (DESIGN decision 3 (a), (b) and the absolute (c), the latter
    on *completed* episodes only)."""
    entries = list(per_episode.values())
    return {"integrity": all(not e.get("integrity_violations") for e in entries),
            "schema": all(e.get("schema_ok", True) for e in entries if e.get("completed")),
            # hard constraints are only defined on a produced output; an unsolved episode has none and only
            # lowers MacroSR through z=0 ("Unsolved tasks lower Q_val, hard-constraint violations make H_val != 1")
            "all_hard_ok": all(e.get("h_ok") for e in entries if e.get("completed"))}


def hval_vector(rep: ValReport, hval_mode: str, reference: ValReport | None) -> dict[str, bool]:
    """H_val of ``rep``.

    ``absolute``: (a) no integrity violation, (b) every completed y passes the output-schema constraint,
    (c) every hard constraint holds on every val episode. ``no_regression`` (default): (a), (b) and (c)
    are required not to *newly* fail relative to ``reference`` (the incumbent), episode by episode, so
    that a failure the incumbent already has (and that an unchanged, reused val entry repeats) does not
    block every later candidate. Always: no val solve raised (``no_solver_error``).
    """
    entries = list(rep.per_episode.values())
    h = {"no_solver_error": all(not e.get("error") for e in entries)}
    ab = absolute_h(rep.per_episode)
    if hval_mode == "absolute":
        h.update(ab)
    elif hval_mode == "no_regression":
        if reference is None:
            h.update(integrity=ab["integrity"], schema=ab["schema"], no_new_hard_violation=True)
        else:
            h.update(integrity=not new_integrity_violations(rep, reference),
                     schema=not new_schema_failures(rep, reference),
                     no_new_hard_violation=not new_hard_violations(rep, reference))
    else:
        raise ValueError(f"unknown hval_mode {hval_mode!r}; expected one of {HVAL_MODES}")
    return h


def aggregate_report(program_version: str, per_episode: dict[str, dict], hval_mode: str,
                     reference: ValReport | None = None, wall_s: float = 0.0) -> ValReport:
    """MacroSR, mean normalized score (both macro over disciplines), H_val and costs of a report."""
    entries = list(per_episode.values())
    rep = ValReport(program_version=program_version, per_episode=per_episode,
                    macro_sr=_macro(per_episode, "z"), norm_score=_macro(per_episode, "norm_score"),
                    h_val={}, cost=_sum_costs(e.get("cost", {}) for e in entries),
                    incurred=_sum_costs(e.get("cost", {}) for e in entries if not e.get("reused")),
                    n_resolved=sum(1 for e in entries if not e.get("reused")),
                    n_reused=sum(1 for e in entries if e.get("reused")), wall_s=wall_s,
                    h_absolute=absolute_h(per_episode))
    rep.h_val = hval_vector(rep, hval_mode, reference)
    return rep


def improved_regressed(cand: ValReport, inc: ValReport, mode: str, eps: float) -> tuple[list[str], list[str]]:
    """Val episodes that individually improved / regressed under ``cand`` relative to ``inc``.

    ``macrosr``: z 0 -> 1 / 1 -> 0. ``macrosr_then_score``: z changed, or (z equal and) the normalized score
    moved by at least ``eps``. ``score``: the normalized score moved by at least ``eps`` (either sign).
    Episodes missing from either report and episodes with a solver error do not count (they are handled by
    H_val ``no_solver_error``).
    """
    thr = eps if eps > 0 else _TIE
    better: list[str] = []
    worse: list[str] = []
    for eid, e in cand.per_episode.items():
        b = inc.per_episode.get(eid)
        if b is None or e.get("error") or b.get("error"):
            continue
        dz = int(e.get("z", 0) or 0) - int(b.get("z", 0) or 0)
        dn = _num(e.get("norm_score")) - _num(b.get("norm_score"))
        if mode == "macrosr":
            up, down = dz > 0, dz < 0
        elif mode == "macrosr_then_score":
            up, down = dz > 0 or (dz == 0 and dn >= thr), dz < 0 or (dz == 0 and dn <= -thr)
        else:  # "score"
            up, down = dn >= thr, dn <= -thr
        if up:
            better.append(eid)
        elif down:
            worse.append(eid)
    return sorted(better), sorted(worse)


def qval_key(rep: ValReport, mode: str) -> tuple[float, ...]:
    """Sort key of Q_val (for the per-round argmax)."""
    if mode == "macrosr":
        return (rep.macro_sr,)
    if mode == "macrosr_then_score":
        return (rep.macro_sr, rep.norm_score)
    if mode == "score":
        return (rep.norm_score,)
    raise ValueError(f"unknown qval {mode!r}; expected one of {QVAL_MODES}")


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(name))[:120] or "_"


def _fresh_dir(base: Path) -> Path:
    d, i = base, 1
    while d.exists():
        i += 1
        d = base.with_name(f"{base.name}__{i}")
    d.mkdir(parents=True)
    return d


class ValidationGate:
    """Eq. 2 feasibility + Eq. 3 strict improvement on D_val (thread-safe evaluate)."""

    def __init__(self, evo_cfg: Any, solver_cfg: Any, solver: Any, val_episodes: Any, run_dir: str | Path, *,
                 max_workers: int = 6, retriever_factory: Callable[[AgentProgram], Any] | None = None) -> None:
        if getattr(evo_cfg, "qval", "macrosr_then_score") not in QVAL_MODES:
            raise ValueError(f"unknown qval {evo_cfg.qval!r}; expected one of {QVAL_MODES}")
        if getattr(evo_cfg, "hval_mode", "no_regression") not in HVAL_MODES:
            raise ValueError(f"unknown hval_mode {evo_cfg.hval_mode!r}; expected one of {HVAL_MODES}")
        self.evo_cfg = evo_cfg
        self.solver_cfg = solver_cfg
        self.solver = solver
        if isinstance(val_episodes, dict):
            val_episodes = [ep for d in sorted(val_episodes) for ep in val_episodes[d]]
        self.val_episodes: list[Any] = list(val_episodes)
        ids = [ep.id for ep in self.val_episodes]
        if len(set(ids)) != len(ids):
            raise ValueError("validation episodes must have unique ids")
        self.run_dir = Path(run_dir)
        self.max_workers = max(1, int(max_workers))
        self._retriever_factory = retriever_factory
        self._lock = threading.Lock()
        self._sleep: Callable[[float], None] = time.sleep   # replaced in tests

    # -------------------------------------------------------------------------------- evaluation
    def _retriever(self, program: AgentProgram) -> Any:
        if self._retriever_factory is not None:
            return self._retriever_factory(program)
        from ..core.retrieval import Retriever

        return Retriever(program)

    def _solve_one(self, program: AgentProgram, ep: Any, slice_hash: str, base: Path) -> dict:
        t0 = time.monotonic()
        retries = max(0, int(getattr(self.evo_cfg, "infra_retries", 0) or 0))
        backoff = float(getattr(self.evo_cfg, "infra_backoff_s", 0.0) or 0.0)
        res, err, d = None, None, base
        for attempt in range(retries + 1):
            with self._lock:   # directory allocation is the only shared mutable state
                d = _fresh_dir(base / _safe(ep.id))
            try:
                res = self.solver.solve(ep, program, mode="val", run_dir=str(d))
            except Exception as ex:  # counted as an unsolved episode and flagged in H_val["no_solver_error"]
                log.exception("validation solve of %s with program %s failed", ep.id, program.version)
                return _failed_entry(ep, slice_hash, f"{type(ex).__name__}: {ex}"[:500], str(d),
                                     round(time.monotonic() - t0, 3))
            err = infra_error_of(res)
            if err is None:
                return episode_entry(res, ep, slice_hash, str(d))
            log.warning("validation solve of %s (program %s) hit an infrastructure error (attempt %d/%d): %s",
                        ep.id, program.version, attempt + 1, retries + 1, err)
            if attempt < retries:
                self._sleep(backoff * (2 ** attempt))
        # A gateway outage must never become a silent z=0: return an error entry (H_val no_solver_error = False;
        # the evolver's incumbent refresh re-solves errored incumbent entries).
        return _failed_entry(ep, slice_hash, f"infrastructure error after {retries + 1} attempt(s): {err}"[:500],
                             str(d), round(time.monotonic() - t0, 3), cost=_numeric_usage(getattr(res, "usage", {})))

    def evaluate(self, program: AgentProgram, reuse_from: ValReport | None = None) -> ValReport:
        """Solve D_val with ``program`` (mode "val"), reusing unchanged-slice entries of ``reuse_from``."""
        t0 = time.monotonic()
        retr = self._retriever(program)
        k_s = int(getattr(self.solver_cfg, "retrieve_skills_k", 4))
        k_o = int(getattr(self.solver_cfg, "retrieve_ops_k", 6))
        lazy = bool(getattr(self.evo_cfg, "lazy_revalidation", True))
        entries: dict[str, dict] = {}
        jobs: list[tuple[Any, str]] = []
        for ep in self.val_episodes:
            sh = retr.slice_hash(ep, k_s, k_o)
            old = reuse_from.per_episode.get(ep.id) if (lazy and reuse_from is not None) else None
            if old is not None and old.get("slice_hash") == sh and not old.get("error"):
                e = copy.deepcopy(old)
                e["reused"] = True
                e["reused_from"] = old.get("reused_from") or reuse_from.program_version
                entries[ep.id] = e
            else:
                jobs.append((ep, sh))
        base = self.run_dir / _safe(program.version)
        if jobs:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(jobs)),
                                    thread_name_prefix="val") as pool:
                futs = {pool.submit(self._solve_one, program, ep, sh, base): ep.id for ep, sh in jobs}
                for f in as_completed(futs):
                    entries[futs[f]] = f.result()
        ordered = {ep.id: entries[ep.id] for ep in self.val_episodes}
        rep = aggregate_report(program.version, ordered, self.evo_cfg.hval_mode, reference=reuse_from,
                               wall_s=round(time.monotonic() - t0, 3))
        rep.budget = _finite(self.budget_for(len(ordered)))
        log.info("val %s: %d re-solved, %d reused of %d episodes", program.version, rep.n_resolved, rep.n_reused,
                 len(ordered))
        return rep

    # ---------------------------------------------------------------------------------- Eq. 2-3
    def budget_for(self, n_val: int) -> dict:
        """The budget B in force for a D_val of ``n_val`` episodes (explicit absolute values win over scaled ones)."""
        e = self.evo_cfg
        abs_tok = float(getattr(e, "budget_tokens", 0.0) or 0.0)
        per_tok = float(getattr(e, "budget_tokens_per_val_episode", 0.0) or 0.0)
        abs_wall = float(getattr(e, "budget_wall_s", 0.0) or 0.0)
        per_wall = float(getattr(e, "budget_wall_s_per_val_episode", 0.0) or 0.0)
        tokens = abs_tok if abs_tok > 0 else (per_tok * n_val if per_tok > 0 else math.inf)
        wall = abs_wall if abs_wall > 0 else (per_wall * n_val if per_wall > 0 else math.inf)
        return {"n_val_episodes": n_val, "tokens": tokens, "wall_s": wall,
                "tokens_source": "budget_tokens" if abs_tok > 0 else "budget_tokens_per_val_episode",
                "wall_source": "budget_wall_s" if abs_wall > 0 else "budget_wall_s_per_val_episode",
                "beta": float(getattr(e, "budget_beta", -1.0))}

    def feasible(self, cand: ValReport, inc: ValReport) -> tuple[bool, dict]:
        """Eq. 2 (without R_src, checked upstream): H_val(A) = 1 and C_val(A) within budget B.

        H_val is recomputed against ``inc`` (see ``hval_vector``); the literal absolute checks are reported
        in ``reasons["h_absolute"]``. The token cost is logical (spent + cached) and must be within the
        absolute budget (scaled with |D_val|, or the explicit ``budget_tokens``) AND within
        ``(1 + budget_beta)`` x the incumbent's logical cost; wall time within its absolute budget.
        """
        mode = self.evo_cfg.hval_mode
        h = hval_vector(cand, mode, inc)
        details: dict[str, Any] = {}
        if mode == "no_regression":
            details = {"new_hard_violations": new_hard_violations(cand, inc),
                       "new_integrity_violations": new_integrity_violations(cand, inc),
                       "new_schema_failures": new_schema_failures(cand, inc)}
        b = self.budget_for(len(cand.per_episode))
        tokens = logical_cost(cand)
        inc_tokens = logical_cost(inc)
        wall = _num(cand.cost.get("wall_s"))
        beta = b["beta"]
        rel_cap = (1.0 + beta) * inc_tokens if (beta >= 0 and inc_tokens > 0) else math.inf
        ok_abs = tokens <= b["tokens"]
        ok_rel = tokens <= rel_cap
        ok_wall = wall <= b["wall_s"]
        within = bool(ok_abs and ok_rel and ok_wall)
        violated = [n for n, ok in (("tokens_abs", ok_abs), ("tokens_rel", ok_rel), ("wall", ok_wall)) if not ok]
        h_ok = all(h.values())
        reasons = {"h_val": h, "h_ok": h_ok, "h_absolute": absolute_h(cand.per_episode), "hval_mode": mode,
                   **details, "cost_tokens": tokens, "cost_tokens_incumbent": inc_tokens,
                   "cost_total_tokens": _num(cand.cost.get("total_tokens")), "cost_wall_s": wall,
                   "budget": _finite(b), "budget_tokens": _fin(b["tokens"]), "budget_wall_s": _fin(b["wall_s"]),
                   "budget_tokens_rel": _fin(rel_cap), "within_budget_abs": ok_abs, "within_budget_rel": ok_rel,
                   "within_budget_wall": ok_wall, "within_budget": within, "budget_violated": violated}
        return bool(h_ok and within), reasons

    def improves(self, cand: ValReport, inc: ValReport) -> tuple[bool, dict]:
        """Eq. 3 strict improvement of Q_val (DESIGN decision 2) plus the noise guard (DESIGN decision 12).

        The literal rule admits any Q_val gain. With ``min_improved_episodes`` = m > 1 the gain must also be
        supported by at least m val episodes that individually improved (``improved_regressed``) and at most
        ``max_regressed_episodes`` (>= 0; -1 = unlimited) episodes that regressed. ``min_improved_episodes: 1``
        restores the literal rule. No extra LLM draws are used: the comparison is over the same entries.
        """
        mode = self.evo_cfg.qval
        eps = float(getattr(self.evo_cfg, "qval_eps", 0.0))
        d_sr = cand.macro_sr - inc.macro_sr
        d_ns = cand.norm_score - inc.norm_score
        score_better = (d_ns >= eps) if eps > 0 else (d_ns > _TIE)
        if mode == "macrosr":
            gain = d_sr > _TIE
        elif mode == "macrosr_then_score":
            gain = d_sr > _TIE or (abs(d_sr) <= _TIE and score_better)
        elif mode == "score":
            gain = score_better
        else:
            raise ValueError(f"unknown qval {mode!r}; expected one of {QVAL_MODES}")
        m = int(getattr(self.evo_cfg, "min_improved_episodes", 1))
        max_reg = int(getattr(self.evo_cfg, "max_regressed_episodes", -1))
        better, worse = improved_regressed(cand, inc, mode, eps)
        supported = True
        if m > 1:
            supported = len(better) >= m and (max_reg < 0 or len(worse) <= max_reg)
        ok = bool(gain and supported)
        return ok, {"qval": mode, "qval_eps": eps, "cand_macro_sr": cand.macro_sr, "inc_macro_sr": inc.macro_sr,
                    "cand_norm_score": cand.norm_score, "inc_norm_score": inc.norm_score,
                    "delta_macro_sr": d_sr, "delta_norm_score": d_ns, "q_gain": bool(gain),
                    "min_improved_episodes": m, "max_regressed_episodes": max_reg, "improved_episodes": better,
                    "regressed_episodes": worse, "gain_supported": bool(supported), "improved": ok}

    def admit(self, cand: ValReport, inc: ValReport) -> tuple[bool, dict]:
        """Eq. 2 + Eq. 3: admit ``cand`` over incumbent ``inc`` iff feasible and strictly better."""
        f_ok, f_reasons = self.feasible(cand, inc)
        i_ok, i_reasons = self.improves(cand, inc)
        ok = bool(f_ok and i_ok)
        return ok, {**f_reasons, **i_reasons, "feasible": f_ok, "admitted": ok,
                    "cand_version": cand.program_version, "inc_version": inc.program_version}
