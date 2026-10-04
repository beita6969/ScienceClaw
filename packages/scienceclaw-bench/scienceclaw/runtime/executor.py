"""Execution-guided orchestration runtime: Execute_{D_E}(G, chi, a) -> (G', chi', f)  (paper Eq. 7).

* :class:`Checkpoint` chi maps node fingerprints to completed :class:`NodeRecord` s whose output values
  are pickled under ``<values_dir>/<fingerprint>/<port>.pkl``. After an edit only nodes whose
  fingerprint is not in chi -- i.e. edited nodes and their descendants -- are (re)executed.
* :class:`Executor` executes a :class:`WorkflowGraph` in topological waves (independent nodes of a
  wave run concurrently, up to ``max_parallel_subprocs``):

  - ``tool``     trusted D_E adapter function, in-process, with a timeout guard thread;
  - ``code``     static integrity scan, then a subprocess worker (:mod:`.sandbox`);
  - ``llm``      prompt template mapped over ``inputs["items"]`` via ``llm.chat_many(role="executor")``;
  - ``operator`` the Operator body expanded inline (nested execution with boundary maps) with its
                 contract kappa checked before and after (violations are diagnostics, not crashes);
  - ``submit``   passes its input ``y`` through; y is checked against the required output schema.

  A node whose required inputs are not wired, or whose upstream is pending, is ``pending``; a node
  downstream of a failed / skipped node is ``skipped``. Edge unit conversions are applied when
  values are fed. Every executed node records ``input_refs`` / ``output_refs`` (pickle paths) --
  boundary replay (paper Eq. 12) restores recorded inputs from them.
* :class:`Feedback` f_{t,k} is what the policy sees: action result, validation and integrity
  errors, per-node status / errors / stdout tails / output summaries, contract violations,
  *visible* constraint checks on y and the *visible* dev score. It never contains hidden scores
  (the hidden evaluator is never called by the runtime).

The executor is not re-entrant: use one instance per solve (per thread).
"""
from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import hashlib
import json
import math
import pickle
import re
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..core.actions import (
    LLM_ITEMS_PORT,
    LLM_OUTPUT_PORT,
    Action,
    apply_action_detailed,
    extract_json_object,
)
from ..core.graph import SUBMIT_PORT, Node, WorkflowGraph
from ..core.operators import OperatorSpec, check_contract_entries
from ..core.schema import PortSchema, apply_conversion, summarize_value, value_has_type
from ..core.trace import NodeRecord, Trace
from .integrity import scan_code
from .sandbox import run_code_node
from .values import save_value

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

MAX_OPERATOR_DEPTH = 6
ERROR_CHARS = 4000
DEFAULT_MAX_NODE_S = 300.0
DEFAULT_MAX_LLM_ITEMS = 256
DEFAULT_MAX_LLM_PROMPT_CHARS = 32_000      # rendered prompt of one llm request (item + shared ports)
DEFAULT_LLM_MAX_TOKENS = 2000              # used when the client exposes no executor role config
LLM_CHUNK_ITEMS = 64                       # llm items are sent in chunks so a node deadline can stop the rest
OPERATOR_DEADLINE_FACTOR = 2.0             # an operator node as a whole gets this x max_node_s
_SAFE_PORT = re.compile(r"[A-Za-z0-9_.-]{1,64}")
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)(?:![rsa])?(?::[^{}]*)?\}")
# a placeholder with optional accessors: {name}, {name:.2f}, {name[0]}, {name.attr}
_FIELD_REF = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)((?:\.[A-Za-z_][A-Za-z0-9_]*|\[[^\]{}]*\])*)(?:![rsa])?(?::[^{}]*)?\}")
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_FENCE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\n?(.*?)```", re.S)
_JSON = json.JSONDecoder(strict=False)

# Interface rules for llm nodes (deliverable format only; no task strategy).
EXECUTOR_SYSTEM_PROMPT = ("You execute one `llm` node of a scientific workflow on a single input item. "
                          "Follow the node instruction for this item only.")
_FORMAT_RULES = {
    "text": "Answer with the requested text only.",
    "json": "Answer with a single JSON value (usually an object) and nothing else.",
    "number": "Answer with a single number and nothing else.",
    "choice": "Answer with exactly one of the allowed options and nothing else: {choices}.",
}


# ============================================================================ data classes
@dataclass
class Checkpoint:
    """chi: node fingerprint -> completed NodeRecord (with ``output_refs`` to pickled values).

    ``salt`` identifies the operator library the records were produced with; an executor whose
    program has different operators does not reuse them.
    """
    records: dict[str, NodeRecord] = field(default_factory=dict)
    values_dir: str = ""
    salt: str = ""

    def copy(self) -> "Checkpoint":
        return Checkpoint(dict(self.records), self.values_dir, self.salt)

    def to_dict(self) -> dict:
        return {"records": {k: v.to_dict() for k, v in self.records.items()}, "values_dir": self.values_dir,
                "salt": self.salt}

    @classmethod
    def from_dict(cls, d: dict) -> "Checkpoint":
        return cls({k: NodeRecord.from_dict(v) for k, v in d.get("records", {}).items()}, d.get("values_dir", ""),
                   d.get("salt", ""))


def _fmt_num(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def format_summary(s: dict | None, max_head: int = 160) -> str:
    """One-line rendering of a :func:`summarize_value` dict."""
    if not s:
        return "-"
    if s.get("is_none"):
        return "None"
    parts = [str(s.get("py_type", "?"))]
    if "shape" in s:
        parts.append("shape=" + "x".join(str(d) for d in s["shape"]))
    if "len" in s and "shape" not in s:
        parts.append(f"len={s['len']}")
    if "dtype" in s:
        parts.append(f"dtype={s['dtype']}")
    if "unit" in s:
        parts.append(f"unit={s['unit']}")
    if "value" in s:
        parts.append(f"value={_fmt_num(s['value'])}")
    if s.get("finite") is False:
        parts.append("NON-FINITE")
    if "finite_frac" in s:
        parts.append(f"finite={s['finite_frac']:.3g}")
    if "min" in s and "max" in s:
        parts.append(f"range=[{_fmt_num(s['min'])}, {_fmt_num(s['max'])}]")
    if "mean" in s:
        parts.append(f"mean={_fmt_num(s['mean'])}")
    if "columns" in s:
        cols = s["columns"]
        parts.append("columns=" + json.dumps(cols[:12]) + ("..." if len(cols) > 12 else ""))
    if "keys" in s:
        parts.append("keys=" + json.dumps(s["keys"][:12]))
    if "item_keys" in s:
        parts.append("item_keys=" + json.dumps(s["item_keys"][:20]))
    if "summary_error" in s:
        parts.append("summary_error=" + str(s["summary_error"])[:120])
    for k in ("n_items", "parse_failures", "llm_errors", "llm_timed_out_items", "max_tokens_clamped_to"):
        if k in s:
            parts.append(f"{k}={s[k]}")
    for k in ("unresolved_placeholders", "parse_failure_examples"):
        if s.get(k):
            parts.append(f"{k}=" + json.dumps(s[k], ensure_ascii=False)[:200])
    if "head" in s:
        head = json.dumps(s["head"], default=str, ensure_ascii=False)
        parts.append("head=" + (head if len(head) <= max_head else head[:max_head] + "..."))
    return " ".join(parts)


def _clip_tail(text: str | None, n: int) -> str:
    if not text:
        return ""
    text = text.rstrip()
    return text if len(text) <= n else "..." + text[-(n - 3):]


def _indent(text: str, pad: str = "      ") -> str:
    return "\n".join(pad + ln for ln in text.splitlines())


@dataclass
class Feedback:
    """f_{t,k} plus action-level information (only visible information)."""
    step: int
    action_ok: bool
    action_error: str | None = None
    validation_errors: list[str] = field(default_factory=list)
    records: dict[str, NodeRecord] = field(default_factory=dict)     # node id -> record, topological order
    submit_ready: bool = False
    y_summary: dict | None = None
    visible_constraints: dict[str, list] = field(default_factory=dict)   # name -> [ok, msg]
    dev: dict | None = None
    integrity_violations: list[str] = field(default_factory=list)
    llm_usage: dict = field(default_factory=dict)
    wall_s: float = 0.0
    action_desc: str = ""
    node_runs: int = 0

    def to_dict(self) -> dict:
        return {"step": self.step, "action_ok": self.action_ok, "action_error": self.action_error,
                "validation_errors": list(self.validation_errors),
                "records": {k: v.to_dict() for k, v in self.records.items()},
                "submit_ready": self.submit_ready, "y_summary": self.y_summary,
                "visible_constraints": self.visible_constraints, "dev": self.dev,
                "integrity_violations": list(self.integrity_violations), "llm_usage": dict(self.llm_usage),
                "wall_s": self.wall_s, "action_desc": self.action_desc, "node_runs": self.node_runs}

    # ------------------------------------------------------------------ rendering
    def _render(self, err_chars: int, out_chars: int, ok_details: bool,
                scrub: Callable[[str], str] | None = None) -> str:
        S = scrub or (lambda t: t)       # applied to free text BEFORE it is clipped, so a cut never splits a path
        lines: list[str] = []
        desc = f" ({self.action_desc})" if self.action_desc else ""
        if self.action_ok:
            lines.append(f"Action{desc}: applied.")
        else:
            lines.append(f"Action{desc}: REJECTED, canvas unchanged.")
            lines.append(_indent(_clip_tail(S(self.action_error or ""), err_chars), "  "))
        extra_verrs = [v for v in self.validation_errors if v not in (self.action_error or "")]
        if extra_verrs:
            lines.append("Graph validation errors:")
            lines += [f"  - {S(v)}" for v in extra_verrs]
        if self.integrity_violations:
            lines.append("Integrity violations (code not executed):")
            lines += [f"  - {S(v)}" for v in self.integrity_violations[:20]]
        if self.records:
            lines.append("Execution (topological order):")
            for nid, r in self.records.items():
                cached = ", cached" if r.cached else ""
                head = f"  [{nid}] {r.kind or '?'} {r.status.upper() if r.status != 'ok' else 'ok'} ({r.wall_s:.2f}s{cached})"
                if r.status in ("pending", "skipped"):
                    lines.append(f"{head}: {S(r.error or '')}".rstrip(": "))
                    continue
                lines.append(head)
                if r.status == "error" and r.error:
                    lines.append("      error:")
                    lines.append(_indent(_clip_tail(S(r.error), err_chars), "        "))
                if r.status == "ok" and (ok_details or not r.cached):
                    for port, summ in r.outputs_summary.items():
                        lines.append(f"      out {port}: {format_summary(summ)}")
                for cv in r.contract_violations[:10]:
                    lines.append(f"      contract/schema: {S(str(cv))}")
                if out_chars and r.stdout_tail and (r.status == "error" or not r.cached):
                    lines.append("      stdout/stderr tail:")
                    lines.append(_indent(_clip_tail(S(r.stdout_tail), out_chars), "        "))
        else:
            lines.append("Execution: (empty canvas)")
        if self.submit_ready:
            lines.append(f"Submitted output y: {format_summary(self.y_summary)}")
        else:
            lines.append("Submitted output y: not available (no submit node, or its input is not computed yet)")
        if self.visible_constraints:
            lines.append("Visible constraint checks on y:")
            for name, (ok, msg) in self.visible_constraints.items():
                lines.append(f"  - {name}: {'PASS' if ok else 'FAIL'}" + (f" -- {msg}" if msg else ""))
        if self.dev is not None:
            lines.append("Dev score on visible data: " + json.dumps(self.dev, default=_json_default)[:600])
        return "\n".join(lines)

    def render(self, max_chars: int = 6000, scrub: Callable[[str], str] | None = None) -> str:
        """Text shown to the policy; progressively less detailed until it fits ``max_chars``.

        ``scrub`` (e.g. the solver's ``scrub_volatile``) is applied to every free-text field before it is clipped
        and to the assembled text before it is measured or truncated: a clip or the final cut can never split a
        run-specific path and leave an unscrubbed fragment (DESIGN decision 8).
        """
        text = ""
        for err_chars, out_chars, ok_details in ((1500, 800, True), (600, 250, True), (240, 0, False)):
            text = self._render(err_chars, out_chars, ok_details, scrub)
            if scrub is not None:
                text = scrub(text)
            if len(text) <= max_chars:
                return text
        return text[: max(0, max_chars - 26)] + "\n...[feedback truncated]"


def _json_default(x: Any) -> Any:
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


# ============================================================================ helpers
@dataclass
class _Outcome:
    status: str                                  # "ok" | "error"
    outputs: dict | None = None
    error: str | None = None
    stdout_tail: str = ""
    contract_violations: list[str] = field(default_factory=list)
    llm_usage: dict = field(default_factory=dict)
    transient: bool = False                      # do not cache (e.g. LLM API failure)
    timed_out: bool = False                      # this node exceeded its own time limit (retried once, see _run_node)
    diagnostics: dict = field(default_factory=dict)   # merged into the primary output summary


class _InputError(Exception):
    """Inputs of a node could not be assembled (load / unit conversion failure)."""


class _GraphRun:
    """State of one (possibly nested) graph execution."""

    def __init__(self, deadline: float | None = None) -> None:
        self.records: dict[str, NodeRecord] = {}
        self.refs: dict[str, dict[str, str]] = {}
        self.mem: dict[tuple[str, str], Any] = {}
        self.transient: set[str] = set()
        self.lock = threading.Lock()
        self.deadline = deadline          # time.monotonic() limit of an operator body (None: no limit)
        self.n_timeouts = 0               # nodes of this run that hit a time limit

    def get_value(self, nid: str, port: str) -> Any:
        from .values import load_value

        key = (nid, port)
        with self.lock:
            if key in self.mem:
                return self.mem[key]
            path = self.refs[nid][port]
        value = load_value(path)
        with self.lock:
            return self.mem.setdefault(key, value)


def add_usage(acc: dict, u: dict | None) -> dict:
    """Sum the numeric entries of ``u`` into ``acc`` (in place)."""
    for k, v in (u or {}).items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        acc[k] = acc.get(k, 0) + v
    return acc


def _safe_port(p: str) -> str:
    return p if _SAFE_PORT.fullmatch(p) and p not in (".", "..") else "p_" + hashlib.sha1(p.encode()).hexdigest()[:12]


def _exc_text(ex: BaseException) -> str:
    tb = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
    return _clip_tail(f"{type(ex).__name__}: {ex}\n{tb}", ERROR_CHARS)


def _call_with_timeout(fn: Any, args: tuple, timeout_s: float, name: str,
                       on_done: Any = None, on_timeout: Any = None) -> tuple[Any, BaseException | None, bool]:
    """Run ``fn(*args)`` in a daemon thread; returns (result, exception, timed_out).

    A timed-out call cannot be killed; its thread is abandoned (tools are trusted adapter code).
    ``on_done()`` runs in the worker thread when ``fn`` returns or raises -- also after the caller
    gave up -- so a resource can be held for as long as the abandoned call is really running;
    ``on_timeout(thread)`` is told about the abandoned thread.
    """
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["result"] = fn(*args)
        except BaseException as ex:  # noqa: BLE001 - forwarded to the caller and reported as node error
            box["exc"] = ex
        finally:
            if on_done is not None:
                on_done()

    th = threading.Thread(target=target, name=f"tool-{name}", daemon=True)
    try:
        th.start()
    except BaseException:        # noqa: BLE001 - the worker never ran: release what on_done would have released
        if on_done is not None:
            on_done()
        raise
    th.join(max(0.0, timeout_s))
    if th.is_alive():
        if on_timeout is not None:
            on_timeout(th)
        return None, None, True
    return box.get("result"), box.get("exc"), False


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_))


def _safe_summary(value: Any, schema: PortSchema | None = None) -> dict:
    """:func:`summarize_value` that can never raise (a summary problem must not discard a valid edit)."""
    try:
        return summarize_value(value, schema)
    except Exception as ex:  # noqa: BLE001 - e.g. OverflowError from np.asarray on huge ints
        return {"py_type": type(value).__name__, "summary_error": f"{type(ex).__name__}: {ex}"[:200]}


def schema_issues(value: Any, schema: PortSchema, port: str = SUBMIT_PORT) -> list[str]:
    """Visible check of a value against a port schema (type, shape, prob range)."""
    if value is None:
        return [f"{port}: value is None"]
    t = schema.type
    is_df = pd is not None and isinstance(value, pd.DataFrame)
    ok = value_has_type(value, t)
    issues: list[str] = []
    if not ok:
        issues.append(f"{port}: required type {t}, got {type(value).__name__}")
    if schema.shape is not None and t in ("array", "table", "series", "list", "any"):
        try:
            shp = list(value.shape) if is_df else list(np.shape(value))
        except (ValueError, TypeError):
            shp = None
        if shp is None:
            issues.append(f"{port}: value has no regular shape; required {list(schema.shape)}")
        else:
            bind: dict[str, int] = {}
            bad = len(shp) != len(schema.shape)
            for d, s in zip(schema.shape, shp):
                if isinstance(d, int) and d != s:
                    bad = True
                elif isinstance(d, str):
                    if bind.setdefault(d, s) != s:
                        bad = True
            if bad:
                issues.append(f"{port}: shape {shp} does not match required {list(schema.shape)}")
    if schema.dtype == "prob" and ok and t in ("array", "number", "series", "list", "table"):
        try:
            arr = np.asarray(value.select_dtypes("number") if is_df else value, dtype=float)
            if arr.size and (np.nanmin(arr) < 0 or np.nanmax(arr) > 1):
                issues.append(f"{port}: probabilities outside [0, 1]")
        except (ValueError, TypeError):
            issues.append(f"{port}: required dtype prob but values are not numeric")
    return issues


# --------------------------------------------------------------------------- llm nodes
def _to_text(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False, default=_json_default)
    return str(v)


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _template_context(item: Any, shared: dict | None) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    for k, v in (shared or {}).items():
        raw[str(k)] = v
    if isinstance(item, dict):
        raw.update({str(k): v for k, v in item.items()})
    raw["item"] = _to_text(item)
    return raw


def render_template(template: str, item: Any, shared: dict | None = None) -> str:
    """Fill ``{field}`` placeholders from the item (dict fields, or ``{item}`` for any item) and from
    the node's other input ports. Values are inserted as text (dicts / lists as JSON). Unknown
    placeholders are left as is; literal braces are tolerated; a format spec that does not fit the
    value (``{v:.2f}`` on a string) falls back to the plain text of the value."""
    raw = _template_context(item, shared)
    fmt_ctx = {k: (v if isinstance(v, (str, int, float, bool)) or _is_number(v) else _to_text(v)) for k, v in raw.items()}
    try:
        return template.format_map(_SafeDict(fmt_ctx))
    except (ValueError, IndexError, KeyError, AttributeError, TypeError):
        def sub(m: re.Match) -> str:
            name = m.group(1)
            if name not in fmt_ctx:
                return m.group(0)
            spec = re.search(r":([^{}]*)\}$", m.group(0))
            if spec is not None and "!" not in m.group(0):
                try:
                    return format(fmt_ctx[name], spec.group(1))
                except (ValueError, TypeError):
                    pass
            return _to_text(raw[name])

        return _PLACEHOLDER.sub(sub, template)


def template_fields(template: str) -> tuple[list[str], list[str]]:
    """(placeholder names in order of appearance, placeholders using attribute / index access).

    ``{{`` / ``}}`` escapes are ignored. Accessors such as ``{item[0]}`` or ``{doc.text}`` are not
    supported (values are inserted as text), so they are reported separately."""
    names: list[str] = []
    accessors: list[str] = []
    for m in _FIELD_REF.finditer(template.replace("{{", "").replace("}}", "")):
        if m.group(2):
            accessors.append(m.group(0))
        if m.group(1) not in names:
            names.append(m.group(1))
    return names, accessors


def _parse_number(text: str) -> float | None:
    """A single finite number from ``text``; None if there is none, or more than one distinct number."""
    t = text.replace("\u2212", "-").strip()
    if t.endswith("%") or re.search(r"\d\s*%", t):
        return None                       # percentages are ambiguous (80% = 80 or 0.8?): report a parse failure
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", t)                     # thousands separators
    try:
        v = float(t)
        return v if math.isfinite(v) else None
    except ValueError:
        pass
    t = re.sub(r"^\s*(?:\d+[.)]|\(\d+\))\s+", "", t)                 # list numbering "1. The answer is ..."
    vals: list[float] = []
    for m in _NUMBER.finditer(t):
        try:
            v = float(m.group())
        except ValueError:
            continue
        if math.isfinite(v):
            vals.append(v)
    return vals[0] if vals and len(set(vals)) == 1 else None


def parse_llm_text(text: str | None, mode: str, choices: list[str] | None = None) -> Any:
    """Parse one executor answer; None when it cannot be parsed in ``mode``."""
    t = (text or "").strip()
    if not t:
        return None
    if mode == "text":
        return t
    if mode == "json":
        m = _FENCE.search(t)
        body = m.group(1).strip() if m else t
        try:
            return _JSON.decode(body)
        except json.JSONDecodeError:
            return extract_json_object(t)
    if mode == "number":
        return _parse_number(t)
    if mode == "choice":
        opts = list(choices or [])
        norm = t.strip("`'\".:;!() \n\t").lower()
        for c in opts:
            if norm == c.lower():
                return c
        low = t.lower()
        hits = [c for c in opts if re.search(r"(?<![A-Za-z0-9])" + re.escape(c.lower()) + r"(?![A-Za-z0-9])", low)]
        return hits[0] if len(hits) == 1 else None
    return None


def _as_items(x: Any) -> list:
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if pd is not None and isinstance(x, pd.DataFrame):
        return x.to_dict("records")
    if pd is not None and isinstance(x, pd.Series):
        return x.tolist()
    raise TypeError(f"input port {LLM_ITEMS_PORT!r} must be a list (of dicts or strings), got {type(x).__name__}")


def _operators_salt(program: Any) -> str:
    ops = getattr(program, "operators", None) or {}
    blob = json.dumps({k: v.to_dict() for k, v in sorted(ops.items())}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ============================================================================ executor
class Executor:
    """Eq. 7 executor for one episode (D_E = episode tools, budget) and a fixed program A_r."""

    def __init__(self, episode: Any, program: Any, llm: Any, run_dir: str | Path, max_parallel_subprocs: int = 4) -> None:
        self.episode = episode
        self.program = program
        self.llm = llm
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.max_parallel = max(1, int(max_parallel_subprocs))
        budget = getattr(episode, "budget", None)
        self.max_node_s = float(getattr(budget, "max_node_s", DEFAULT_MAX_NODE_S))
        self.max_llm_items = int(getattr(budget, "max_llm_items", DEFAULT_MAX_LLM_ITEMS))
        self.tag = str(getattr(episode, "id", "") or "")
        self.required_output: PortSchema = getattr(episode, "required_output", None) or PortSchema()
        self._salt = _operators_salt(program)
        self.max_llm_prompt_chars = int(getattr(budget, "max_llm_prompt_chars", DEFAULT_MAX_LLM_PROMPT_CHARS))
        self._tool_lock = threading.Lock()      # tool calls run one at a time; the tool thread releases it itself
        self._tool_zombie: tuple[str, threading.Thread] | None = None   # timed-out tool call still running
        self._timeouts: dict[str, int] = {}                             # node fingerprint -> timeouts seen
        self._stats_lock = threading.Lock()
        self._last: tuple[str, Checkpoint, dict[str, NodeRecord], Any] | None = None
        self.stats = {"node_runs": 0, "cached": 0, "executions": 0}
        self.last_wall_s = 0.0

    # ------------------------------------------------------------------ public API
    def new_checkpoint(self) -> Checkpoint:
        values = self.run_dir / "values"
        values.mkdir(parents=True, exist_ok=True)
        return Checkpoint({}, str(values), self._salt)

    def execute(self, graph: WorkflowGraph, checkpoint: Checkpoint | None = None
                ) -> tuple[Checkpoint, dict[str, NodeRecord], Any]:
        """Execute ``graph`` reusing chi; returns (chi', records in topological order, y or None)."""
        t0 = time.monotonic()
        cp = self._prepare_checkpoint(checkpoint)
        with self._stats_lock:
            self.stats["executions"] += 1
        verrs = graph.validate()
        if verrs:
            records = self._invalid_records(graph, verrs)
            self._remember(graph, cp, records, None)
            self.last_wall_s = time.monotonic() - t0
            return cp, records, None
        run = self._execute_graph(graph, Path(cp.values_dir), cp, {}, 0, parallel=True)
        records = {nid: run.records[nid] for nid in graph.topo_order()}
        # A transient node (partial LLM failure, timeout) is re-run next time and may then return other
        # values; fingerprints depend on the spec, not on values, so its descendants must not be cached either.
        taint = run.transient | graph.descendants(run.transient)
        for rec in records.values():
            if rec.cached or rec.status not in ("ok", "error") or rec.node_id in taint:
                continue
            cp.records[rec.fingerprint] = rec
        y = None
        sub = graph.submit_node()
        if sub is not None and records[sub].status == "ok":
            y = run.get_value(sub, SUBMIT_PORT)
        self._remember(graph, cp, records, y)
        self.last_wall_s = time.monotonic() - t0
        return cp, records, y

    def apply(self, graph: WorkflowGraph, checkpoint: Checkpoint | None, action: Action, step: int
              ) -> tuple[WorkflowGraph, Checkpoint, Feedback, Any]:
        """Eq. 7: apply the atomic edit, execute affected nodes, return (G', chi', f, y)."""
        t0 = time.monotonic()
        new_graph, err, verrs = apply_action_detailed(graph, action, self.episode, self.program, step=step)
        if err is None:
            g = new_graph
            cp, records, y = self.execute(g, checkpoint)
            fresh = True
        else:
            g = graph
            cp, records, y, fresh = self._current_state(graph, checkpoint)
        fb = self._feedback(step, action, g, records, y, err, verrs, fresh, time.monotonic() - t0)
        return g, cp, fb, y

    def run_operator(self, op: OperatorSpec, inputs: dict) -> tuple[dict, NodeRecord]:
        """Execute one OperatorSpec in isolation from boundary input values (boundary replay)."""
        t0 = time.monotonic()
        fp = _operator_fingerprint(op, inputs)
        root = self.run_dir / "values" / fp
        input_refs: dict[str, str] = {}
        for p, v in inputs.items():
            path = root / "inputs" / f"{_safe_port(p)}.pkl"
            save_value(v, path)
            input_refs[p] = str(path)
        try:
            out = self._run_operator_spec(op, dict(inputs), root / "body", 0)
        except Exception as ex:  # noqa: BLE001 - executor bug surfaces as an operator error record
            out = _Outcome("error", error=_exc_text(ex))
        status, error = out.status, out.error
        outputs = out.outputs or {}
        if status == "ok":
            missing = [p for p in op.outputs if p not in outputs]
            if missing:
                status, error = "error", f"operator produced no value for boundary output(s) {missing}"
        output_refs: dict[str, str] = {}
        summaries: dict[str, dict] = {}
        if status == "ok":
            for p in op.outputs:
                path = root / f"{_safe_port(p)}.pkl"
                save_value(outputs[p], path)
                output_refs[p] = str(path)
                summaries[p] = _safe_summary(outputs[p], op.outputs[p])
        rec = NodeRecord(node_id=op.ref, fingerprint=fp, status=status, wall_s=time.monotonic() - t0,
                         error=_clip_tail(error, ERROR_CHARS) or None, stdout_tail=out.stdout_tail,
                         outputs_summary=summaries, contract_violations=list(out.contract_violations),
                         input_refs=input_refs, output_refs=output_refs, llm_usage=dict(out.llm_usage),
                         kind="operator", cached=False)
        return ({p: outputs[p] for p in op.outputs} if status == "ok" else {}), rec

    def usage(self) -> dict:
        with self._stats_lock:
            return dict(self.stats)

    # ------------------------------------------------------------------ state helpers
    def _prepare_checkpoint(self, checkpoint: Checkpoint | None) -> Checkpoint:
        if checkpoint is None:
            return self.new_checkpoint()
        cp = checkpoint.copy()
        if not cp.values_dir:
            cp.values_dir = str(self.run_dir / "values")
        Path(cp.values_dir).mkdir(parents=True, exist_ok=True)
        if cp.salt != self._salt:     # produced under a different operator library: do not reuse
            cp = Checkpoint({}, cp.values_dir, self._salt)
        return cp

    def _remember(self, graph: WorkflowGraph, cp: Checkpoint, records: dict[str, NodeRecord], y: Any) -> None:
        self._last = (graph.graph_fingerprint(), cp, records, y)

    def _current_state(self, graph: WorkflowGraph, checkpoint: Checkpoint | None
                       ) -> tuple[Checkpoint, dict[str, NodeRecord], Any, bool]:
        """State of an unchanged graph (after a rejected action) without re-running anything."""
        last = self._last
        if last is not None and checkpoint is last[1] and last[0] == graph.graph_fingerprint():
            recs = {k: (dataclasses.replace(r, cached=True) if r.status in ("ok", "error") else r)
                    for k, r in last[2].items()}
            return checkpoint, recs, last[3], False
        cp, records, y = self.execute(graph, checkpoint)
        return cp, records, y, True

    @staticmethod
    def _invalid_records(graph: WorkflowGraph, verrs: list[str]) -> dict[str, NodeRecord]:
        try:
            fps = graph.fingerprints()
        except (RecursionError, KeyError):   # cyclic or dangling graph: fingerprints are undefined
            fps = {}
        msg = "graph is invalid, nothing executed: " + "; ".join(verrs)
        return {nid: NodeRecord(nid, fps.get(nid, ""), "pending", error=msg, kind=n.kind)
                for nid, n in sorted(graph.nodes.items())}

    # ------------------------------------------------------------------ feedback
    def _feedback(self, step: int, action: Action, g: WorkflowGraph, records: dict[str, NodeRecord], y: Any,
                  err: str | None, verrs: list[str], fresh: bool, wall: float) -> Feedback:
        usage: dict = {}
        runs = 0
        if fresh:
            for r in records.values():
                if not r.cached and r.status in ("ok", "error"):
                    runs += 1
                    add_usage(usage, r.llm_usage)
        vis: dict[str, list] = {}
        dev = None
        if y is not None and self.episode is not None:
            trace = Trace(dict(records), list(records), dict(usage), wall, str(self.run_dir))
            h, msgs = self.episode.check_constraints(y, trace, visible_only=True)
            vis = {k: [bool(v), str(msgs.get(k, ""))] for k, v in h.items()}
            dev = self.episode.dev_evaluate(y)
        integrity = [f"{nid}: {v}" for nid, n in g.nodes.items() if n.kind == "code" for v in scan_code(n.code or "")]
        return Feedback(step=step, action_ok=err is None, action_error=err, validation_errors=list(verrs),
                        records=records, submit_ready=y is not None,
                        y_summary=_safe_summary(y, self.required_output) if y is not None else None,
                        visible_constraints=vis, dev=dev, integrity_violations=integrity, llm_usage=usage,
                        wall_s=wall, action_desc=action.describe() if hasattr(action, "describe") else action.type,
                        node_runs=runs)

    # ------------------------------------------------------------------ graph execution
    def _execute_graph(self, graph: WorkflowGraph, values_root: Path, cache: Checkpoint | None,
                       injected: dict[tuple[str, str], Any], depth: int, parallel: bool,
                       deadline: float | None = None) -> _GraphRun:
        run = _GraphRun(deadline)
        fps = graph.fingerprints()
        order = graph.topo_order()
        inj: dict[str, dict[str, Any]] = {}
        for (n, p), v in injected.items():
            inj.setdefault(n, {})[p] = v
        remaining = list(order)
        while remaining:
            wave = [n for n in remaining if all(p in run.records for p in graph.predecessors(n))]
            if not wave:   # cannot happen for a validated DAG
                raise RuntimeError(f"no executable wave among {remaining}")
            to_run: list[str] = []
            for nid in wave:
                node = graph.nodes[nid]
                state = self._input_state(graph, nid, run, inj.get(nid, {}))
                if state is not None:
                    run.records[nid] = NodeRecord(nid, fps[nid], state[0], error=state[1], kind=node.kind)
                    continue
                hit = self._cache_lookup(cache, fps[nid]) if cache is not None else None
                if hit is not None:
                    run.records[nid] = hit
                    run.refs[nid] = dict(hit.output_refs)
                    with self._stats_lock:
                        self.stats["cached"] += 1
                    continue
                to_run.append(nid)
            if parallel and self.max_parallel > 1 and len(to_run) > 1:
                with cf.ThreadPoolExecutor(max_workers=min(self.max_parallel, len(to_run)),
                                           thread_name_prefix="node") as pool:
                    futs = [pool.submit(self._run_node, graph, nid, fps[nid], run, inj.get(nid, {}), values_root, depth)
                            for nid in to_run]
                    for f in futs:
                        f.result()
            else:
                for nid in to_run:
                    self._run_node(graph, nid, fps[nid], run, inj.get(nid, {}), values_root, depth)
            remaining = [n for n in remaining if n not in run.records]
        return run

    @staticmethod
    def _input_state(graph: WorkflowGraph, nid: str, run: _GraphRun, injected: dict[str, Any]) -> tuple[str, str] | None:
        ins = graph.in_edges(nid)
        bad = sorted({e.src for e in ins if run.records[e.src].status in ("error", "skipped")})
        if bad:
            return "skipped", f"upstream node(s) {bad} failed or were skipped"
        pend = sorted({e.src for e in ins if run.records[e.src].status == "pending"})
        missing = [p for p in graph.missing_inputs(nid) if p not in injected]
        if pend or missing:
            parts = []
            if missing:
                parts.append(f"input port(s) {missing} not connected")
            if pend:
                parts.append(f"waiting on pending node(s) {pend}")
            return "pending", "; ".join(parts)
        return None

    @staticmethod
    def _cache_lookup(cache: Checkpoint, fp: str) -> NodeRecord | None:
        rec = cache.records.get(fp)
        if rec is None or rec.status not in ("ok", "error"):
            return None
        if rec.status == "ok" and not all(Path(p).exists() for p in rec.output_refs.values()):
            return None
        return dataclasses.replace(rec, cached=True)

    def _load_inputs(self, graph: WorkflowGraph, nid: str, fp: str, run: _GraphRun, injected: dict[str, Any],
                     values_root: Path) -> tuple[dict, dict[str, str]]:
        inputs: dict[str, Any] = {}
        refs: dict[str, str] = {}
        for e in graph.in_edges(nid):
            try:
                v = run.get_value(e.src, e.src_port)
            except (KeyError, OSError, EOFError, pickle.UnpicklingError) as ex:
                raise _InputError(f"could not load value of {e.src}.{e.src_port}: {type(ex).__name__}: {ex}") from ex
            conv = e.conversion_dict
            if conv:
                try:
                    v = apply_conversion(v, conv)
                except (TypeError, ValueError) as ex:
                    raise _InputError(f"unit conversion on edge {e.render()} failed: {type(ex).__name__}: {ex}") from ex
                path = values_root / fp / "inputs" / f"{_safe_port(e.dst_port)}.pkl"
                save_value(v, path)
                refs[e.dst_port] = str(path)
            else:
                refs[e.dst_port] = run.refs[e.src][e.src_port]
            inputs[e.dst_port] = v
        for p, v in injected.items():
            if p in inputs:
                continue
            path = values_root / fp / "inputs" / f"{_safe_port(p)}.pkl"
            save_value(v, path)
            refs[p] = str(path)
            inputs[p] = v
        return inputs, refs

    def _run_node(self, graph: WorkflowGraph, nid: str, fp: str, run: _GraphRun, injected: dict[str, Any],
                  values_root: Path, depth: int) -> None:
        node = graph.nodes[nid]
        t0 = time.monotonic()
        input_refs: dict[str, str] = {}
        try:
            inputs, input_refs = self._load_inputs(graph, nid, fp, run, injected, values_root)
        except _InputError as ex:
            out = _Outcome("error", error=str(ex))
        else:
            if run.deadline is not None and run.deadline - time.monotonic() <= 0:
                out = _Outcome("error", timed_out=True,
                               error="timeout: the enclosing operator exceeded its time limit before this node started")
            else:
                try:
                    out = self._dispatch(node, inputs, values_root, fp, depth, run.deadline,
                                         self._input_units(graph, nid))
                except Exception as ex:  # noqa: BLE001 - a node failure must never crash the executor; it is recorded
                    out = _Outcome("error", error=_exc_text(ex))
        if out.timed_out:
            # Timeouts depend on machine load: the node is retried once (its outcome and that of its
            # descendants are not cached the first time); a repeated timeout of the same spec is cached
            # like any other error so that a truly endless node does not cost max_node_s on every edit.
            with self._stats_lock:
                self._timeouts[fp] = self._timeouts.get(fp, 0) + 1
                out.transient = out.transient or self._timeouts[fp] < 2
            with run.lock:
                run.n_timeouts += 1
        status, error = out.status, out.error
        expected = [SUBMIT_PORT] if node.kind == "submit" else list(node.outputs)
        outputs = out.outputs if status == "ok" else None
        if status == "ok":
            if not isinstance(outputs, dict):
                status, error = "error", f"{node.kind} node returned {type(outputs).__name__}, expected a dict of output ports"
            else:
                missing = [p for p in expected if p not in outputs]
                if missing:
                    status, error = "error", (f"{node.kind} node returned no value for declared output port(s) "
                                              f"{missing}; got {sorted(map(str, outputs))[:20]}")
        output_refs: dict[str, str] = {}
        summaries: dict[str, dict] = {}
        if status == "ok":
            try:
                for p in expected:
                    path = values_root / fp / f"{_safe_port(p)}.pkl"
                    save_value(outputs[p], path)
                    output_refs[p] = str(path)
                    schema = self.required_output if node.kind == "submit" else node.outputs.get(p)
                    summaries[p] = _safe_summary(outputs[p], schema)
            except (pickle.PicklingError, TypeError, AttributeError, ValueError, OSError, RecursionError) as ex:
                status, error = "error", f"could not store output value: {type(ex).__name__}: {ex}"
                output_refs, summaries = {}, {}
            if status == "ok" and out.diagnostics and expected:
                summaries[expected[0]].update(out.diagnostics)
        rec = NodeRecord(node_id=nid, fingerprint=fp, status=status, wall_s=time.monotonic() - t0,
                         error=_clip_tail(error, ERROR_CHARS) or None, stdout_tail=out.stdout_tail or "",
                         outputs_summary=summaries, contract_violations=list(out.contract_violations),
                         input_refs=input_refs, output_refs=output_refs, llm_usage=dict(out.llm_usage),
                         kind=node.kind, cached=False)
        with run.lock:
            run.records[nid] = rec
            if status == "ok":
                run.refs[nid] = output_refs
                for p in expected:
                    run.mem[(nid, p)] = outputs[p]
            if out.transient:
                run.transient.add(nid)
        with self._stats_lock:
            self.stats["node_runs"] += 1

    # ------------------------------------------------------------------ node kinds
    @staticmethod
    def _input_units(graph: WorkflowGraph, nid: str) -> dict[str, str | None]:
        """Unit really delivered to each connected input port: the edge conversion target, else the unit of
        the source port (``None``: unspecified upstream, compatible with any unit like ``schema.compat``)."""
        units: dict[str, str | None] = {}
        for e in graph.in_edges(nid):
            conv = e.conversion_dict
            if conv:
                units[e.dst_port] = conv.get("to")
                continue
            src = graph.nodes.get(e.src)
            sch = src.outputs.get(e.src_port) if src is not None else None
            units[e.dst_port] = sch.unit if sch is not None else None
        return units

    def _dispatch(self, node: Node, inputs: dict, values_root: Path, fp: str, depth: int,
                  deadline: float | None = None, units: dict[str, str | None] | None = None) -> _Outcome:
        if node.kind == "tool":
            return self._run_tool(node, inputs, deadline)
        if node.kind == "code":
            return self._run_code(node, inputs, deadline)
        if node.kind == "llm":
            return self._run_llm(node, inputs, deadline)
        if node.kind == "operator":
            oid = (node.ref or "")[3:] if (node.ref or "").startswith("op:") else (node.ref or "")
            op = (getattr(self.program, "operators", None) or {}).get(oid)
            if op is None:
                return _Outcome("error", error=f"unknown operator {node.ref!r} in the current program")
            return self._run_operator_spec(op, inputs, values_root / fp / "body", depth, deadline, units)
        if node.kind == "submit":
            y = inputs.get(SUBMIT_PORT)
            return _Outcome("ok", {SUBMIT_PORT: y}, contract_violations=schema_issues(y, self.required_output))
        return _Outcome("error", error=f"unknown node kind {node.kind!r}")

    def _node_timeout(self, node: Node, deadline: float | None = None) -> float:
        """Seconds a node may run: config ``timeout_s`` (at most the budget's ``max_node_s``), and never
        past the ``deadline`` (time.monotonic) of an enclosing operator."""
        t = self.max_node_s
        cfg = node.config.get("timeout_s")
        if _is_number(cfg) and float(cfg) > 0:
            t = min(t, float(cfg))
        if deadline is not None:
            t = min(t, max(0.1, deadline - time.monotonic()))
        return t

    def _acquire_tool_lock(self, name: str, deadline: float | None) -> str | None:
        """Tools are trusted adapter code that need not be thread-safe: one call at a time. Returns an error
        text when the lock cannot be taken. A call that timed out keeps the lock until it really ends, so a
        new call is refused at once while such an abandoned call is still running."""
        limit = time.monotonic() + self.max_node_s
        if deadline is not None:
            limit = min(limit, deadline)
        while not self._tool_lock.acquire(timeout=0.05):
            z = self._tool_zombie
            if z is not None and z[1].is_alive():
                return (f"tool {name!r} was not started: the earlier call to tool {z[0]!r} timed out and is still "
                        "running (tool calls run one at a time)")
            if time.monotonic() >= limit:
                return f"timeout: tool {name!r} waited for the tool lock (another tool call is still running)"
        return None

    def _run_tool(self, node: Node, inputs: dict, deadline: float | None = None) -> _Outcome:
        spec = self.episode.tool(node.ref) if self.episode is not None and node.ref else None
        if spec is None:
            return _Outcome("error", error=f"unknown tool {node.ref!r} (not provided by this task)")
        busy = self._acquire_tool_lock(spec.name, deadline)
        if busy is not None:
            return _Outcome("error", error=busy, timed_out=busy.startswith("timeout"), transient=True)
        timeout = self._node_timeout(node, deadline)
        # the lock is released by the worker thread itself (also for an abandoned, timed-out call)
        result, exc, timed_out = _call_with_timeout(
            spec.fn, (dict(inputs), dict(node.config)), timeout, spec.name, on_done=self._tool_lock.release,
            on_timeout=lambda th: setattr(self, "_tool_zombie", (spec.name, th)))
        if timed_out:
            return _Outcome("error", error=f"timeout: tool {spec.name!r} exceeded {timeout:.1f} s", timed_out=True)
        if exc is not None:
            return _Outcome("error", error=f"tool {spec.name!r} raised {type(exc).__name__}: {exc}")
        if not isinstance(result, dict):
            return _Outcome("error", error=f"tool {spec.name!r} returned {type(result).__name__}, expected a dict")
        return _Outcome("ok", result)

    def _run_code(self, node: Node, inputs: dict, deadline: float | None = None) -> _Outcome:
        viol = scan_code(node.code or "")
        if viol:
            return _Outcome("error", error="integrity scan rejected this code (not executed):\n- " + "\n- ".join(viol))
        outputs, meta = run_code_node(node, inputs, self.run_dir, self._node_timeout(node, deadline))
        if meta.get("status") == "ok" and outputs is not None:
            return _Outcome("ok", outputs, stdout_tail=meta.get("stdout_tail") or "")
        return _Outcome("error", error=meta.get("error") or f"code node failed (status {meta.get('status')})",
                        stdout_tail=meta.get("stdout_tail") or "", timed_out=meta.get("status") == "timeout")

    def _llm_token_cap(self) -> int:
        """Largest ``max_tokens`` an llm node may request: the executor role's configured limit."""
        role = getattr(getattr(self.llm, "cfg", None), "executor", None)
        try:
            return max(1, int(getattr(role, "max_tokens")))
        except (TypeError, ValueError, AttributeError):
            return DEFAULT_LLM_MAX_TOKENS

    def _run_llm(self, node: Node, inputs: dict, deadline: float | None = None) -> _Outcome:
        if self.llm is None:
            return _Outcome("error", error="no LLM client is configured for llm nodes")
        try:
            items = _as_items(inputs.get(LLM_ITEMS_PORT))
        except TypeError as ex:
            return _Outcome("error", error=str(ex))
        if len(items) > self.max_llm_items:
            return _Outcome("error", error=(f"llm node has {len(items)} items but the task budget allows at most "
                                            f"{self.max_llm_items} items per llm node"))
        cfg = node.config
        mode = cfg.get("parse", "text")
        if mode not in _FORMAT_RULES:
            return _Outcome("error", error=f"unknown parse mode {mode!r}")
        choices = [str(c) for c in (cfg.get("choices") or [])]
        system = EXECUTOR_SYSTEM_PROMPT + " " + _FORMAT_RULES[mode].format(choices=", ".join(choices))
        shared = {k: v for k, v in inputs.items() if k != LLM_ITEMS_PORT}
        template = node.prompt or ""
        diagnostics: dict[str, Any] = {}
        # ---- template checks (RT-4): the item must reach the prompt; unresolved names are reported
        names, accessors = template_fields(template)
        if accessors:
            return _Outcome("error", error=(
                f"llm prompt uses attribute / index access {accessors[:3]}: placeholders insert the value as text, "
                "so write {name} (dict items expose their keys as {key}, any item as {item})"))
        item_fields = {str(k) for it in items if isinstance(it, dict) for k in it}
        known = set(map(str, shared)) | item_fields | {"item"}
        unresolved = [n for n in names if n not in known]
        uses_item = "item" in names or any(n in item_fields for n in names)
        if len(items) > 1 and not uses_item:
            hint = (f"; the items are dicts with keys {sorted(item_fields)[:12]}: use {{key}} or {{item}}"
                    if item_fields else "; use {item} for the item")
            miss = f" (unresolved placeholders: {unresolved})" if unresolved else ""
            return _Outcome("error", error=f"llm prompt does not use the item: all {len(items)} requests would be "
                                           f"identical{miss}{hint}")
        if unresolved:
            diagnostics["unresolved_placeholders"] = unresolved[:10]
        batch = [[{"role": "system", "content": system},
                  {"role": "user", "content": render_template(template, it, shared)}] for it in items]
        if not batch:
            return _Outcome("ok", {LLM_OUTPUT_PORT: []}, diagnostics={"n_items": 0, "parse_failures": 0, "llm_errors": 0})
        # ---- cost guards (RT-11): rendered prompt size and max_tokens
        sizes = [len(m[0]["content"]) + len(m[1]["content"]) for m in batch]
        worst = max(range(len(sizes)), key=sizes.__getitem__)
        if sizes[worst] > self.max_llm_prompt_chars:
            return _Outcome("error", error=(
                f"rendered llm prompt of item {worst} has {sizes[worst]} characters, above the limit of "
                f"{self.max_llm_prompt_chars}: pass only the fields the llm node needs (a shared input port is "
                "rendered in full into every item's prompt)"))
        kw: dict[str, Any] = {"tag": self.tag}
        if cfg.get("max_tokens") is not None:
            mt = cfg["max_tokens"]
            if not _is_number(mt) or float(mt) < 1:
                return _Outcome("error", error=f"llm config max_tokens must be a positive integer, got {mt!r}")
            cap = self._llm_token_cap()
            kw["max_tokens"] = min(int(mt), cap)
            if int(mt) > cap:
                diagnostics["max_tokens_clamped_to"] = cap
        if cfg.get("temperature") is not None:
            kw["temperature"] = float(cfg["temperature"])
        if mode == "json":
            kw["json_mode"] = True
        if cfg.get("seed") is not None:
            kw["cache_salt"] = f"seed={cfg['seed']}"
        # ---- run in chunks under the node deadline (RT-6)
        timeout = self._node_timeout(node, deadline)
        t_end = time.monotonic() + timeout
        usage: dict = {}
        resps: list[Any] = []
        for start in range(0, len(batch), LLM_CHUNK_ITEMS):
            remaining = t_end - time.monotonic()
            if remaining <= 0:
                break
            chunk = batch[start:start + LLM_CHUNK_ITEMS]
            got, exc, timed_out = _call_with_timeout(
                lambda chunk=chunk: self.llm.chat_many("executor", chunk, return_exceptions=True, **kw), (), remaining, "llm")
            if timed_out:
                break
            if exc is not None:      # API / client failure: transient node error (not cached)
                return _Outcome("error", error=f"LLM call failed: {type(exc).__name__}: {exc}", llm_usage=usage,
                                transient=True)
            resps.extend(got)
        n_timed_out = len(batch) - len(resps)
        parsed: list[Any] = []
        failures = errors = 0
        first_err = ""
        examples: list[str] = []
        for r in resps:
            add_usage(usage, getattr(r, "usage", None))
            if r is None or getattr(r, "error", None):
                errors += 1
                first_err = first_err or str(getattr(r, "error", "no response"))
                parsed.append(None)
                continue
            text = getattr(r, "text", "")
            v = parse_llm_text(text, mode, choices)
            if v is None:
                failures += 1
                if (text or "").strip() and len(examples) < 3:
                    examples.append(text.strip()[:100])
            parsed.append(v)
        parsed += [None] * n_timed_out
        if n_timed_out:
            diagnostics["llm_timed_out_items"] = n_timed_out
        if errors + n_timed_out == len(batch):
            if n_timed_out and not errors:
                return _Outcome("error", error=f"timeout: no llm item finished within {timeout:.1f} s", llm_usage=usage,
                                timed_out=True)
            more = f" ({n_timed_out} more timed out)" if n_timed_out else ""
            return _Outcome("error", error=f"all {errors} LLM calls failed{more}: {first_err}", llm_usage=usage,
                            transient=True, timed_out=bool(n_timed_out))
        if examples:
            diagnostics["parse_failure_examples"] = examples
        diagnostics.update({"n_items": len(items), "parse_failures": failures, "llm_errors": errors})
        return _Outcome("ok", {LLM_OUTPUT_PORT: parsed}, llm_usage=usage, diagnostics=diagnostics,
                        transient=errors > 0 or n_timed_out > 0)

    def _run_operator_spec(self, op: OperatorSpec, inputs: dict, body_root: Path, depth: int,
                           deadline: float | None = None, units: dict[str, str | None] | None = None) -> _Outcome:
        if depth >= MAX_OPERATOR_DEPTH:
            return _Outcome("error", error=f"operator nesting deeper than {MAX_OPERATOR_DEPTH} (recursive operator {op.ref}?)")
        cv = [f"pre: {v}" for v in check_contract_entries(op.contract.pre, inputs, op.inputs, units)]
        body = op.body
        verrs = body.validate()
        if verrs:
            return _Outcome("error", error=f"operator {op.ref} body is invalid: " + "; ".join(verrs), contract_violations=cv)
        injected: dict[tuple[str, str], Any] = {}
        for bport, targets in op.input_map.items():
            if bport not in inputs:
                continue
            for tn, tp in targets:
                if tn not in body.nodes or tp not in body.nodes[tn].inputs:
                    return _Outcome("error", error=f"operator {op.ref} input_map {bport} -> {tn}.{tp} does not exist in its body",
                                    contract_violations=cv)
                injected[(tn, tp)] = inputs[bport]
        # the operator as a whole gets a time limit as well (its internal nodes have their own)
        own = time.monotonic() + OPERATOR_DEADLINE_FACTOR * self.max_node_s
        run = self._execute_graph(body, body_root, None, injected, depth + 1, parallel=False,
                                  deadline=own if deadline is None else min(own, deadline))
        usage: dict = {}
        stdout_parts: list[str] = []
        internal_errors: list[str] = []
        for n in body.topo_order():
            r = run.records[n]
            add_usage(usage, r.llm_usage)
            if r.stdout_tail:
                stdout_parts.append(f"[{op.ref}/{n}] {_clip_tail(r.stdout_tail, 1000)}")
            if r.status == "error":
                internal_errors.append(f"internal node {n} ({r.kind}) error: {_clip_tail(r.error, 1200)}")
            cv += [f"{n}: {v}" for v in r.contract_violations]
        outputs: dict[str, Any] = {}
        failures: list[str] = []
        for bport, (sn, sp) in op.output_map.items():
            r = run.records.get(sn)
            if r is None:
                failures.append(f"output {bport}: unknown internal node {sn}")
            elif r.status != "ok":
                failures.append(f"output {bport}: internal node {sn} is {r.status}" + (f" ({r.error})" if r.status != "error" else ""))
            elif sp not in run.refs.get(sn, {}):
                failures.append(f"output {bport}: internal node {sn} has no output port {sp!r}")
            else:
                outputs[bport] = run.get_value(sn, sp)
        stdout = _clip_tail("\n".join(stdout_parts), 4000)
        if failures:
            msg = f"operator {op.ref} failed inside its body: " + "; ".join(internal_errors or failures)
            return _Outcome("error", error=msg, stdout_tail=stdout, contract_violations=cv, llm_usage=usage,
                            transient=bool(run.transient), timed_out=run.n_timeouts > 0)
        post_vals = {**inputs, **outputs}
        post_schemas = {**op.inputs, **op.outputs}
        cv += [f"post: {v}" for v in check_contract_entries(op.contract.post, post_vals, post_schemas)]
        return _Outcome("ok", outputs, stdout_tail=stdout, contract_violations=cv, llm_usage=usage,
                        transient=bool(run.transient))


def _operator_fingerprint(op: OperatorSpec, inputs: dict) -> str:
    h = hashlib.sha256(json.dumps(op.to_dict(), sort_keys=True, default=str).encode())
    try:
        h.update(pickle.dumps(inputs, protocol=5))
    except (pickle.PicklingError, TypeError, AttributeError):
        h.update(repr(sorted(inputs)).encode())   # unpicklable inputs: still a stable id for the spec
    return h.hexdigest()[:20]


# DESIGN 8.3 lists these next to the executor; they live in runtime.replay (lazy to avoid an import cycle).
def replay(graph: WorkflowGraph, episode: Any, program: Any, llm: Any, run_dir: str | Path, **kw: Any) -> tuple[Any, Trace]:
    """See :func:`scienceclaw.runtime.replay.replay`."""
    from .replay import replay as _replay

    return _replay(graph, episode, program, llm, run_dir, **kw)


def run_operator_isolated(op: OperatorSpec, inputs: dict, program: Any, llm: Any, run_dir: str | Path,
                          episode: Any = None) -> tuple[dict, NodeRecord]:
    """See :func:`scienceclaw.runtime.replay.run_operator_isolated`."""
    from .replay import run_operator_isolated as _run

    return _run(op, inputs, program, llm, run_dir, episode=episode)
