"""Live tasks: an executable task specification D_t = (D_T, D_V, D_E) declared by a plain JSON document.

A benchmark episode comes from a dataset adapter. A *live* task comes from a user request: the gateway agent declares the
objective, the input files, the schema of the deliverable and the hard constraints, and this module turns that declaration
into the same :class:`scienceclaw.bench.task.Episode` the engine already knows how to orchestrate, execute, replay and verify:

* every input file becomes a ``load_<name>`` tool of D_E (read-only, restricted to the allowed input roots);
* the declared constraints become the visible hard constraints of D_V;
* verification is the paper's Pass for tasks without hidden labels: the replayed workflow reproduces its output from a reset
  state, completes within budget and satisfies every hard constraint.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from scienceclaw.bench.task import Budget, ConstraintSpec, Episode, EvalResult, ToolSpec
from scienceclaw.core.operators import check_contract_entries
from scienceclaw.core.schema import PORT_TYPES, PortSchema

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,39}$")
_FORMAT_TYPE = {"csv": "table", "tsv": "table", "parquet": "table", "json": "any", "npy": "array", "npz": "dict",
                "txt": "text", "text": "text"}
_CHECKS = ("finite", "shape", "type", "range", "nonempty", "len_eq_input")
MAX_INPUT_BYTES = 512 * 1024 * 1024


class SpecError(ValueError):
    """The task declaration is invalid; the message says what to change."""


def _schema(d: dict | None, default_type: str = "any") -> PortSchema:
    d = dict(d or {})
    t = str(d.get("type", default_type))
    if t not in PORT_TYPES:
        raise SpecError(f"unknown port type {t!r}; use one of {PORT_TYPES}")
    shape = d.get("shape")
    return PortSchema(type=t, shape=tuple(shape) if shape is not None else None, unit=d.get("unit"),
                      dtype=d.get("dtype"), description=str(d.get("description", "")))


def resolve_input(path: str, roots: list[Path]) -> Path:
    """The file behind ``path`` if it lies inside an allowed input root (symlinks resolved)."""
    p = Path(path).expanduser()
    p = (p if p.is_absolute() else roots[0] / p).resolve()
    if not any(p == r.resolve() or r.resolve() in p.parents for r in roots):
        raise SpecError(f"input {path!r} is outside the allowed input roots {[str(r) for r in roots]}")
    if not p.is_file():
        raise SpecError(f"input file not found: {path!r}")
    if p.stat().st_size > MAX_INPUT_BYTES:
        raise SpecError(f"input {path!r} is larger than {MAX_INPUT_BYTES >> 20} MB")
    return p


def _loader(path: Path, fmt: str):
    def load(inputs: dict, config: dict) -> dict:
        if fmt in ("csv", "tsv"):
            import pandas as pd
            return {"data": pd.read_csv(path, sep="\t" if fmt == "tsv" else ",")}
        if fmt == "parquet":
            import pandas as pd
            return {"data": pd.read_parquet(path)}
        if fmt == "json":
            return {"data": json.loads(path.read_text(encoding="utf-8"))}
        if fmt == "npy":
            return {"data": np.load(path, allow_pickle=False)}
        if fmt == "npz":
            with np.load(path, allow_pickle=False) as z:
                return {"data": {k: z[k] for k in z.files}}
        return {"data": path.read_text(encoding="utf-8")}
    return load


def _constraint(entry: dict, loaders: dict[str, Any]) -> ConstraintSpec:
    check = str(entry.get("check", ""))
    name = str(entry.get("name") or check)
    if check not in _CHECKS:
        raise SpecError(f"constraint {name!r}: unknown check {check!r}; use one of {_CHECKS}")
    value = entry.get("value")
    desc = str(entry.get("description") or f"{check}" + (f" {value}" if value is not None else "") + " on the deliverable y")
    if check == "len_eq_input":
        if value not in loaders:
            raise SpecError(f"constraint {name!r}: value must name a declared input, one of {sorted(loaders)}")
        load = loaders[value]

        def same_length(y: Any, trace: Any) -> tuple[bool, str]:
            try:
                n_in, n_out = len(load({}, {})["data"]), len(y)
            except Exception as ex:
                return False, f"cannot compare lengths: {type(ex).__name__}: {ex}"
            return (n_in == n_out, "ok" if n_in == n_out else f"len(y) = {n_out} but the input has {n_in} rows")
        return ConstraintSpec(name, desc, same_length, True)

    def run(y: Any, trace: Any) -> tuple[bool, str]:
        bad = check_contract_entries([{"port": "y", "check": check, "value": value}], {"y": y}, {"y": PortSchema()})
        return (not bad, "; ".join(bad) if bad else "ok")
    return ConstraintSpec(name, desc, run, True)


def build_live_episode(spec: dict[str, Any], *, task_id: str, input_roots: list[Path]) -> Episode:
    """Turn a task declaration into an executable episode (raises :class:`SpecError` with an actionable message)."""
    if not isinstance(spec, dict):
        raise SpecError("the task must be a JSON object")
    objective = str(spec.get("objective", "")).strip()
    if not objective:
        raise SpecError("'objective' is required: say what the workflow must compute and deliver")
    if not input_roots:
        raise SpecError("no input roots are configured")

    tools: list[ToolSpec] = []
    cache: dict[str, Any] = {}
    names: set[str] = set()
    for item in spec.get("inputs", []) or []:
        name = str(item.get("name", ""))
        if not _NAME.match(name) or name in names:
            raise SpecError(f"input name {name!r} must be a unique identifier (letters, digits, underscore)")
        names.add(name)
        path = resolve_input(str(item.get("path", "")), input_roots)
        fmt = str(item.get("format") or path.suffix.lstrip(".").lower() or "txt")
        if fmt not in _FORMAT_TYPE:
            raise SpecError(f"input {name!r}: unsupported format {fmt!r}; use one of {sorted(_FORMAT_TYPE)}")
        port = _schema({"type": item.get("type") or _FORMAT_TYPE[fmt], "shape": item.get("shape"), "unit": item.get("unit"),
                        "description": item.get("description") or f"contents of {path.name}"})
        tools.append(ToolSpec(name=f"load_{name}", description=str(item.get("description") or f"Read-only: loads the input file {path.name} ({fmt})."),
                              inputs={}, outputs={"data": port}, fn=_loader(path, fmt)))
        cache[name] = path

    loaders = {t.name[len("load_"):]: t.fn for t in tools}
    constraints = [_constraint(c, loaders) for c in spec.get("constraints", []) or []]
    out = _schema(spec.get("required_output"), "any")

    def evaluate(y: Any, trace: Any) -> EvalResult:
        h = {}
        msgs = {}
        for cs in constraints:
            ok, msg = cs.check(y, trace)
            h[cs.name], msgs[cs.name] = bool(ok), msg
        ok_all = all(h.values())
        share = sum(h.values()) / len(h) if h else 1.0
        return EvalResult(metrics={"constraints_passed": float(sum(h.values())), "constraints_total": float(len(h))},
                          primary=1.0 if ok_all else 0.0, direction="max", h=h, h_msgs=msgs, accepted=ok_all, completed=True,
                          details={"norm_score": share})

    b = dict(spec.get("budget") or {})
    budget = Budget(**{k: type(getattr(Budget(), k))(v) for k, v in b.items() if k in Budget.__dataclass_fields__})
    return Episode(
        id=task_id, discipline=str(spec.get("discipline") or "LIVE"), family=str(spec.get("family") or "Live"), split="live",
        task_type=str(spec.get("task_type") or "analysis"), objective=objective, required_output=out, tools=tools,
        constraints=constraints, budget=budget, lineage={"source": "live", "inputs": {k: str(v) for k, v in cache.items()}},
        acceptance=str(spec.get("acceptance") or "Every hard constraint holds and the workflow reproduces its output from a reset state."),
        tags=[str(t) for t in spec.get("tags", []) or []], metric="constraints", direction="max",
        _evaluate=evaluate, _dev_evaluate=None)
