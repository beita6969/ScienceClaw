"""Live tasks: an executable task specification D_t = (D_T, D_V, D_E) declared by a plain JSON document.

A *live* task comes from a user request: the gateway agent declares the
objective, the input files, the schema of the deliverable and the hard constraints, and this module turns that declaration
into the same :class:`scienceclaw.task.Episode` the engine already knows how to orchestrate, execute, replay and verify:

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

from scienceclaw.task import Budget, ConstraintSpec, Episode, EvalResult, ToolSpec
from scienceclaw.core.operators import check_contract_entries
from scienceclaw.core.schema import PORT_TYPES, PortSchema

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,39}$")
_FORMAT_TYPE = {"csv": "table", "tsv": "table", "parquet": "table", "json": "any", "npy": "array", "npz": "dict",
                "txt": "text", "text": "text"}
_CHECKS = ("finite", "shape", "type", "range", "nonempty", "len_eq_input", "metric")
_METRICS = ("mae", "mse", "rmse", "smape", "r2", "accuracy", "f1_macro", "auc")
_HIGHER_IS_BETTER = frozenset({"r2", "accuracy", "f1_macro", "auc"})
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


DENIED_PARTS = frozenset({".ssh", ".aws", ".gnupg", ".kube", ".docker", ".netrc"})
_denied_paths: list[Path] = []


def deny_paths(paths: list[Path]) -> None:
    """Locations no task input may point into (the engine's own state, credential stores), whatever the input roots say."""
    _denied_paths[:] = [Path(p).expanduser().resolve() for p in paths]


def check_roots(roots: list[Path]) -> list[str]:
    """Reasons why a set of input roots is unsafe (empty when acceptable): a root must not be the filesystem root or the
    home directory, nor contain the engine's state or a credential store."""
    problems = []
    for r in roots:
        rr = r.expanduser().resolve()
        if rr == Path(rr.anchor) or rr == Path.home().resolve():
            problems.append(f"{rr} is too broad to serve as an input root")
        for d in _denied_paths:
            if d == rr or rr in d.parents:
                problems.append(f"{rr} contains {d}, which must stay out of reach")
    return problems


def resolve_input(path: str, roots: list[Path]) -> Path:
    """The file behind ``path`` if it lies inside an allowed input root (symlinks resolved)."""
    p = Path(path).expanduser()
    p = (p if p.is_absolute() else roots[0] / p).resolve()
    if not any(p == r.resolve() or r.resolve() in p.parents for r in roots):
        raise SpecError(f"input {path!r} is outside the allowed input roots {[str(r) for r in roots]}")
    if DENIED_PARTS & set(p.parts) or any(d == p or d in p.parents for d in _denied_paths):
        raise SpecError(f"input {path!r} points into a protected location")
    if not p.is_file():
        raise SpecError(f"input file not found: {path!r}")
    if p.stat().st_size > MAX_INPUT_BYTES:
        raise SpecError(f"input {path!r} is larger than {MAX_INPUT_BYTES >> 20} MB")
    return p


def _loader(path: Path, fmt: str):
    def load(inputs: dict, config: dict) -> dict:
        if path.resolve() != path:
            raise SpecError(f"input {path.name!r} changed after the task was declared (it is now a link)")
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


def _holdout(path: Path, column: str | None) -> np.ndarray:
    fmt = path.suffix.lstrip(".").lower()
    if fmt in ("csv", "tsv", "parquet"):
        import pandas as pd
        df = pd.read_parquet(path) if fmt == "parquet" else pd.read_csv(path, sep="\t" if fmt == "tsv" else ",")
        return df[column].to_numpy() if column else df.iloc[:, -1].to_numpy()
    if fmt == "npy":
        return np.load(path, allow_pickle=False)
    if fmt == "json":
        return np.asarray(json.loads(path.read_text(encoding="utf-8")))
    raise SpecError(f"holdout {path.name!r}: unsupported format {fmt!r}")


def _score(metric: str, truth: np.ndarray, y: np.ndarray) -> float:
    from sklearn import metrics as skm
    if metric == "mae":
        return float(np.mean(np.abs(truth - y)))
    if metric == "mse":
        return float(np.mean((truth - y) ** 2))
    if metric == "rmse":
        return float(np.sqrt(np.mean((truth - y) ** 2)))
    if metric == "smape":
        return float(100.0 * np.mean(2.0 * np.abs(truth - y) / np.maximum(np.abs(truth) + np.abs(y), 1e-12)))
    if metric == "r2":
        return float(skm.r2_score(truth, y))
    if metric == "accuracy":
        return float(skm.accuracy_score(truth, y))
    if metric == "f1_macro":
        return float(skm.f1_score(truth, y, average="macro"))
    return float(skm.roc_auc_score(truth, y))


def _metric_constraint(entry: dict, input_roots: list[Path]) -> ConstraintSpec:
    """A quality criterion against a held-out file the workflow never sees: it is evaluator-only (not a visible constraint)."""
    name = str(entry.get("name") or "metric")
    metric = str(entry.get("metric", ""))
    direction = str(entry.get("direction") or ("max" if metric in _HIGHER_IS_BETTER else "min"))
    if metric not in _METRICS or direction not in ("min", "max") or not isinstance(entry.get("value"), (int, float)):
        raise SpecError(f"constraint {name!r}: metric needs 'metric' in {_METRICS}, 'direction' min|max and a numeric 'value'")
    truth = np.asarray(_holdout(resolve_input(str(entry.get("target", "")), input_roots), entry.get("column"))).reshape(-1)
    threshold, op = float(entry["value"]), ("<=" if direction == "min" else ">=")

    def measure(y: Any) -> float:
        return _score(metric, truth, np.asarray(y).reshape(-1))

    def check(y: Any, trace: Any) -> tuple[bool, str]:
        try:
            value = measure(y)
        except Exception as ex:
            return False, f"cannot score the deliverable: {type(ex).__name__}: {ex}"
        if not np.isfinite(value):
            return False, f"{metric} is not finite"
        return (value <= threshold if direction == "min" else value >= threshold), f"{metric}={value:.6g} ({op} {threshold:g} required)"

    def grade(y: Any) -> float:
        try:
            value = measure(y)
        except Exception:
            return 0.0
        if direction == "min":
            return 1.0 if value <= threshold else (threshold / value if value > 0 else 0.0)
        return 1.0 if value >= threshold else (max(0.0, value) / threshold if threshold > 0 else 0.0)

    desc = str(entry.get("description") or f"{metric} against held-out data must be {op} {threshold:g}")
    return ConstraintSpec(name, desc, check, False, grade)


def _schema_constraint(schema: PortSchema) -> ConstraintSpec:
    """The declared deliverable schema (type, shape, probability range) as a visible hard constraint."""
    from scienceclaw.runtime.executor import schema_issues

    def check(y: Any, trace: Any) -> tuple[bool, str]:
        issues = schema_issues(y, schema)
        return not issues, "; ".join(issues) if issues else "ok"
    return ConstraintSpec("output_schema", f"the deliverable y matches the declared output: {schema.render()}", check, True)


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
    constraints = [_metric_constraint(c, input_roots) if c.get("check") == "metric" else _constraint(c, loaders)
                   for c in spec.get("constraints", []) or []]
    out = _schema(spec.get("required_output"), "any")
    if out.type == "any" and out.shape is None and not constraints:
        raise SpecError("a task needs something to verify: declare at least one constraint or a typed 'required_output'")
    constraints.insert(0, _schema_constraint(out))

    def evaluate(y: Any, trace: Any) -> EvalResult:
        h = {}
        msgs = {}
        for cs in constraints:
            ok, msg = cs.check(y, trace)
            h[cs.name], msgs[cs.name] = bool(ok), msg
        ok_all = all(h.values())
        grades = [float(cs.grade(y)) if cs.grade else float(h[cs.name]) for cs in constraints]
        share = sum(grades) / len(grades) if grades else 1.0
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
