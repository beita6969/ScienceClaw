"""Operator candidates o_hat_{i,U} = (G+_i[U], Gamma_dU, kappa_{i,U}, rho_{i,U}) and boundary replay (Eq. 12).

For an executable component U of G+ (from ``split.split_edits``):

* body       = G+[U]  (the executable subgraph, internal edges only);
* boundary   = ``G+.boundary(U)``: one operator input per incoming edge, named ``"<dst>__<dst_port>"``
  (schema = destination port schema, ``input_map`` -> [(dst, dst_port)]), and one operator output per
  distinct (src, src_port) leaving U, named ``"<src>__<src_port>"`` (``output_map`` -> (src, src_port)).
  The node feeding ``submit`` is covered because its edge into the submit node leaves U;
* kappa      = pre-conditions from the recorded boundary input values of tau+ (type, shape rank, unit,
  finite, nonempty), post-conditions from the recorded output values (finite, range with 10 % slack,
  len_eq_port where lengths matched) and applicability {disciplines, task_types, input_types}. Every
  derived entry is checked against the recorded values themselves and dropped if it does not hold there;
* rho        = {episode, k_plus, k_minus, program_version, node_ids, ...};
* name / description: one small ``patch`` call (documentation only, never a judge) with a deterministic
  fallback; id = slug(name) + short hash of the body.

Boundary replay (BReplay) restores the recorded inputs at dU- from tau+ (``NodeRecord.input_refs`` of the
destination nodes), executes the operator in isolation in fresh directories ``repeats`` times
(``runtime.replay.run_operator_isolated``) and compares every output at dU+ with the recorded value
(``NodeRecord.output_refs`` of the source nodes) using ``runtime.values.outputs_match`` with the episode
tolerance. Only operators that pass enter the bundle.
"""
from __future__ import annotations

import logging
import pickle
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..core.graph import WorkflowGraph
from ..core.operators import Contract, OperatorSpec, check_contract_entries
from ..core.program import AgentProgram
from ..core.schema import PortSchema
from ..core.trace import NodeRecord, Trace
from .attribution import EvolutionInstance
from .split import reentrant_nodes
from .skill_patch import _instance_tokens, _scrub, call_patch_model, short_hash, slugify

__all__ = ["make_operator_candidate", "operator_candidate_with_log", "boundary_replay", "boundary_ports",
           "derive_contract", "load_recorded_value", "generalize_boundary_shapes", "RANGE_SLACK"]

log = logging.getLogger(__name__)

RANGE_SLACK = 0.10
# Observed output ranges from ONE source instance are not a valid post-condition on other episodes
# (OOD outputs legitimately leave them), so range checks are off by default.
RANGE_IN_CONTRACT = False
_MIN_LEN_FOR_LEN_EQ = 2


# ------------------------------------------------------------------------------------------ helpers
def load_recorded_value(path: str) -> Any:
    """Load a pickled value recorded by the runtime (``runtime.values.load_value`` when available)."""
    try:
        from ..runtime.values import load_value
    except ImportError:
        with open(path, "rb") as fh:
            return pickle.load(fh)  # noqa: S301 - values written by our own runtime in the run dir
    return load_value(path)


def _record(trace: Trace | None, nid: str) -> NodeRecord | None:
    if trace is None:
        return None
    rec = trace.records.get(nid)
    if rec is not None:
        return rec
    return next((r for r in trace.records.values() if getattr(r, "node_id", None) == nid), None)


def _graph_plus(inst: EvolutionInstance) -> WorkflowGraph:
    return WorkflowGraph.from_dict(inst.e_plus.graph_dict or {})


def boundary_ports(g: WorkflowGraph, U: set[str]) -> tuple[dict[str, PortSchema], dict[str, list[tuple[str, str]]],
                                                          dict[str, PortSchema], dict[str, tuple[str, str]]]:
    """(inputs, input_map, outputs, output_map) of component U of graph g (Gamma over dU)."""
    incoming, outgoing = g.boundary(U)
    inputs: dict[str, PortSchema] = {}
    input_map: dict[str, list[tuple[str, str]]] = {}
    for e in sorted(incoming, key=lambda e: (e.dst, e.dst_port, e.src, e.src_port)):
        name = f"{e.dst}__{e.dst_port}"
        inputs[name] = g.nodes[e.dst].inputs.get(e.dst_port, PortSchema())
        input_map[name] = [(e.dst, e.dst_port)]
    outputs: dict[str, PortSchema] = {}
    output_map: dict[str, tuple[str, str]] = {}
    for src, port in sorted({(e.src, e.src_port) for e in outgoing}):
        name = f"{src}__{port}"
        outputs[name] = g.nodes[src].outputs.get(port, PortSchema())
        output_map[name] = (src, port)
    return inputs, input_map, outputs, output_map


def generalize_boundary_shapes(inputs: dict[str, PortSchema], outputs: dict[str, PortSchema]
                               ) -> tuple[dict[str, PortSchema], dict[str, PortSchema]]:
    """Replace the concrete integer dimensions of the boundary schemas by symbols (Gamma over dU).

    The policy often declares instance sizes (e.g. ``shape=[48, 2]``) on the ports of the nodes it writes;
    copied verbatim, they would make the operator incompatible with every episode of another size. Each
    distinct integer becomes one symbol (``s0``, ``s1``, ... in order of first appearance over the sorted
    ports), so the signature still shows which sizes coincided in the source instance, while ``compat``
    binds symbols per edge. Ranks, types, units and dtypes are unchanged; existing symbols are kept.
    """
    sym: dict[int, str] = {}

    def gen(schema: PortSchema) -> PortSchema:
        if schema.shape is None:
            return schema
        dims = []
        for d in schema.shape:
            if isinstance(d, int) and not isinstance(d, bool):
                dims.append(sym.setdefault(d, f"s{len(sym)}"))
            else:
                dims.append(d)
        return replace(schema, shape=tuple(dims))

    return ({k: gen(inputs[k]) for k in sorted(inputs)}, {k: gen(outputs[k]) for k in sorted(outputs)})


def _float_array(v: Any):
    import numpy as np

    if v is None or isinstance(v, (str, bytes, dict)):
        return None
    try:
        arr = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return None
    return arr


def _rank(v: Any) -> int | None:
    import numpy as np

    if v is None or isinstance(v, (str, bytes, dict)):
        return None
    try:
        return len(np.shape(v))
    except (TypeError, ValueError):
        return None


def _length(v: Any) -> int | None:
    if v is None or isinstance(v, (str, bytes)):
        return None
    try:
        return len(v)
    except TypeError:
        return None


def _pre_checks(port: str, v: Any, schema: PortSchema) -> list[dict]:
    import numpy as np

    out: list[dict] = []
    if schema.type != "any":
        out.append({"port": port, "check": "type", "value": schema.type})
    r = _rank(v)
    if r:
        out.append({"port": port, "check": "shape", "value": [f"d{i}" for i in range(r)]})
    if schema.unit is not None:
        out.append({"port": port, "check": "unit", "value": schema.unit})
    arr = _float_array(v)
    if arr is not None and arr.size and bool(np.isfinite(arr).all()):
        out.append({"port": port, "check": "finite"})
    n = _length(v)
    if n is not None and n > 0:
        out.append({"port": port, "check": "nonempty"})
    return out


def _post_checks(port: str, v: Any, in_lengths: dict[str, int], out_lengths: dict[str, int]) -> list[dict]:
    import numpy as np

    out: list[dict] = []
    arr = _float_array(v)
    if arr is not None and arr.size and not isinstance(v, bool) and getattr(arr, "dtype", None) is not None:
        fin = np.isfinite(arr)
        if bool(fin.all()):
            out.append({"port": port, "check": "finite"})
        is_bool = getattr(np.asarray(v), "dtype", None) is not None and np.asarray(v).dtype == bool
        if fin.any() and not is_bool:
            lo, hi = float(arr[fin].min()), float(arr[fin].max())
            span = hi - lo
            slack = RANGE_SLACK * span if span > 0 else RANGE_SLACK * max(abs(lo), abs(hi))
            if RANGE_IN_CONTRACT and slack > 0:
                out.append({"port": port, "check": "range", "value": [lo - slack, hi + slack]})
    n = _length(v)
    if n is not None and n >= _MIN_LEN_FOR_LEN_EQ:
        match = next((p for p, m in sorted(in_lengths.items()) if m == n), None)
        if match is None:
            match = next((p for p, m in sorted(out_lengths.items()) if m == n and p != port), None)
        if match is not None:
            out.append({"port": port, "check": "len_eq_port", "value": match})
    return out


def derive_contract(inst: EvolutionInstance, g_plus: WorkflowGraph, inputs: dict[str, PortSchema],
                    input_map: dict[str, list[tuple[str, str]]], outputs: dict[str, PortSchema],
                    output_map: dict[str, tuple[str, str]], episode: Any) -> tuple[Contract, dict]:
    """kappa from the boundary slice of tau+ (recorded input/output values) and the episode D_i."""
    trace = inst.e_plus.trace
    notes: dict[str, Any] = {"inputs_loaded": [], "outputs_loaded": [], "load_errors": {}, "dropped": []}
    in_vals: dict[str, Any] = {}
    for name, targets in input_map.items():
        dst, port = targets[0]
        rec = _record(trace, dst)
        path = (rec.input_refs or {}).get(port) if rec is not None else None
        if not path:
            continue
        try:
            in_vals[name] = load_recorded_value(path)
            notes["inputs_loaded"].append(name)
        except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ImportError) as ex:
            notes["load_errors"][name] = f"{type(ex).__name__}: {ex}"
    out_vals: dict[str, Any] = {}
    for name, (src, port) in output_map.items():
        rec = _record(trace, src)
        path = (rec.output_refs or {}).get(port) if rec is not None else None
        if not path:
            continue
        try:
            out_vals[name] = load_recorded_value(path)
            notes["outputs_loaded"].append(name)
        except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ImportError) as ex:
            notes["load_errors"][name] = f"{type(ex).__name__}: {ex}"
    pre: list[dict] = []
    for name, v in in_vals.items():
        pre += _pre_checks(name, v, inputs[name])
    for name, sch in inputs.items():  # schema-only checks for ports whose value was not recorded
        if name not in in_vals and sch.unit is not None:
            pre.append({"port": name, "check": "unit", "value": sch.unit})
    in_len = {k: n for k, v in in_vals.items() if (n := _length(v)) is not None}
    out_len = {k: n for k, v in out_vals.items() if (n := _length(v)) is not None}
    post: list[dict] = []
    for name, v in out_vals.items():
        post += _post_checks(name, v, in_len, out_len)
    # kappa must hold on the source instance itself: drop any derived entry violated by the recorded values
    both = {**in_vals, **out_vals}
    schemas = {**inputs, **outputs}
    kept_pre = [c for c in pre if not check_contract_entries([c], both, schemas)]
    kept_post = [c for c in post if not check_contract_entries([c], both, schemas)]
    notes["dropped"] = [c for c in pre + post if c not in kept_pre and c not in kept_post]
    applicability = {
        "disciplines": [episode.discipline] if getattr(episode, "discipline", None) else [],
        "task_types": [episode.task_type] if getattr(episode, "task_type", None) else [],
        "input_types": sorted({s.type for s in inputs.values()} - {"any"}),
    }
    return Contract(pre=kept_pre, post=kept_post, applicability=applicability), notes


_DOC_SYSTEM = """You write documentation for a reusable, typed workflow operator extracted from a verified scientific workflow.
Given its internal nodes and typed boundary ports, return a short snake_case name describing what it computes and a one-to-three sentence description of what it does, what it expects and what it returns. Describe the computation generically: no instance-specific numbers, ids, file names or answers.
Output ONLY one JSON object: {"name": "<snake_case_name>", "description": "<text>", "tags": ["<tag>", ...]}"""


def _doc_messages(body: WorkflowGraph, inputs: dict[str, PortSchema], outputs: dict[str, PortSchema],
                  contract: Contract, episode: Any) -> list[dict]:
    lines = [f"discipline: {getattr(episode, 'discipline', '')}; task type: {getattr(episode, 'task_type', '')}",
             "inputs: " + ", ".join(f"{k}: {v.render()}" for k, v in inputs.items()),
             "outputs: " + ", ".join(f"{k}: {v.render()}" for k, v in outputs.items()),
             f"contract: pre={contract.pre} post={contract.post}", "internal nodes:"]
    for nid in body.topo_order():
        n = body.nodes[nid]
        lines.append(f"- [{nid}] kind={n.kind} config={n.config}")
        if n.code:
            code = n.code if len(n.code) <= 1500 else n.code[:1500] + "\n# ...(truncated)"
            lines.append("  code:\n" + "\n".join("    " + ln for ln in code.splitlines()))
        if n.prompt:
            lines.append(f"  prompt: {n.prompt[:600]}")
    lines.append("internal edges: " + ("; ".join(e.render() for e in body.edges) or "(none)"))
    return [{"role": "system", "content": _DOC_SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def _fallback_name(body: WorkflowGraph, episode: Any) -> str:
    kinds = "_".join(sorted({n.kind for n in body.nodes.values()}))
    task = slugify(str(getattr(episode, "task_type", "") or "workflow"), 30).replace("-", "_")
    return f"{task}_{kinds}_block"


# -------------------------------------------------------------------------------------------- main
def operator_candidate_with_log(inst: EvolutionInstance, component: set[str], program: AgentProgram, llm: Any,
                                episode: Any) -> tuple[OperatorSpec | None, dict]:
    """``make_operator_candidate`` plus a JSON-safe log (reason when no candidate is formed)."""
    U = {str(n) for n in component}
    logd: dict[str, Any] = {"component": sorted(U), "formed": False, "reason": None, "llm": None}
    if not U:
        logd["reason"] = "empty component"
        return None, logd
    g_plus = _graph_plus(inst)
    missing = sorted(U - set(g_plus.nodes))
    if missing:
        logd["reason"] = f"component nodes not in G+: {missing}"
        return None, logd
    reentrant = reentrant_nodes(g_plus, U)
    if reentrant:   # guard (DESIGN 6): a single operator node cannot be wired around such a path without a cycle
        logd["reason"] = (f"component is not convex: a path leaves it and re-enters via {sorted(reentrant)}; "
                          f"dropped instead of poisoning the bundle")
        log.warning("operator candidate for %s dropped: %s", sorted(U), logd["reason"])
        return None, logd
    trace = inst.e_plus.trace
    bad = sorted(n for n in U if (r := _record(trace, n)) is None or r.status != "ok")
    if bad:
        logd["reason"] = f"component nodes without an ok record in tau+: {bad}"
        return None, logd
    inputs, input_map, outputs, output_map = boundary_ports(g_plus, U)
    inputs, outputs = generalize_boundary_shapes(inputs, outputs)
    if not outputs:
        logd["reason"] = "component has no boundary outputs (nothing leaves U)"
        return None, logd
    body = g_plus.subgraph(U)
    contract, cnotes = derive_contract(inst, g_plus, inputs, input_map, outputs, output_map, episode)
    logd["contract_notes"] = cnotes
    obj, call = call_patch_model(llm, _doc_messages(body, inputs, outputs, contract, episode),
                                 tag="evo.operator_doc", max_tokens=1200)
    logd["llm"] = call
    tokens = _instance_tokens(episode)
    name = description = ""
    doc_tags: list[str] = []
    if obj is not None:
        name = slugify(str(obj.get("name") or ""), 60).replace("-", "_") if obj.get("name") else ""
        description, _ = _scrub(" ".join(str(obj.get("description") or "").split())[:800], tokens)
        doc_tags = [str(t).strip()[:40] for t in (obj.get("tags") or []) if str(t).strip()][:5] \
            if isinstance(obj.get("tags"), list) else []
    if not name or name == "x":
        name = _fallback_name(body, episode)
        logd["name_fallback"] = True
    if not description:
        description = (f"Executable {'/'.join(sorted({n.kind for n in body.nodes.values()}))} block with "
                       f"{len(inputs)} input(s) and {len(outputs)} output(s), extracted from a replay-verified "
                       f"{getattr(episode, 'task_type', '') or 'scientific'} workflow.")
    body_hash = short_hash(body.to_dict(include_origin=False), 8)
    op_id = f"{slugify(name, 40)}-{body_hash}"
    tags: list[str] = []
    for t in [getattr(episode, "discipline", ""), getattr(episode, "task_type", ""),
              *(getattr(episode, "tags", None) or [])[:5], *doc_tags]:
        if t and str(t) not in tags:
            tags.append(str(t))
    incoming, outgoing = g_plus.boundary(U)
    provenance = {
        "source": "CC_Gamma(G+)", "episode": inst.episode_id, "k_plus": inst.k_plus, "k_minus": inst.k_minus,
        "program_version": program.version, "node_ids": sorted(U),
        "graph_fp": getattr(inst.e_plus, "graph_fp", ""),
        "boundary_in": [e.render() for e in incoming], "boundary_out": [e.render() for e in outgoing],
    }
    op = OperatorSpec(id=op_id, version=1, name=name, description=description, body=body, inputs=inputs,
                      outputs=outputs, input_map=input_map, output_map=output_map, contract=contract,
                      provenance=provenance, tags=tags)
    logd.update(formed=True, op_id=op_id, name=name, n_inputs=len(inputs), n_outputs=len(outputs),
                n_pre=len(contract.pre), n_post=len(contract.post))
    return op, logd


def make_operator_candidate(inst: EvolutionInstance, component: set[str], program: AgentProgram, llm: Any,
                            episode: Any) -> OperatorSpec | None:
    """Eq. 12: operator candidate for executable component U of G+ (None if U cannot be abstracted)."""
    op, _ = operator_candidate_with_log(inst, component, program, llm, episode)
    return op


def _run_isolated():
    try:
        from ..runtime.replay import run_operator_isolated
    except ImportError:
        from ..runtime.executor import run_operator_isolated  # DESIGN 8.3 lists it next to the executor
    return run_operator_isolated


def boundary_replay(op: OperatorSpec, inst: EvolutionInstance, component: set[str], program: AgentProgram,
                    llm: Any, episode: Any, run_dir: str | Path, repeats: int = 1) -> tuple[bool, dict]:
    """BReplay(o_hat; tau+): isolated re-execution from recorded dU- inputs, compared at dU+ with tau+."""
    reps = max(1, int(repeats))
    tol = dict(getattr(episode, "tolerance", None) or {"rtol": 1e-6, "atol": 1e-8})
    details: dict[str, Any] = {"op_id": op.id, "component": sorted(component), "repeats": reps, "tolerance": tol,
                               "runs": [], "ok": False, "error": None}
    trace = inst.e_plus.trace
    in_paths: dict[str, str] = {}
    missing_in: list[str] = []
    for name, targets in op.input_map.items():
        dst, port = targets[0]
        rec = _record(trace, dst)
        path = (rec.input_refs or {}).get(port) if rec is not None else None
        if path:
            in_paths[name] = path
        else:
            missing_in.append(name)
    out_paths: dict[str, str] = {}
    missing_out: list[str] = []
    for name, (src, port) in op.output_map.items():
        rec = _record(trace, src)
        path = (rec.output_refs or {}).get(port) if rec is not None and rec.status == "ok" else None
        if path:
            out_paths[name] = path
        else:
            missing_out.append(name)
    if missing_in or missing_out:
        details["error"] = f"recorded boundary values missing in tau+: inputs={missing_in} outputs={missing_out}"
        return False, details
    try:
        expected = {k: load_recorded_value(p) for k, p in out_paths.items()}
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ImportError) as ex:
        details["error"] = f"cannot load recorded outputs: {type(ex).__name__}: {ex}"
        return False, details
    run_isolated = _run_isolated()
    from ..runtime.values import outputs_match

    base = Path(run_dir) / "breplay" / op.id
    base.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for i in range(reps):
        rundir = tempfile.mkdtemp(prefix=f"rep{i}_", dir=base)   # fresh directory per repetition
        run: dict[str, Any] = {"rep": i, "dir": rundir, "ok": False}
        t0 = time.monotonic()
        try:
            inputs = {k: load_recorded_value(p) for k, p in in_paths.items()}   # fresh copies every repeat
            outs, rec = run_isolated(op, inputs, program, llm, rundir, episode=episode)
        except Exception as ex:  # an isolated run that crashes is a failed boundary replay, not a crash
            run["error"] = f"{type(ex).__name__}: {ex}"[:500]
            run["wall_s"] = round(time.monotonic() - t0, 3)
            details["runs"].append(run)
            all_ok = False
            break
        run["wall_s"] = round(time.monotonic() - t0, 3)
        run["status"] = getattr(rec, "status", None)
        run["error"] = getattr(rec, "error", None)
        run["contract_violations"] = list(getattr(rec, "contract_violations", []) or [])
        outs = outs or {}
        mismatched = []
        for name, exp in expected.items():
            if name not in outs:
                mismatched.append(f"{name}: missing")
            elif not outputs_match(outs[name], exp, tol):
                mismatched.append(f"{name}: differs beyond tolerance")
        run["mismatched"] = mismatched
        run["ok"] = run["status"] in (None, "ok") and not mismatched
        details["runs"].append(run)
        if not run["ok"]:
            all_ok = False
            break
    details["ok"] = all_ok
    return all_ok, details
