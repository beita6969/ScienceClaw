"""Separation of strategy changes from executable capability (paper Eq. 10).

    dS_i = Pi_ctrl(delta_i)
    dO_i = CC_Gamma(G+_i, Pi_exec(G+_i, delta_i, tau+_i; O_r))

* Pi_ctrl: the control edits of delta (topology, routing and configuration edits that preserve executable
  payloads), decided by ``core.actions.is_control_edit(action, graph_before)``. ``finish`` is not an edit
  of the graph and is never a control edit here.
* Pi_exec: the executable (``code`` / ``llm``) nodes of G+ that delta created or modified, together with the
  executable nodes weakly connected to them (unchanged helper nodes that are wired into the repaired
  block travel with it). Executable nodes that come from a registered operator of O_r are never candidates
  unless delta touched them. When there is no e- (delta covers the whole graph) every executable node that
  is not a registered-operator expansion is a candidate. Unchanged nodes of unrelated components are NOT
  turned into operators (DESIGN 6).
* CC_Gamma: maximal weakly connected components of those nodes in G+ (``WorkflowGraph.weak_components``),
  made CONVEX (``convex_components``): a component is split along a topological cut when a path leaves it
  and re-enters it through outside nodes (code -> tool -> code), because a single operator node could not be
  wired into any DAG that way. The partition as a whole is also required to contract to an acyclic graph.

``graph_before`` of each action is reconstructed structurally, starting from the replayed failing graph
G(e-) (or the empty canvas when there is no e-) and applying the effective actions of delta in order.
Only node kinds / ids matter for the control/exec classification, so ports are not re-derived here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..core.graph import Edge, Node, WorkflowGraph
from ..core.program import AgentProgram
from ..core.schema import PortSchema
from .attribution import EvolutionInstance, to_action

__all__ = ["split_edits", "split_details", "SplitResult", "expand_actions", "apply_structural",
           "is_registered_operator_node", "EXEC_KINDS", "convex_components", "is_convex", "reentrant_nodes"]

log = logging.getLogger(__name__)

EXEC_KINDS = ("code", "llm")


@dataclass
class SplitResult:
    control_edits: list[Any]
    control_steps: list[int]
    components: list[set[str]]
    exec_nodes: set[str]
    touched_nodes: set[str]                       # nodes created/modified by add_node / modify_node in delta
    unregistered: set[str]                        # exec nodes of G+ not produced by a registered operator
    notes: list[str] = field(default_factory=list)
    excluded: set[str] = field(default_factory=set)   # unchanged exec nodes of G+ left out of Pi_exec
    convex_splits: list[dict] = field(default_factory=list)   # components cut to make CC_Gamma convex

    def summary(self) -> dict:
        return {"control_steps": list(self.control_steps),
                "control_types": [getattr(a, "type", "?") for a in self.control_edits],
                "exec_nodes": sorted(self.exec_nodes), "components": [sorted(c) for c in self.components],
                "touched_nodes": sorted(self.touched_nodes), "excluded_exec": sorted(self.excluded),
                "convex_splits": list(self.convex_splits), "notes": list(self.notes)}


def expand_actions(action: Any) -> list[Any]:
    """Flatten a ``batch`` action (single-turn orchestration) into its atomic sub-actions."""
    if getattr(action, "type", None) != "batch":
        return [action]
    out: list[Any] = []
    for sub in (getattr(action, "payload", None) or {}).get("actions", []) or []:
        a = _coerce_sub_action(sub, uses=list(getattr(action, "uses", []) or []))
        if a is not None:
            out.extend(expand_actions(a))
    return out


def _coerce_sub_action(sub: Any, uses: list[str]) -> Any | None:
    if not isinstance(sub, dict):
        return to_action(sub)
    from ..core.actions import Action

    if "payload" in sub:
        return Action.from_dict(sub)
    inner = sub.get("action", sub)
    if not isinstance(inner, dict) or "type" not in inner:
        log.warning("split: ignoring malformed batch sub-action %r", str(sub)[:200])
        return None
    payload = {k: v for k, v in inner.items() if k != "type"}
    return Action(type=inner["type"], payload=payload, uses=list(sub.get("uses", uses) or []))


def _norm_op_id(ref: Any) -> str:
    s = str(ref)
    if s.startswith("op:"):
        s = s[3:]
    return s.split("@v", 1)[0]


def is_registered_operator_node(node: Node, program: AgentProgram) -> bool:
    """True iff the node was produced by (the expansion of) an Operator registered in ``program``."""
    src = (node.origin or {}).get("from_operator")
    return bool(src) and _norm_op_id(src) in program.operators


def apply_structural(graph: WorkflowGraph, action: Any) -> WorkflowGraph:
    """Structural effect of one atomic action on a copy of ``graph`` (no port inference, no validation).

    Used only to recover ``graph_before`` for the control/exec classification of the next action.
    """
    g = graph.copy()
    t = getattr(action, "type", None)
    p = getattr(action, "payload", None) or {}
    try:
        if t == "add_node":
            nd = p.get("node", p)
            node = Node.from_dict(nd)
            g.nodes[node.id] = node
        elif t == "remove_node":
            nid = str(p.get("id"))
            g.nodes.pop(nid, None)
            g.edges = [e for e in g.edges if e.src != nid and e.dst != nid]
        elif t == "modify_node":
            nid = str(p.get("id"))
            node = g.nodes.get(nid)
            if node is not None:
                patch = p.get("patch", {}) or {}
                for key in ("code", "prompt", "ref"):
                    if key in patch:
                        setattr(node, key, patch[key])
                if "config" in patch and isinstance(patch["config"], dict):
                    node.config = dict(patch["config"])
                for key in ("inputs", "outputs"):
                    if key in patch and isinstance(patch[key], dict):
                        setattr(node, key, {k: PortSchema.from_dict(v) for k, v in patch[key].items()})
        elif t == "add_edge":
            g.edges.append(Edge.from_dict(p.get("edge", p)))
        elif t == "remove_edge":   # ports may be omitted (matched like core.actions: src/dst + given ports)
            ed = p.get("edge", p)
            g.edges = [e for e in g.edges
                       if not (e.src == ed.get("src") and e.dst == ed.get("dst")
                               and (not ed.get("src_port") or e.src_port == ed.get("src_port"))
                               and (not ed.get("dst_port") or e.dst_port == ed.get("dst_port")))]
    except (KeyError, TypeError, ValueError) as ex:
        log.warning("split: could not apply %s structurally (%s: %s); graph_before may be approximate",
                    t, type(ex).__name__, ex)
    return g


def _graph_from(ev: Any) -> WorkflowGraph:
    gd = getattr(ev, "graph_dict", None)
    if gd is None and hasattr(ev, "graph"):
        g = ev.graph
        return g if isinstance(g, WorkflowGraph) else WorkflowGraph.from_dict(g)
    return WorkflowGraph.from_dict(gd or {})


def reentrant_nodes(g: WorkflowGraph, U: set[str]) -> set[str]:
    """Nodes outside ``U`` that lie on a path leaving ``U`` and re-entering it (empty iff ``U`` is convex)."""
    U = set(U)
    return (g.descendants(U) & g.ancestors(U)) - U


def is_convex(g: WorkflowGraph, U: set[str]) -> bool:
    return not reentrant_nodes(g, U)


def _split_once(g: WorkflowGraph, C: set[str], bad: set[str]) -> list[set[str]]:
    """Cut ``C`` at the topological cut induced by the re-entrant nodes ``bad``.

    ``after`` = nodes of C downstream of some bad node; ``before`` = the rest. Every path that leaves C
    through a bad node re-enters in ``after``, and no path leads from ``after`` back to ``before``, so the
    cut removes the re-entrant paths. Both parts are non-empty (``bad`` is both below and above C).
    """
    after = C & g.descendants(bad)
    before = C - after
    return [part for part in (before, after) if part]


def _cycle_members(g: WorkflowGraph, comps: list[set[str]]) -> list[int]:
    """Indices of the components that lie on a cycle of the graph obtained by contracting every component."""
    owner = {n: i for i, c in enumerate(comps) for n in c}

    def key(n: str) -> tuple[str, Any]:
        return ("c", owner[n]) if n in owner else ("n", n)

    adj: dict[tuple[str, Any], set[tuple[str, Any]]] = {}
    for e in g.edges:
        a, b = key(e.src), key(e.dst)
        if a != b:
            adj.setdefault(a, set()).add(b)
    members = []
    for i in range(len(comps)):
        start = ("c", i)
        seen: set[tuple[str, Any]] = set()
        stack = list(adj.get(start, ()))
        hit = False
        while stack and not hit:
            x = stack.pop()
            if x == start:
                hit = True
            elif x not in seen:
                seen.add(x)
                stack.extend(adj.get(x, ()))
        if hit:
            members.append(i)
    return members


def convex_components(g: WorkflowGraph, U: set[str]) -> tuple[list[set[str]], list[dict]]:
    """Weak components of ``U`` in ``g``, refined until every part is convex and the partition is DAG-safe.

    Returns (components sorted by their smallest node id, log of the cuts that were made). A cut never
    merges anything and always shrinks a component, so this terminates; an unsplittable singleton cannot
    create a cycle in a DAG.
    """
    cuts: list[dict] = []
    done: list[set[str]] = []
    work = g.weak_components(set(U))
    while True:
        while work:
            C = work.pop()
            bad = reentrant_nodes(g, C)
            if not bad:
                done.append(C)
                continue
            parts = _split_once(g, C, bad)
            cuts.append({"component": sorted(C), "reentrant_via": sorted(bad),
                         "parts": [sorted(x) for x in parts], "why": "path leaves the component and re-enters"})
            for part in parts:
                work.extend(g.weak_components(part))
        # the components are individually convex; make sure contracting all of them keeps the graph acyclic
        done.sort(key=lambda c: sorted(c))
        try:
            g.topo_order()
        except ValueError:
            log.warning("split: G+ is not acyclic; skipping the joint acyclicity check of the components")
            break
        cyc = [i for i in _cycle_members(g, done) if len(done[i]) > 1]
        if not cyc:
            break
        i = max(cyc, key=lambda k: (len(done[k]), sorted(done[k])))
        C = done.pop(i)
        order = {n: k for k, n in enumerate(g.topo_order())}
        ranked = sorted(C, key=lambda n: (order.get(n, 0), n))
        half = len(ranked) // 2
        parts = [set(ranked[:half]), set(ranked[half:])]
        cuts.append({"component": sorted(C), "reentrant_via": [], "parts": [sorted(x) for x in parts],
                     "why": "contracting the components would create a cycle between operators"})
        for part in parts:
            work.extend(g.weak_components(part))
    return sorted(done, key=lambda c: sorted(c)), cuts


def split_details(inst: EvolutionInstance, program: AgentProgram) -> SplitResult:
    """Full Eq. 10 split with bookkeeping (``split_edits`` returns the two public parts)."""
    from ..core.actions import is_control_edit, touched_nodes  # lazy: owned by another module

    g = _graph_from(inst.e_minus) if inst.e_minus is not None else WorkflowGraph()
    control: list[Any] = []
    control_steps: list[int] = []
    touched: set[str] = set()
    notes: list[str] = []
    steps = list(inst.delta_steps) if len(inst.delta_steps) == len(inst.delta) else [-1] * len(inst.delta)
    for step, action in zip(steps, inst.delta):
        for a in expand_actions(action):
            t = getattr(a, "type", None)
            if t == "finish":
                continue
            if is_control_edit(a, g):
                control.append(a)
                control_steps.append(step)
            if t in ("add_node", "modify_node"):
                touched |= {str(n) for n in touched_nodes(a)}
            g = apply_structural(g, a)
    g_plus = _graph_from(inst.e_plus)
    unregistered = {nid for nid, n in g_plus.nodes.items()
                    if n.kind in EXEC_KINDS and not is_registered_operator_node(n, program)}
    # candidate pool: executable nodes that delta touched, or that are not an expansion of a registered operator
    pool = {nid for nid, n in g_plus.nodes.items() if n.kind in EXEC_KINDS and (nid in touched or nid in unregistered)}
    gone = sorted(n for n in touched if n not in g_plus.nodes)
    if gone:
        notes.append(f"touched nodes absent from G+ (removed later in delta): {gone}")
    if inst.e_minus is None or not _graph_from(inst.e_minus).nodes:
        exec_nodes = set(pool)     # no e-: delta covers the whole graph
    else:
        # Pi_exec = nodes touched in delta + the executable nodes weakly connected to them
        seeds = pool & touched
        exec_nodes = set()
        for comp in g_plus.weak_components(pool):
            if comp & seeds:
                exec_nodes |= comp
    excluded = {nid for nid, n in g_plus.nodes.items() if n.kind in EXEC_KINDS} - exec_nodes
    comps, cuts = convex_components(g_plus, exec_nodes) if exec_nodes else ([], [])
    for c in cuts:
        log.info("split: cut component %s into %s (%s)", c["component"], c["parts"], c["why"])
    return SplitResult(control, control_steps, comps, exec_nodes, touched, unregistered, notes, excluded, cuts)


def split_edits(inst: EvolutionInstance, program: AgentProgram) -> tuple[list[Any], list[set[str]]]:
    """Eq. 10: (control edits Pi_ctrl(delta), executable components CC_Gamma(G+, Pi_exec(...)))."""
    res = split_details(inst, program)
    return res.control_edits, res.components
