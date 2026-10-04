"""Typed scientific workflow graph G = (N, E, ψ) with port-level dependencies (paper Eq. 5)."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Iterable

from .schema import PortSchema, compat

NODE_KINDS = ("tool", "operator", "code", "llm", "submit")
SUBMIT_PORT = "y"
TOTAL_CODE_CHARS = 12_000       # render_compact: all code bodies together (about 12k characters)
MIN_CODE_SHARE = 400            # render_compact: a code body is never cut below this many characters


def _one_line(text: str, n: int) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[: max(0, n - 3)] + "..."


CODE_INDENT = 6                 # render_compact indents code lines by this many spaces


def _rendered_len(code: str, indent: int = 0) -> int:
    return sum(len(ln) + indent + 1 for ln in code.splitlines())


def cut_code(code: str, limit: int | None, indent: int = 0) -> str:
    """``code`` cut to at most ~``limit`` rendered characters (every line counts ``indent`` + 1 extra) at a line
    boundary, ending with a marker that says the code is only cut for display (the stored code is complete)
    and how much is not shown."""
    if limit is None or _rendered_len(code, indent) <= limit:
        return code
    lines, kept, used = code.splitlines(), [], 0
    for ln in lines:
        if used + len(ln) + indent + 1 > limit:
            break
        kept.append(ln)
        used += len(ln) + indent + 1
    if not kept:                                       # a single line longer than the limit: cut inside it
        kept = [lines[0][: max(1, limit - indent - 1)]]
        rest = lines[0][len(kept[0]):] + "".join("\n" + ln for ln in lines[1:])
        n_lines = len(lines)
    else:
        rest = "\n".join(lines[len(kept):])
        n_lines = len(lines) - len(kept)
    return ("\n".join(kept) + f"\n# [cut for display: {n_lines} more lines ({len(rest)} chars) of this node's code "
            "are not shown here; the stored code is complete]")


def _code_limits(codes: dict[str, str], per_node: int | None, total: int | None,
                 indent: int = 0) -> dict[str, int | None]:
    """Per-node limits in rendered characters: ``per_node`` first, then water-filling so that the sum stays within
    ``total`` (the longest bodies lose the most; no body is cut below ``MIN_CODE_SHARE`` characters)."""
    full = {k: _rendered_len(v, indent) for k, v in codes.items()}
    lens = {k: (n if per_node is None else min(n, per_node)) for k, n in full.items()}
    limits: dict[str, int | None] = {k: per_node for k in codes}
    if total is None or sum(lens.values()) <= total:
        return limits
    lam, remaining, todo = 0, total, sorted(lens.values())
    for i, ln in enumerate(todo):                     # largest level lam with sum(min(len, lam)) <= total
        share = remaining // (len(todo) - i)
        if ln <= share:
            remaining -= ln
            continue
        lam = share
        break
    lam = max(lam, MIN_CODE_SHARE)
    return {k: (lam if lens[k] > lam else limits[k]) for k in codes}


def _schemas_to_dict(d: dict[str, PortSchema]) -> dict:
    return {k: v.to_dict() for k, v in d.items()}


def _schemas_from_dict(d: dict | None) -> dict[str, PortSchema]:
    return {k: PortSchema.from_dict(v) for k, v in (d or {}).items()}


@dataclass
class Node:
    id: str
    kind: str
    ref: str | None = None
    code: str | None = None
    prompt: str | None = None
    config: dict = field(default_factory=dict)
    inputs: dict[str, PortSchema] = field(default_factory=dict)
    outputs: dict[str, PortSchema] = field(default_factory=dict)
    origin: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in NODE_KINDS:
            raise ValueError(f"unknown node kind {self.kind!r}; expected one of {NODE_KINDS}")
        self.inputs = _schemas_from_dict(self.inputs) if not all(isinstance(v, PortSchema) for v in self.inputs.values()) else self.inputs
        self.outputs = _schemas_from_dict(self.outputs) if not all(isinstance(v, PortSchema) for v in self.outputs.values()) else self.outputs

    def spec_dict(self) -> dict:
        """Everything that determines the node's computation (origin excluded)."""
        return {
            "id": self.id,
            "kind": self.kind,
            "ref": self.ref,
            "code": self.code,
            "prompt": self.prompt,
            "config": self.config,
            "inputs": _schemas_to_dict(self.inputs),
            "outputs": _schemas_to_dict(self.outputs),
        }

    def to_dict(self) -> dict:
        d = self.spec_dict()
        d["origin"] = self.origin
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        return cls(
            id=str(d["id"]),
            kind=d["kind"],
            ref=d.get("ref"),
            code=d.get("code"),
            prompt=d.get("prompt"),
            config=dict(d.get("config") or {}),
            inputs=_schemas_from_dict(d.get("inputs")),
            outputs=_schemas_from_dict(d.get("outputs")),
            origin=dict(d.get("origin") or {}),
        )

    @property
    def generated(self) -> bool:
        return self.kind in ("code", "llm")


@dataclass(frozen=True)
class Edge:
    src: str
    src_port: str
    dst: str
    dst_port: str
    conversion: tuple | None = None  # frozen form of a dict: tuple(sorted(items))

    @staticmethod
    def make(src: str, src_port: str, dst: str, dst_port: str, conversion: dict | None = None) -> "Edge":
        conv = tuple(sorted(conversion.items())) if conversion else None
        return Edge(src, src_port, dst, dst_port, conv)

    @property
    def conversion_dict(self) -> dict | None:
        return dict(self.conversion) if self.conversion else None

    def to_dict(self) -> dict:
        d = {"src": self.src, "src_port": self.src_port, "dst": self.dst, "dst_port": self.dst_port}
        if self.conversion:
            d["conversion"] = self.conversion_dict
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Edge":
        return cls.make(d["src"], d.get("src_port", "out"), d["dst"], d.get("dst_port", "in"), d.get("conversion"))

    def render(self) -> str:
        c = f" [convert {self.conversion_dict}]" if self.conversion else ""
        return f"{self.src}.{self.src_port} -> {self.dst}.{self.dst_port}{c}"


class WorkflowGraph:
    def __init__(self, nodes: dict[str, Node] | None = None, edges: Iterable[Edge] | None = None) -> None:
        self.nodes: dict[str, Node] = dict(nodes or {})
        self.edges: list[Edge] = list(edges or [])

    # ------------------------------------------------------------------ basic
    def copy(self) -> "WorkflowGraph":
        return WorkflowGraph(copy.deepcopy(self.nodes), list(self.edges))

    def in_edges(self, nid: str) -> list[Edge]:
        return [e for e in self.edges if e.dst == nid]

    def out_edges(self, nid: str) -> list[Edge]:
        return [e for e in self.edges if e.src == nid]

    def predecessors(self, nid: str) -> list[str]:
        return sorted({e.src for e in self.in_edges(nid)})

    def successors(self, nid: str) -> list[str]:
        return sorted({e.dst for e in self.out_edges(nid)})

    def descendants(self, nids: Iterable[str]) -> set[str]:
        seen: set[str] = set()
        stack = list(nids)
        while stack:
            n = stack.pop()
            for s in self.successors(n):
                if s not in seen:
                    seen.add(s)
                    stack.append(s)
        return seen

    def ancestors(self, nids: Iterable[str]) -> set[str]:
        seen: set[str] = set()
        stack = list(nids)
        while stack:
            n = stack.pop()
            for p in self.predecessors(n):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return seen

    def submit_node(self) -> str | None:
        subs = [n.id for n in self.nodes.values() if n.kind == "submit"]
        return subs[0] if subs else None

    # -------------------------------------------------------------- structure
    def topo_order(self) -> list[str]:
        indeg = {n: 0 for n in self.nodes}
        for e in self.edges:
            if e.dst in indeg and e.src in indeg:
                indeg[e.dst] += 1
        ready = sorted(n for n, d in indeg.items() if d == 0)
        order: list[str] = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            for e in self.out_edges(n):
                if e.dst in indeg:
                    indeg[e.dst] -= 1
                    if indeg[e.dst] == 0:
                        ready.append(e.dst)
                        ready.sort()
        if len(order) != len(self.nodes):
            raise ValueError("workflow graph has a cycle")
        return order

    def validate(self) -> list[str]:
        errs: list[str] = []
        subs = [n for n in self.nodes.values() if n.kind == "submit"]
        if len(subs) > 1:
            errs.append(f"more than one submit node: {[n.id for n in subs]}")
        seen_dst: dict[tuple[str, str], Edge] = {}
        for e in self.edges:
            if e.src not in self.nodes:
                errs.append(f"edge {e.render()}: unknown source node {e.src}")
                continue
            if e.dst not in self.nodes:
                errs.append(f"edge {e.render()}: unknown destination node {e.dst}")
                continue
            so = self.nodes[e.src].outputs.get(e.src_port)
            di = self.nodes[e.dst].inputs.get(e.dst_port)
            if so is None:
                errs.append(f"edge {e.render()}: node {e.src} has no output port {e.src_port!r} (has {list(self.nodes[e.src].outputs)})")
                continue
            if di is None:
                errs.append(f"edge {e.render()}: node {e.dst} has no input port {e.dst_port!r} (has {list(self.nodes[e.dst].inputs)})")
                continue
            ok, why = compat(so, di, e.conversion_dict)
            if not ok:
                errs.append(f"edge {e.render()}: incompatible schemas {so.render()} -> {di.render()}: {why}")
            key = (e.dst, e.dst_port)
            if key in seen_dst:
                errs.append(f"input port {e.dst}.{e.dst_port} has more than one incoming edge")
            seen_dst[key] = e
        try:
            self.topo_order()
        except ValueError as ex:
            errs.append(str(ex))
        return errs

    def missing_inputs(self, nid: str) -> list[str]:
        wired = {e.dst_port for e in self.in_edges(nid)}
        node = self.nodes[nid]
        optional = set(node.config.get("optional_inputs", []))
        return [p for p in node.inputs if p not in wired and p not in optional]

    # ------------------------------------------------------------ fingerprint
    def fingerprint(self, nid: str, _memo: dict | None = None) -> str:
        memo = _memo if _memo is not None else {}
        if nid in memo:
            return memo[nid]
        node = self.nodes[nid]
        ups = []
        for e in sorted(self.in_edges(nid), key=lambda e: (e.dst_port, e.src, e.src_port)):
            ups.append([e.dst_port, self.fingerprint(e.src, memo), e.src_port, e.conversion_dict])
        blob = json.dumps({"spec": node.spec_dict(), "ups": ups}, sort_keys=True, default=str)
        h = hashlib.sha256(blob.encode()).hexdigest()[:20]
        memo[nid] = h
        return h

    def fingerprints(self) -> dict[str, str]:
        memo: dict[str, str] = {}
        return {n: self.fingerprint(n, memo) for n in self.nodes}

    def graph_fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(include_origin=False), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:20]

    # ------------------------------------------------------------- subgraphs
    def boundary(self, U: set[str]) -> tuple[list[Edge], list[Edge]]:
        incoming = [e for e in self.edges if e.dst in U and e.src not in U]
        outgoing = [e for e in self.edges if e.src in U and e.dst not in U]
        return incoming, outgoing

    def subgraph(self, U: set[str]) -> "WorkflowGraph":
        return WorkflowGraph(
            {n: copy.deepcopy(self.nodes[n]) for n in U},
            [e for e in self.edges if e.src in U and e.dst in U],
        )

    def weak_components(self, U: set[str]) -> list[set[str]]:
        """Maximal weakly connected components of the node subset U (edges restricted to U)."""
        adj: dict[str, set[str]] = {n: set() for n in U}
        for e in self.edges:
            if e.src in U and e.dst in U:
                adj[e.src].add(e.dst)
                adj[e.dst].add(e.src)
        comps: list[set[str]] = []
        seen: set[str] = set()
        for n in sorted(U):
            if n in seen:
                continue
            comp, stack = set(), [n]
            while stack:
                x = stack.pop()
                if x in comp:
                    continue
                comp.add(x)
                stack.extend(adj[x] - comp)
            seen |= comp
            comps.append(comp)
        return comps

    # -------------------------------------------------------------- serialize
    def to_dict(self, include_origin: bool = True) -> dict:
        return {
            "nodes": [n.to_dict() if include_origin else n.spec_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorkflowGraph":
        nodes = {}
        for nd in d.get("nodes", []):
            n = Node.from_dict(nd)
            nodes[n.id] = n
        return cls(nodes, [Edge.from_dict(e) for e in d.get("edges", [])])

    def render_compact(self, max_code_chars: int | None = None, max_total_code_chars: int | None = TOTAL_CODE_CHARS,
                       max_prompt_chars: int = 2000, max_desc_chars: int = 160) -> str:
        """Text view of the canvas for the policy.

        Code is shown in full: ``max_code_chars`` caps every code node individually (``None``: no per-node
        cap) and ``max_total_code_chars`` caps all code bodies together (``None``: no total cap). When the
        total is exceeded the longest bodies are cut first (water-filling), at a line boundary, and each cut
        ends with a marker line saying that the stored code is complete and how much is not shown. Port
        descriptions of tool / operator / code ports are listed as ``in <port>:`` / ``out <port>:`` lines.
        """
        if not self.nodes:
            return "(empty canvas)"
        lines = []
        order = self.topo_order() if not self.validate_cycles() else sorted(self.nodes)
        limits = _code_limits({nid: self.nodes[nid].code for nid in order if self.nodes[nid].code},
                              max_code_chars, max_total_code_chars, CODE_INDENT)
        for nid in order:
            n = self.nodes[nid]
            ins = ", ".join(f"{k}:{v.render()}" for k, v in n.inputs.items()) or "-"
            outs = ", ".join(f"{k}:{v.render()}" for k, v in n.outputs.items()) or "-"
            head = f"[{n.id}] kind={n.kind}" + (f" ref={n.ref}" if n.ref else "")
            lines.append(f"{head}\n    inputs: {ins}\n    outputs: {outs}")
            for side, ports in (("in", n.inputs), ("out", n.outputs)):
                for k, v in ports.items():
                    if v.description:
                        lines.append(f"    {side} {k}: {_one_line(v.description, max_desc_chars)}")
            if n.config:
                cfg = json.dumps(n.config, default=str)
                lines.append(f"    config: {cfg[:400]}" + (f" ...[config cut for display: {len(cfg) - 400} more chars]"
                                                           if len(cfg) > 400 else ""))
            if n.code:
                shown = cut_code(n.code, limits[nid], CODE_INDENT)
                lines.append("    code:\n" + "\n".join(" " * CODE_INDENT + ln for ln in shown.splitlines()))
            if n.prompt:
                p = n.prompt
                lines.append("    prompt: " + p[:max_prompt_chars] + (
                    f" ...[prompt cut for display: {len(p) - max_prompt_chars} more chars; the stored prompt is complete]"
                    if len(p) > max_prompt_chars else ""))
        lines.append("edges:")
        lines.extend("  " + e.render() for e in self.edges) if self.edges else lines.append("  (none)")
        return "\n".join(lines)

    def validate_cycles(self) -> bool:
        try:
            self.topo_order()
            return False
        except ValueError:
            return True
