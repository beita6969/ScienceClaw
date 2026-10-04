"""Typed Operators for the main entry points of the tool library.

Each operator is a single ``code`` node that calls one ``scilib`` function, with port schemas (type, shape, unit) and pre/post
conditions that the executor checks on the delivered values. They make the pretrained models first-class building blocks of a
workflow canvas: the policy wires an operator instead of writing import and calling code, and a contract violation is reported
as a diagnostic. They start every program as library components (``provenance.source == "library"``).
"""
from __future__ import annotations

from typing import Any

from scienceclaw.core.graph import Node, WorkflowGraph
from scienceclaw.core.operators import Contract, OperatorSpec
from scienceclaw.program.specs import all_specs

NODE = "call"


def _build(spec: dict[str, Any]) -> OperatorSpec:
    body = WorkflowGraph()
    body.nodes[NODE] = Node(id=NODE, kind="code", code="def run(inputs, config):\n    " + spec["code"].replace("\n", "\n    ") + "\n",
                            inputs=dict(spec["inputs"]), outputs=dict(spec["outputs"]))
    return OperatorSpec(
        id=spec["id"], version=1, name=spec["id"], description=spec["description"], body=body,
        inputs=dict(spec["inputs"]), outputs=dict(spec["outputs"]),
        input_map={k: [(NODE, k)] for k in spec["inputs"]}, output_map={k: (NODE, k) for k in spec["outputs"]},
        contract=Contract(pre=list(spec["pre"]), post=list(spec["post"]),
                          applicability={"tools": [spec["tool"]], "input_types": sorted({p.type for p in spec["inputs"].values()})}),
        provenance={"source": "library", "tool": spec["tool"]}, tags=list(spec["tags"]))


def library_operators() -> dict[str, OperatorSpec]:
    """The typed operators wrapping the pretrained tools, keyed by operator id."""
    return {s["id"]: _build(s) for s in all_specs()}
