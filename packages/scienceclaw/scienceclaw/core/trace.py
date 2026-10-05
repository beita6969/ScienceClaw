"""Execution trace τ and replay-verified evidence e_{t,k} (paper Eq. 8)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class NodeRecord:
    node_id: str
    fingerprint: str
    status: str                      # "ok" | "error" | "skipped" | "pending"
    wall_s: float = 0.0
    error: str | None = None
    stdout_tail: str = ""
    outputs_summary: dict[str, dict] = field(default_factory=dict)
    contract_violations: list[str] = field(default_factory=list)
    input_refs: dict[str, str] = field(default_factory=dict)    # port -> pickle path of the value fed in
    output_refs: dict[str, str] = field(default_factory=dict)   # port -> pickle path of the produced value
    llm_usage: dict = field(default_factory=dict)
    kind: str = ""
    cached: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "NodeRecord":
        return cls(**d)


@dataclass
class Trace:
    records: dict[str, NodeRecord] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    llm_usage: dict = field(default_factory=dict)       # aggregated {"prompt_tokens","completion_tokens","calls"}
    wall_s: float = 0.0
    run_dir: str = ""
    probes: dict = field(default_factory=dict)          # probe name -> {"ok", "msg"} (Episode.run_probes; hidden checks)

    def ok(self) -> bool:
        return all(r.status == "ok" for r in self.records.values())

    def errors(self) -> dict[str, str]:
        return {n: (r.error or r.status) for n, r in self.records.items() if r.status != "ok"}

    def to_dict(self) -> dict:
        d = {"records": {k: v.to_dict() for k, v in self.records.items()}, "order": self.order,
             "llm_usage": self.llm_usage, "wall_s": self.wall_s, "run_dir": self.run_dir}
        if self.probes:
            d["probes"] = dict(self.probes)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Trace":
        return cls({k: NodeRecord.from_dict(v) for k, v in d.get("records", {}).items()}, list(d.get("order", [])),
                   dict(d.get("llm_usage", {})), float(d.get("wall_s", 0.0)), d.get("run_dir", ""),
                   dict(d.get("probes", {})))


@dataclass
class Evidence:
    """e_{t,k}: replay-regenerated evidence of the workflow after step k."""
    step: int
    graph_dict: dict
    y: Any
    trace: Trace
    eval: Any            # scienceclaw.task.EvalResult
    passed: bool
    graph_fp: str = ""

    def summary(self) -> dict:
        ev = self.eval
        return {"step": self.step, "passed": self.passed, "graph_fp": self.graph_fp,
                "primary": getattr(ev, "primary", None), "z": getattr(ev, "z", None),
                "h": getattr(ev, "h", None), "errors": self.trace.errors()}
