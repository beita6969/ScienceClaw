"""Typed executable Operators 𝒪_r with contracts κ and provenance ρ (paper Eq. 12)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .graph import WorkflowGraph
from .schema import PortSchema, value_has_type


@dataclass
class Contract:
    """κ: pre-, post- and applicability conditions.

    pre/post entries are dicts: {"port": name, "check": kind, "value": ...}; supported checks:
      finite            all numeric entries finite
      shape             value = list (ints or symbolic strings); rank/int dims must match
      type              value = PortSchema.type; the *value* must be a Python object of that port type
      unit              value = unit string; compared to the unit actually delivered to the port (``units``
                        of :func:`check_contract_entries`: the source port unit, or the edge conversion
                        target) and, when the caller cannot tell, to the port schema unit
      range             value = [lo, hi] (inclusive; None = open)
      nonempty          len(x) > 0
      len_eq_port       value = other port name; len(this) == len(other)
    """
    pre: list[dict] = field(default_factory=list)
    post: list[dict] = field(default_factory=list)
    applicability: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"pre": self.pre, "post": self.post, "applicability": self.applicability}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Contract":
        d = d or {}
        return cls(list(d.get("pre", [])), list(d.get("post", [])), dict(d.get("applicability", {})))


@dataclass
class OperatorSpec:
    id: str
    version: int
    name: str
    description: str
    body: WorkflowGraph
    inputs: dict[str, PortSchema]
    outputs: dict[str, PortSchema]
    input_map: dict[str, list[tuple[str, str]]]
    output_map: dict[str, tuple[str, str]]
    contract: Contract = field(default_factory=Contract)
    provenance: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    @property
    def ref(self) -> str:
        return f"op:{self.id}"

    @property
    def version_id(self) -> str:
        return f"op:{self.id}@v{self.version}"

    def signature(self) -> str:
        ins = ", ".join(f"{k}: {v.render()}" for k, v in self.inputs.items())
        outs = ", ".join(f"{k}: {v.render()}" for k, v in self.outputs.items())
        return f"{self.ref}({ins}) -> ({outs})"

    def render(self) -> str:
        app = self.contract.applicability
        lines = [f"### {self.signature()}", f"name: {self.name}", f"description: {self.description.strip()}"]
        for kind, ports in (("in", self.inputs), ("out", self.outputs)):
            lines += [f"{kind} {k}: {v.description.strip()}" for k, v in ports.items() if v.description.strip()]
        if self.contract.pre:
            lines.append(f"preconditions: {self.contract.pre}")
        if self.contract.post:
            lines.append(f"postconditions: {self.contract.post}")
        if app:
            lines.append(f"applicability: {app}")
        lines.append(f"internal nodes: {len(self.body.nodes)} ({', '.join(sorted({n.kind for n in self.body.nodes.values()}))})")
        return "\n".join(lines)

    def search_text(self) -> str:
        app = self.contract.applicability
        return " ".join([self.name, self.description, " ".join(self.tags),
                         " ".join(str(v) for vals in app.values() for v in (vals if isinstance(vals, list) else [vals])),
                         " ".join(p.type for p in list(self.inputs.values()) + list(self.outputs.values()))])

    def to_dict(self) -> dict:
        return {
            "id": self.id, "version": self.version, "name": self.name, "description": self.description,
            "body": self.body.to_dict(),
            "inputs": {k: v.to_dict() for k, v in self.inputs.items()},
            "outputs": {k: v.to_dict() for k, v in self.outputs.items()},
            "input_map": {k: [list(t) for t in v] for k, v in self.input_map.items()},
            "output_map": {k: list(v) for k, v in self.output_map.items()},
            "contract": self.contract.to_dict(), "provenance": self.provenance, "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OperatorSpec":
        return cls(
            id=d["id"], version=int(d["version"]), name=d.get("name", d["id"]), description=d.get("description", ""),
            body=WorkflowGraph.from_dict(d["body"]),
            inputs={k: PortSchema.from_dict(v) for k, v in d["inputs"].items()},
            outputs={k: PortSchema.from_dict(v) for k, v in d["outputs"].items()},
            input_map={k: [tuple(t) for t in v] for k, v in d["input_map"].items()},
            output_map={k: tuple(v) for k, v in d["output_map"].items()},
            contract=Contract.from_dict(d.get("contract")), provenance=dict(d.get("provenance", {})),
            tags=list(d.get("tags", [])),
        )


def check_contract_entries(entries: list[dict], values: dict, schemas: dict[str, PortSchema],
                           units: dict[str, str | None] | None = None) -> list[str]:
    """Return human-readable violations of κ pre/post entries on concrete port values.

    ``units`` optionally maps a port to the unit that is really delivered to it (the executor derives it
    from the incoming edge). ``None`` for a port means "unspecified upstream", which is compatible with any
    unit exactly as in :func:`scienceclaw.core.schema.compat`. Without an entry the declared schema unit is used.
    """
    import numpy as np

    viol: list[str] = []
    for c in entries:
        port, kind, val = c.get("port"), c.get("check"), c.get("value")
        if port not in values:
            continue
        x = values[port]
        try:
            if kind == "finite":
                arr = np.asarray(x, dtype=float)
                if not np.isfinite(arr).all():
                    viol.append(f"{port}: non-finite values")
            elif kind == "shape":
                shp = list(np.shape(x))
                if len(shp) != len(val) or any(isinstance(v, int) and v != s for v, s in zip(val, shp)):
                    viol.append(f"{port}: shape {shp} violates {val}")
            elif kind == "type":
                if not value_has_type(x, str(val)):
                    viol.append(f"{port}: value of type {type(x).__name__} is not a {val}")
            elif kind == "unit":
                if units is not None and port in units:
                    if units[port] is not None and units[port] != val:
                        viol.append(f"{port}: unit {units[port]} violates {val}")
                else:
                    sch = schemas.get(port)
                    if sch is not None and sch.unit != val:
                        viol.append(f"{port}: unit {sch.unit} violates {val}")
            elif kind == "range":
                arr = np.asarray(x, dtype=float)
                lo, hi = val
                if lo is not None and np.nanmin(arr) < lo:
                    viol.append(f"{port}: min {np.nanmin(arr):.4g} < {lo}")
                if hi is not None and np.nanmax(arr) > hi:
                    viol.append(f"{port}: max {np.nanmax(arr):.4g} > {hi}")
            elif kind == "nonempty":
                if len(x) == 0:
                    viol.append(f"{port}: empty")
            elif kind == "len_eq_port":
                if val in values and len(x) != len(values[val]):
                    viol.append(f"{port}: len {len(x)} != len({val}) {len(values[val])}")
        except Exception as ex:  # a check that cannot be evaluated is itself a violation
            viol.append(f"{port}: check {kind} failed to evaluate ({type(ex).__name__}: {ex})")
    return viol
