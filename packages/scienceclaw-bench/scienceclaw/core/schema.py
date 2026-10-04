"""Port schemas Γ = (type, shape, unit, provenance) and the Compat relation (paper Eq. 5)."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any

PORT_TYPES = ("any", "number", "text", "array", "table", "list", "dict", "series")


@dataclass(frozen=True)
class PortSchema:
    type: str = "any"
    shape: tuple | None = None
    unit: str | None = None
    dtype: str | None = None
    description: str = ""
    provenance: tuple[str, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        if self.type not in PORT_TYPES:
            raise ValueError(f"unknown port type {self.type!r}; expected one of {PORT_TYPES}")
        if self.shape is not None and not isinstance(self.shape, tuple):
            object.__setattr__(self, "shape", tuple(self.shape))

    def with_provenance(self, *items: str) -> "PortSchema":
        return replace(self, provenance=tuple(self.provenance) + tuple(items))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["shape"] = list(self.shape) if self.shape is not None else None
        d["provenance"] = list(self.provenance)
        return {k: v for k, v in d.items() if v not in (None, "", [])} | {"type": self.type}

    @classmethod
    def from_dict(cls, d: dict | "PortSchema" | None) -> "PortSchema":
        if d is None:
            return cls()
        if isinstance(d, PortSchema):
            return d
        if isinstance(d, str):
            return cls(type=d)
        shape = d.get("shape")
        return cls(
            type=d.get("type", "any"),
            shape=tuple(shape) if shape is not None else None,
            unit=d.get("unit"),
            dtype=d.get("dtype"),
            description=d.get("description", ""),
            provenance=tuple(d.get("provenance", ())),
        )

    def render(self) -> str:
        parts = [self.type]
        if self.shape is not None:
            parts.append("shape=" + "x".join(str(s) for s in self.shape))
        if self.dtype:
            parts.append(f"dtype={self.dtype}")
        if self.unit is not None:
            parts.append(f"unit={self.unit}")
        return "<" + ", ".join(parts) + ">"


def _unify_shapes(a: tuple | None, b: tuple | None) -> tuple[bool, str]:
    if a is None or b is None:
        return True, ""
    if len(a) != len(b):
        return False, f"rank mismatch {a} vs {b}"
    bind: dict[str, Any] = {}
    for x, y in zip(a, b):
        if isinstance(x, int) and isinstance(y, int):
            if x != y:
                return False, f"dim mismatch {a} vs {b}"
            continue
        for sym, other in ((x, y), (y, x)):
            if isinstance(sym, str):
                if sym in bind and bind[sym] != other and not isinstance(other, str):
                    return False, f"symbol {sym} bound to {bind[sym]} and {other}"
                if not isinstance(other, str):
                    bind[sym] = other
    return True, ""


def compat(out: PortSchema, inp: PortSchema, conversion: dict | None = None) -> tuple[bool, str]:
    """Compat_{D_t}(Γ_out, Γ_in): matching type and shape, and explicit unit conversion."""
    if out.type != inp.type and "any" not in (out.type, inp.type):
        # a 'series' can feed an 'array' port and vice versa; lists can feed arrays
        loose = {frozenset(("series", "array")), frozenset(("list", "array"))}
        if frozenset((out.type, inp.type)) not in loose:
            return False, f"type mismatch {out.type} -> {inp.type}"
    ok, why = _unify_shapes(out.shape, inp.shape)
    if not ok:
        return False, why
    if out.unit is None or inp.unit is None or out.unit == inp.unit:
        if conversion:
            return False, "conversion given but units already match or are unspecified"
        return True, ""
    if not conversion:
        return False, f"unit mismatch {out.unit} -> {inp.unit} requires an explicit conversion on the edge"
    if conversion.get("from") != out.unit or conversion.get("to") != inp.unit:
        return False, f"conversion {conversion} does not map {out.unit} -> {inp.unit}"
    if "factor" not in conversion:
        return False, "conversion needs a numeric 'factor' (and optional 'offset')"
    return True, ""


def apply_conversion(value: Any, conversion: dict | None) -> Any:
    """y = factor * x + offset, elementwise for numbers / numpy arrays / lists / pandas objects."""
    if not conversion:
        return value
    a = float(conversion.get("factor", 1.0))
    b = float(conversion.get("offset", 0.0))
    try:
        return value * a + b
    except TypeError:
        if isinstance(value, list):
            return [apply_conversion(v, conversion) for v in value]
        raise


def value_has_type(value: Any, t: str) -> bool:
    """True iff ``value`` is an acceptable Python object for a port of type ``t``.

    number -> int/float/0-d array; text -> str; table -> pandas.DataFrame; dict -> dict; list -> list/tuple;
    array -> numpy.ndarray (list/tuple/pandas.Series accepted); series -> pandas.Series (ndarray/list accepted);
    ``any`` and unknown types accept everything.
    """
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None  # type: ignore
    try:
        import pandas as pd
    except Exception:  # pragma: no cover
        pd = None  # type: ignore
    is_df = pd is not None and isinstance(value, pd.DataFrame)
    is_series = pd is not None and isinstance(value, pd.Series)
    is_arr = np is not None and isinstance(value, np.ndarray)
    is_num = (isinstance(value, (int, float)) and not isinstance(value, bool)) or (
        np is not None and isinstance(value, (np.integer, np.floating)))
    return {
        "number": is_num or (is_arr and value.ndim == 0),
        "text": isinstance(value, str),
        "table": is_df,
        "dict": isinstance(value, dict),
        "list": isinstance(value, (list, tuple)),
        "array": is_arr or isinstance(value, (list, tuple)) or is_series,
        "series": is_series or is_arr or isinstance(value, list),
    }.get(t, True)


def summarize_value(value: Any, schema: PortSchema | None = None, max_head: int = 5) -> dict:
    """Compact, JSON-safe description of a value for diagnostics f and contracts κ."""
    s: dict[str, Any] = {"py_type": type(value).__name__}
    if schema is not None and schema.unit is not None:
        s["unit"] = schema.unit
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None  # type: ignore
    try:
        import pandas as pd
    except Exception:  # pragma: no cover
        pd = None  # type: ignore

    def _num_stats(arr) -> None:
        arr = np.asarray(arr)
        s["shape"] = list(arr.shape)
        s["dtype"] = str(arr.dtype)
        if arr.size and np.issubdtype(arr.dtype, np.number):
            f = np.isfinite(arr)
            s["finite_frac"] = float(f.mean())
            if f.any():
                v = arr[f].astype(float)
                s["min"], s["max"], s["mean"] = float(v.min()), float(v.max()), float(v.mean())

    if value is None:
        s["is_none"] = True
    elif isinstance(value, bool):
        s["value"] = value
    elif isinstance(value, (int, float)):
        s["value"] = value
        s["finite"] = math.isfinite(float(value))
    elif isinstance(value, str):
        s["len"] = len(value)
        s["head"] = value[:200]
    elif np is not None and isinstance(value, np.ndarray):
        _num_stats(value)
        if value.dtype.kind in "OUS":
            s["head"] = [str(x)[:80] for x in value.ravel()[:max_head]]
    elif pd is not None and isinstance(value, pd.DataFrame):
        s["shape"] = list(value.shape)
        s["columns"] = [str(c) for c in value.columns[:30]]
        num = value.select_dtypes("number")
        if num.size:
            arr = num.to_numpy(dtype=float)
            f = np.isfinite(arr)
            s["finite_frac"] = float(f.mean())
    elif pd is not None and isinstance(value, pd.Series):
        _num_stats(value.to_numpy()) if value.dtype.kind in "iufb" else s.update(shape=[len(value)])
    elif isinstance(value, dict):
        s["len"] = len(value)
        s["keys"] = [str(k) for k in list(value)[:20]]
    elif isinstance(value, (list, tuple)):
        s["len"] = len(value)
        head = value[:max_head]
        s["head"] = [str(x)[:80] for x in head]
        if value and isinstance(value[0], dict):
            # keys of the records (union over the first items, in order): str(item)[:80] hides all but the first few
            keys: list[str] = []
            for x in value[:max_head]:
                if isinstance(x, dict):
                    keys += [str(k) for k in x if str(k) not in keys]
            s["item_keys"] = keys[:20]
        if np is not None and value and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value):
            _num_stats(np.asarray(value, dtype=float))
            s["py_type"] = "list"
    return s
