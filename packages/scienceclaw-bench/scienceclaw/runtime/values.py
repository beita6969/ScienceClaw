"""Value storage (pickles) and tolerance-aware output comparison.

``save_value`` / ``load_value`` exchange port values between the executor, the code-node worker
and boundary replay (pickle protocol 5; numpy / pandas objects are fine). ``outputs_match`` is the
reproducibility / boundary-replay comparison: nested dict / list / tuple / numpy / pandas / numbers /
strings, numeric leaves compared with ``|a - b| <= atol + rtol * |b|`` and NaN == NaN.
"""
from __future__ import annotations

import math
import numbers
import os
import pickle
import threading
from pathlib import Path
from typing import Any

import numpy as np

try:  # pandas is a hard dependency of the package, but keep the module importable without it
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

PICKLE_PROTOCOL = 5
DEFAULT_TOL = {"rtol": 1e-6, "atol": 1e-8}
_MAX_DEPTH = 64


def save_value(obj: Any, path: str | Path) -> None:
    """Atomically pickle ``obj`` to ``path`` (parent directories are created)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    try:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=PICKLE_PROTOCOL)
        os.replace(tmp, p)
    except BaseException:
        tmp.unlink(missing_ok=True)   # our own partial temp file; the error is re-raised
        raise


def load_value(path: str | Path) -> Any:
    """Load a value written by :func:`save_value`."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------------------ comparison
def _tol(tol: dict | None) -> tuple[float, float]:
    t = {**DEFAULT_TOL, **(tol or {})}
    return float(t["rtol"]), float(t["atol"])


def _is_number(x: Any) -> bool:
    return isinstance(x, numbers.Number) and not isinstance(x, (bool, np.bool_))


def _scalar_match(a: Any, b: Any, rtol: float, atol: float) -> bool:
    try:
        ca, cb = complex(a), complex(b)
    except (TypeError, ValueError):
        return False
    if ca == cb:
        return True
    for x, y in ((ca.real, cb.real), (ca.imag, cb.imag)):
        if math.isnan(x) or math.isnan(y):
            if not (math.isnan(x) and math.isnan(y)):
                return False
            continue
        if math.isinf(x) or math.isinf(y):
            if x != y:
                return False
            continue
        if abs(x - y) > atol + rtol * abs(y):
            return False
    return True


def _array_match(x: np.ndarray, y: np.ndarray, rtol: float, atol: float, depth: int) -> bool:
    if x.shape != y.shape:
        return False
    if x.size == 0:
        return True
    kx, ky = x.dtype.kind, y.dtype.kind
    if kx == "b" and ky == "b":
        return bool(np.array_equal(x, y))
    if kx in "biufc" and ky in "biufc":
        return bool(np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True))
    if kx in "mM" and ky in "mM":
        xn, yn = np.isnat(x), np.isnat(y)
        return bool(np.array_equal(xn, yn) and np.array_equal(x[~xn], y[~yn]))
    if kx in "US" and ky in "US":
        return bool(np.array_equal(x, y))
    return all(_match(u, v, rtol, atol, depth + 1) for u, v in zip(x.ravel().tolist(), y.ravel().tolist()))


def _match(a: Any, b: Any, rtol: float, atol: float, depth: int) -> bool:
    if depth > _MAX_DEPTH:
        return False
    if a is b:
        return True
    if a is None or b is None:
        return False
    if pd is not None:
        if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
            if not (isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame)):
                return False
            if a.shape != b.shape or [str(c) for c in a.columns] != [str(c) for c in b.columns]:
                return False
            if not a.index.equals(b.index):
                return False
            return all(_array_match(a.iloc[:, i].to_numpy(), b.iloc[:, i].to_numpy(), rtol, atol, depth)
                       for i in range(a.shape[1]))
        if isinstance(a, pd.Series) or isinstance(b, pd.Series):
            if not (isinstance(a, pd.Series) and isinstance(b, pd.Series)):
                return False
            return a.index.equals(b.index) and _array_match(a.to_numpy(), b.to_numpy(), rtol, atol, depth)
        if isinstance(a, pd.Index) and isinstance(b, pd.Index):
            return _array_match(a.to_numpy(), b.to_numpy(), rtol, atol, depth)
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)) or set(a) != set(b):
            return False
        return all(_match(a[k], b[k], rtol, atol, depth + 1) for k in a)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            xa, xb = np.asarray(a), np.asarray(b)
        except (ValueError, TypeError):
            return False
        return _array_match(xa, xb, rtol, atol, depth)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))) or len(a) != len(b):
            return False
        return all(_match(u, v, rtol, atol, depth + 1) for u, v in zip(a, b))
    if isinstance(a, (str, bytes)) or isinstance(b, (str, bytes)):
        return type(a) is type(b) and a == b
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return isinstance(a, (bool, np.bool_)) and isinstance(b, (bool, np.bool_)) and bool(a) == bool(b)
    if _is_number(a) and _is_number(b):
        return _scalar_match(a, b, rtol, atol)
    if isinstance(a, (set, frozenset)) and isinstance(b, (set, frozenset)):
        return a == b
    try:
        r = a == b
    except Exception:   # objects whose __eq__ raises are simply not equal
        return False
    if isinstance(r, (bool, np.bool_)):
        return bool(r)
    try:
        return bool(np.all(r))
    except (TypeError, ValueError):
        return False


def outputs_match(a: Any, b: Any, tol: dict | None = None) -> bool:
    """True iff ``a`` and ``b`` are equal up to ``tol`` (``{"rtol", "atol"}``), recursively.

    numeric leaves: ``|a - b| <= atol + rtol * |b|`` (NaN equals NaN, infinities must match exactly);
    arrays / pandas objects: same shape (and same index / columns for pandas); strings exact;
    dicts: same keys; lists and tuples are interchangeable sequences.
    """
    rtol, atol = _tol(tol)
    return _match(a, b, rtol, atol, 0)
