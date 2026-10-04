"""runtime.values: pickle store and tolerance-aware comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scienceclaw.runtime.values import load_value, outputs_match, save_value

TOL = {"rtol": 1e-6, "atol": 1e-8}


def test_save_load_roundtrip_numpy_pandas(tmp_path) -> None:
    obj = {"a": np.arange(6.0).reshape(2, 3), "df": pd.DataFrame({"x": [1.0, np.nan], "s": ["u", "v"]}),
           "s": pd.Series([1, 2], index=["p", "q"]), "l": [1, "two", None], "t": (1.5,)}
    p = tmp_path / "deep" / "dir" / "v.pkl"
    save_value(obj, p)
    back = load_value(p)
    assert outputs_match(obj, back, TOL)
    assert not list(p.parent.glob("*.tmp-*"))            # atomic write leaves no temp files


def test_save_unpicklable_raises_and_cleans_up(tmp_path) -> None:
    p = tmp_path / "x.pkl"
    with pytest.raises(Exception):
        save_value({"f": lambda z: z}, p)
    assert not p.exists() and not list(tmp_path.iterdir())


@pytest.mark.parametrize("a,b,expected", [
    (1.0, 1.0 + 1e-9, True),
    (1.0, 1.001, False),
    (float("nan"), float("nan"), True),
    (float("inf"), float("inf"), True),
    (float("inf"), float("-inf"), False),
    (1, 1.0, True),
    (np.float32(0.5), 0.5, True),
    (True, True, True),
    (True, 1, False),
    ("abc", "abc", True),
    ("abc", "abd", False),
    (None, None, True),
    (None, 0, False),
    ([1.0, 2.0], (1.0, 2.0 + 1e-10), True),
    ([1.0, 2.0], [1.0], False),
    ({"a": [1, {"b": np.array([1.0, np.nan])}]}, {"a": [1, {"b": np.array([1.0, np.nan])}]}, True),
    ({"a": 1}, {"a": 1, "b": 2}, False),
    (np.array([1.0, 2.0]), np.array([1.0, 2.0 + 1e-7]), True),
    (np.array([1.0, 2.0]), np.array([1.0, 2.1]), False),
    (np.array([1.0, 2.0]), np.array([[1.0, 2.0]]), False),
    (np.array([1, 2]), [1, 2], True),
    (np.array(["a", "b"]), np.array(["a", "b"]), True),
    (np.array([True, False]), np.array([True, True]), False),
    (np.array([1, "x"], dtype=object), np.array([1, "x"], dtype=object), True),
    (1 + 2j, 1 + 2j, True),
])
def test_outputs_match_cases(a, b, expected) -> None:
    assert outputs_match(a, b, TOL) is expected


def test_outputs_match_pandas() -> None:
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "s": ["a", "b", "c"]})
    df2 = df.copy()
    df2.loc[0, "x"] = 1.0 + 1e-9
    assert outputs_match(df, df2, TOL)
    df3 = df.copy()
    df3.loc[2, "s"] = "z"
    assert not outputs_match(df, df3, TOL)
    assert not outputs_match(df, df.rename(columns={"x": "y"}), TOL)
    assert not outputs_match(df, df.set_index(pd.Index([5, 6, 7])), TOL)
    assert not outputs_match(df, df.to_numpy(), TOL)
    s = pd.Series([1.0, 2.0], index=["a", "b"])
    assert outputs_match(s, s + 1e-12, TOL)
    assert not outputs_match(s, pd.Series([1.0, 2.0], index=["a", "c"]), TOL)


def test_tolerance_controls_comparison() -> None:
    assert not outputs_match(1.0, 1.01, {"rtol": 1e-3, "atol": 0})
    assert outputs_match(1.0, 1.01, {"rtol": 0.02, "atol": 0})
    assert outputs_match(np.array([0.0]), np.array([1e-3]), {"rtol": 0, "atol": 1e-2})
    assert outputs_match(1.0, 1.0 + 1e-7)            # default tolerance
