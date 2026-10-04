"""scilib.tsfm: availability, input checks, remote routing, whitelist and the factual interface text (no torch, no weights)."""
from __future__ import annotations

import numpy as np
import pytest

import scilib
from scilib import _remote, tsfm
from test_scilib_remote import broker  # noqa: F401  (fixture)

H = [np.arange(30, dtype=float), np.arange(20, dtype=float) + 5.0]


def test_unavailable_without_stack_or_spool(monkeypatch):
    monkeypatch.delenv(_remote.SPOOL_ENV, raising=False)
    monkeypatch.setattr(tsfm, "_local_ok", lambda *a: False)
    assert not tsfm.available()
    assert scilib.describe_extra("tsfm") == ""
    with pytest.raises(RuntimeError, match="neither local weights"):
        tsfm.forecast(H, 4)


def test_describe_extra_shown_only_when_available(monkeypatch):
    monkeypatch.delenv("SCIENCECLAW_NO_SCILIB", raising=False)
    monkeypatch.setattr(tsfm, "available", lambda: True)
    assert scilib.describe_extra("tsfm").startswith("\n\nLibrary `scilib.tsfm`, importable in code nodes:")


def test_bad_inputs():
    with pytest.raises(ValueError, match="model must be"):
        tsfm.forecast(H, 4, model="nope")
    with pytest.raises(ValueError, match="horizon must be"):
        tsfm.forecast(H, 0)
    with pytest.raises(ValueError, match="horizon must be"):
        tsfm.forecast(H, 65, model="chronos_bolt")
    with pytest.raises(ValueError, match="horizon must be"):
        tsfm.forecast(H, 1025)
    with pytest.raises(ValueError, match="quantiles"):
        tsfm.forecast(H, 4, quantiles=(0.5, 0.1))
    with pytest.raises(ValueError, match="quantiles"):
        tsfm.forecast(H, 4, quantiles=(0.0, 0.5))
    with pytest.raises(ValueError, match="quantiles"):
        tsfm.forecast(H, 4, quantiles=())
    with pytest.raises(ValueError, match="context_length"):
        tsfm.forecast(H, 4, context_length=2)
    with pytest.raises(ValueError, match="at least one series"):
        tsfm.forecast([], 4)
    with pytest.raises(ValueError, match="at least 3 observed"):
        tsfm.forecast([np.array([1.0, 2.0])], 4)
    with pytest.raises(ValueError, match="at least 3 observed"):
        tsfm.forecast([np.array([1.0, np.nan, np.nan, 2.0])], 4)
    with pytest.raises(ValueError, match="infinite"):
        tsfm.forecast([np.array([1.0, np.inf, 2.0, 3.0])], 4)
    with pytest.raises(ValueError, match="lengths"):
        tsfm.forecast(np.zeros((2, 10)), 4, lengths=[10])
    with pytest.raises(ValueError, match="lengths"):
        tsfm.forecast(np.zeros((2, 10)), 4, lengths=[10, 11])
    with pytest.raises(ValueError, match="lengths"):
        tsfm.forecast(np.zeros(10), 4, lengths=[10])


def test_pad_and_lengths_roundtrip():
    rows, h, q, ctx = tsfm._check(H, 4, (0.1, 0.9), "chronos_2", None, None)
    pad, ln = tsfm._pad(rows)
    assert pad.shape == (2, 30) and list(ln) == [30, 20]
    assert np.isnan(pad[1, :10]).all() and np.array_equal(pad[1, 10:], H[1])
    rows2, *_ = tsfm._check(pad, 4, (0.1, 0.9), "chronos_2", None, ln)
    assert np.array_equal(rows2[0], H[0]) and np.array_equal(rows2[1], H[1])
    assert (h, q, ctx) == (4, (0.1, 0.9), None)


def test_remote_routing(broker, monkeypatch):  # noqa: F811
    box, seen, _ = broker
    monkeypatch.setattr(tsfm, "_local_ok", lambda *a: False)
    assert tsfm.available()
    got = []

    def handler(fn, a):
        got.append(a)
        return np.ones((a["histories"].shape[0], a["horizon"], len(a["quantiles"])), dtype=np.float32)

    box["handler"] = handler
    f = tsfm.forecast(H, 4, quantiles=(0.1, 0.5, 0.9), context_length=12)
    assert f.shape == (2, 4, 3) and f.dtype == np.float32
    assert seen[-1]["module"] == "tsfm" and seen[-1]["fn"] == "forecast"
    a = got[-1]
    assert a["model"] == "chronos_2" and a["horizon"] == 4 and a["context_length"] == 12 and a["quantiles"] == [0.1, 0.5, 0.9]
    assert a["histories"].shape == (2, 30) and list(a["lengths"]) == [30, 20]
    assert np.array_equal(a["histories"][0], H[0]) and np.isnan(a["histories"][1, :10]).all()
    tsfm.forecast(H, 6, model="chronos_bolt")
    assert got[-1]["model"] == "chronos_bolt" and got[-1]["horizon"] == 6 and got[-1]["context_length"] is None


def test_worker_whitelist_contains_the_call():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location("sc_worker", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "remote" / "worker.py")
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    assert ("tsfm", "forecast") in w.ALLOWED


def test_remote_argument_names_match_the_function():
    import inspect
    names = set(inspect.signature(tsfm.forecast).parameters)
    assert {"histories", "lengths", "horizon", "quantiles", "model", "context_length"} <= names


def test_interface_text_is_factual():
    txt = tsfm.__doc__.lower()
    for word in ("recommend", "should", "best", "baseline", "margin", "accept", "state of the art", "reference", "worst", "naive",
                 "persistence", "climatolog", "last value", "last-value"):
        assert word not in txt
    for name in tsfm.__all__:
        assert name in tsfm.__doc__
