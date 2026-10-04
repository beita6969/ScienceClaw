"""scilib.audioenc: availability, input checks, remote routing, whitelist and the factual interface text (no torch, no weights)."""
from __future__ import annotations

import numpy as np
import pytest

import scilib
from scilib import _remote, audioenc as ae
from test_scilib_remote import broker  # noqa: F401  (fixture)

W = np.random.default_rng(0).normal(size=(3, 8000)).astype(np.float32) * 0.1


def test_unavailable_without_stack_or_spool(monkeypatch):
    monkeypatch.delenv(_remote.SPOOL_ENV, raising=False)
    monkeypatch.setattr(ae, "_local_ok", lambda *a: False)
    assert not ae.available()
    assert scilib.describe_extra("audioenc") == ""
    with pytest.raises(RuntimeError, match="neither local weights"):
        ae.embed(W)


def test_describe_extra_shown_only_when_available(monkeypatch):
    monkeypatch.delenv("SCIENCECLAW_NO_SCILIB", raising=False)
    monkeypatch.setattr(ae, "available", lambda: True)
    assert scilib.describe_extra("audioenc").startswith("\n\nLibrary `scilib.audioenc`, importable in code nodes:")


def test_bad_inputs():
    with pytest.raises(ValueError, match="model must be"):
        ae.embed(W, model="nope")
    with pytest.raises(ValueError, match="kind must be"):
        ae.embed(W, model="ast_audioset", kind="proj")
    with pytest.raises(ValueError, match="kind must be"):
        ae.embed(W, model="clap_htsat", kind="logits")
    with pytest.raises(ValueError, match="2-D"):
        ae.embed(W[0])
    with pytest.raises(ValueError, match="non-finite"):
        ae.embed(np.where(np.arange(8000) == 3, np.nan, W))
    with pytest.raises(ValueError, match="lengths"):
        ae.embed(W, lengths=[10, 10])
    with pytest.raises(ValueError, match="lengths"):
        ae.embed(W, lengths=[10, 0, 10])
    with pytest.raises(ValueError, match="lengths"):
        ae.embed(W, lengths=[10, 10, 9000])
    with pytest.raises(ValueError, match="whole number"):
        ae.embed(W, sample_rate=16000.5)


def test_remote_routing(broker, monkeypatch):  # noqa: F811
    box, seen, _ = broker
    monkeypatch.setattr(ae, "_local_ok", lambda *a: False)
    assert ae.available()
    got = []

    def handler(fn, a):
        got.append(a)
        return np.ones((a["waveforms"].shape[0], 768), dtype=np.float32)

    box["handler"] = handler
    e = ae.embed(W, lengths=[8000, 7000, 6000], model="ast_audioset", kind="pooled")
    assert e.shape == (3, 768) and e.dtype == np.float32
    assert seen[-1]["module"] == "audioenc" and seen[-1]["fn"] == "embed"
    a = got[-1]
    assert a["model"] == "ast_audioset" and a["kind"] == "pooled" and a["sample_rate"] == 16000.0
    assert a["waveforms"].dtype == np.float32 and a["waveforms"].shape == (3, 8000) and np.array_equal(a["waveforms"], W)
    assert list(a["lengths"]) == [8000, 7000, 6000]
    ae.embed(W, model="clap_htsat", kind="proj")
    assert got[-1]["model"] == "clap_htsat" and list(got[-1]["lengths"]) == [8000] * 3


def test_windows_and_resampling_helpers():
    x = np.arange(100, dtype=np.float32)
    assert len(ae._windows(x, 100)) == 1
    w = ae._windows(x, 60)
    assert len(w) == 2 and np.array_equal(w[0], x[:60]) and np.array_equal(w[1], x[40:])
    assert ae._resample(x, 16000, 16000) is x
    y = ae._resample(np.zeros(1600, dtype=np.float32), 16000, 48000)
    assert y.shape == (4800,) and y.dtype == np.float32
    assert ae._resample(np.zeros(2205, dtype=np.float32), 22050, 16000).shape == (1600,)


def test_worker_whitelist_contains_the_call():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location("sc_worker", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "remote" / "worker.py")
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    assert ("audioenc", "embed") in w.ALLOWED


def test_interface_text_is_factual():
    txt = ae.__doc__.lower()
    for word in ("recommend", "should", "best", "baseline", "margin", "accept", "state of the art", "reference", "anomal"):
        assert word not in txt
    for name in ae.__all__:
        assert name in ae.__doc__
