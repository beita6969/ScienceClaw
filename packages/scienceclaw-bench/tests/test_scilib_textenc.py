"""scilib.textenc: availability, input checks, remote routing and the factual interface text (no torch, no weights)."""
from __future__ import annotations

import numpy as np
import pytest

import scilib
from scilib import _remote, textenc as te
from test_scilib_remote import broker  # noqa: F401  (fixture)

SPANS = ["The Receiving Party shall not disclose Confidential Information.", "This Agreement is governed by New York law."]
HYPS = ["Receiving Party shall not disclose Confidential Information.", "Agreement shall be governed by New York law.",
        "Receiving Party shall return all materials."]


def test_unavailable_without_stack_or_spool(monkeypatch):
    monkeypatch.delenv(_remote.SPOOL_ENV, raising=False)
    monkeypatch.setattr(te, "_local_ok", lambda *a: False)
    assert not te.available()
    assert scilib.describe_extra("textenc") == ""
    for call in (lambda: te.embed(SPANS), lambda: te.relevance(SPANS, HYPS), lambda: te.nli(SPANS, HYPS)):
        with pytest.raises(RuntimeError, match="neither local weights"):
            call()


def test_describe_extra_shown_only_when_available(monkeypatch):
    monkeypatch.delenv("SCIENCECLAW_NO_SCILIB", raising=False)
    monkeypatch.setattr(te, "available", lambda: True)
    assert scilib.describe_extra("textenc").startswith("\n\nLibrary `scilib.textenc`, importable in code nodes:")


def test_bad_inputs():
    with pytest.raises(ValueError, match="list of strings"):
        te.embed("a single string")
    with pytest.raises(ValueError, match="list of strings"):
        te.relevance([1, 2], HYPS)
    with pytest.raises(ValueError, match="model must be"):
        te.nli(SPANS, HYPS, model="bge_large_en")
    with pytest.raises(ValueError, match="model must be"):
        te.embed(SPANS, model="nope")
    assert te.embed([]).shape == (0, 1024)


def test_remote_routing(broker, monkeypatch):  # noqa: F811
    box, seen, _ = broker
    monkeypatch.setattr(te, "_local_ok", lambda *a: False)
    assert te.available()

    def handler(fn, a):
        if fn == "embed":
            return np.ones((len(a["texts"]), 1024), dtype=np.float32)
        if fn == "relevance":
            return np.arange(len(a["passages"]) * len(a["queries"]), dtype=np.float32).reshape(len(a["passages"]), len(a["queries"]))
        return np.zeros((len(a["premises"]), len(a["hypotheses"]), 3), dtype=np.float32)

    box["handler"] = handler
    e = te.embed(SPANS, query=True)
    assert e.shape == (2, 1024) and seen[-1]["module"] == "textenc" and seen[-1]["args"]["query"] is True
    r = te.relevance(SPANS, HYPS)
    assert r.shape == (2, 3) and seen[-1]["fn"] == "relevance" and seen[-1]["args"]["queries"] == HYPS
    n = te.nli(SPANS, HYPS)
    assert n.shape == (2, 3, 3) and seen[-1]["fn"] == "nli" and seen[-1]["args"]["premises"] == SPANS
    te.embed(["", "x"])
    assert seen[-1]["args"]["texts"] == [".", "x"]


def test_worker_whitelist_contains_the_calls():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location("sc_worker", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "remote" / "worker.py")
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    for fn in ("embed", "relevance", "nli"):
        assert ("textenc", fn) in w.ALLOWED


def test_interface_text_is_factual():
    txt = te.__doc__.lower()
    for word in ("recommend", "should", "best", "baseline", "margin", "accept", "state of the art", "reference"):
        assert word not in txt


def test_special_token_template_is_read_from_the_pair_encoding():
    from scilib import textenc

    vocab = {"hello": [11, 12], "world": [21]}

    def tok(a, b=None, add_special_tokens=True):
        ids = vocab[a]
        if b is None:
            return {"input_ids": ids if not add_special_tokens else [0] + ids + [2]}
        return {"input_ids": [0] + ids + [2, 2] + vocab[b] + [2]}      # <s> a </s></s> b </s>

    pre, mid, post = textenc._template(tok)
    assert (pre, mid, post) == ([0], [2, 2], [2])
