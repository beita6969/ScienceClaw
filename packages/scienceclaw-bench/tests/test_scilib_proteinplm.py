"""scilib.proteinplm: availability, remote routing, feature assembly from (faked) log-probabilities, window assignment; and the
extra feature columns of scilib.proteinfit."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scilib
from scilib import _remote, proteinfit as pf
from scilib import proteinplm as pp
from test_scilib_remote import broker  # noqa: F401  (fixture)

SEQ = "MKTAYIAKQRQISFVKSHFSRQ"


def _fake_lp(L: int, masked: bool) -> np.ndarray:
    rng = np.random.default_rng(1 if masked else 2)
    x = rng.normal(size=(L, 20))
    return x - np.logaddexp.reduce(x, axis=1, keepdims=True)


def test_unavailable_without_stack_or_spool(monkeypatch):
    monkeypatch.delenv(_remote.SPOOL_ENV, raising=False)
    monkeypatch.setattr(pp, "_local_ok", lambda *a: False)
    assert not pp.available()
    assert scilib.describe_extra("proteinplm") == ""
    with pytest.raises(RuntimeError, match="neither local weights"):
        pp.site_logprobs(SEQ)


def test_describe_extra_shown_only_when_available(monkeypatch):
    monkeypatch.delenv("SCIENCECLAW_NO_SCILIB", raising=False)
    monkeypatch.setattr(pp, "available", lambda: True)
    assert scilib.describe_extra("proteinplm").startswith("\n\nLibrary `scilib.proteinplm`, importable in code nodes:")


def test_bad_inputs():
    with pytest.raises(ValueError, match="standard"):
        pp.site_logprobs("MKTZ")
    with pytest.raises(ValueError, match="model must be"):
        pp.site_logprobs(SEQ, model="nope")


def test_remote_routing(broker, monkeypatch):  # noqa: F811
    box, seen, _ = broker
    monkeypatch.setattr(pp, "_local_ok", lambda *a: False)
    assert pp.available()
    box["handler"] = lambda fn, a: {k: _fake_lp(len(a["positions"][k] or range(len(s))), a["masked"]) for k, s in a["wild_type"].items()}
    out = pp.site_logprobs(SEQ, positions=[3, 1, 3], masked=False)
    assert out.shape == (3, 20) and np.allclose(out[0], out[2])
    assert seen[0]["args"]["positions"] == {"_": [1, 3]}
    assert seen[0]["module"] == "proteinplm" and seen[0]["fn"] == "site_logprobs"
    assert seen[0]["args"]["masked"] is False and seen[0]["args"]["wild_type"] == {"_": SEQ}
    d = pp.site_logprobs({"a": SEQ, "b": SEQ[:10]}, {"a": [1, 2], "b": None})
    assert set(d) == {"a", "b"} and d["a"].shape == (2, 20) and d["b"].shape == (10, 20)


def test_plm_features_assembly(monkeypatch):
    lp = {True: _fake_lp(len(SEQ), True), False: _fake_lp(len(SEQ), False)}

    def fake(wild_type, positions=None, masked=True, model="esm2_650m"):
        return {k: lp[masked][np.array(positions[k]) - 1] for k in wild_type}
    monkeypatch.setattr(pp, "site_logprobs", fake)
    tab = pd.DataFrame({"assay_id": ["a"] * 3, "position": [2, 2, 5], "wt_aa": ["K", "K", "Y"], "mut_aa": ["A", "L", "G"]},
                       index=[10, 11, 12])
    F = pp.plm_features(tab, {"a": SEQ})
    assert list(F.index) == [10, 11, 12] and list(F.columns) == pp.FEATURES
    ai = {a: i for i, a in enumerate(pp.AA)}
    M, W = lp[True], lp[False]
    assert F["esm_llr_masked"].iloc[0] == pytest.approx(M[1, ai["A"]] - M[1, ai["K"]])
    assert F["esm_llr_wt"].iloc[2] == pytest.approx(W[4, ai["G"]] - W[4, ai["Y"]])
    assert F["esm_logp_wt_masked"].iloc[1] == pytest.approx(M[1, ai["K"]])
    assert F["esm_entropy_masked"].iloc[0] == F["esm_entropy_masked"].iloc[1] > 0
    assert F["esm_site_mean_llr"].iloc[0] == pytest.approx(np.delete(M[1], ai["K"]).mean() - M[1, ai["K"]])
    with pytest.raises(ValueError, match="disagrees"):
        pp.plm_features(tab.assign(wt_aa="W"), {"a": SEQ})
    with pytest.raises(ValueError, match="outside"):
        pp.plm_features(tab.assign(position=[2, 2, 99]), {"a": SEQ})


def test_windows_cover_positions_with_centered_windows():
    short = pp._windows(500, np.arange(500))
    assert list(short) == [0] and len(short[0]) == 500
    L = 3000
    w = pp._windows(L, np.array([0, 10, 1500, 2999]))
    assert sorted(w) == [0, 989, 1978] or all(0 <= s <= L - pp._WIN for s in w)
    for s, ps in w.items():
        assert all(s <= p < s + pp._WIN for p in ps)
    assert sorted(p for ps in w.values() for p in ps) == [0, 10, 1500, 2999]


def _toy():
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list(pf.AA), 40))
    rows = []
    for p in range(1, 41):
        for m in rng.choice(list(pf.AA), 8, replace=False):
            if m != seq[p - 1]:
                rows.append((p, seq[p - 1], m))
    t = pd.DataFrame(rows, columns=["position", "wt_aa", "mut_aa"])
    t["DMS_score"] = rng.normal(size=len(t)) + 0.5 * (t["position"] % 7 == 0)
    return t, seq


def test_fit_predict_extra_columns():
    t, seq = _toy()
    tr, q = t.iloc[:200].reset_index(drop=True), t.iloc[200:260].reset_index(drop=True)
    rng = np.random.default_rng(3)
    Etr = pd.DataFrame({"zs": rng.normal(size=len(tr)), "zs2": rng.normal(size=len(tr))})
    Eq = pd.DataFrame({"zs": rng.normal(size=len(q)), "zs2": rng.normal(size=len(q))})
    base = pf.fit_predict(tr, seq, q, models=("ridge", "lgbm"))
    plus = pf.fit_predict(tr, seq, q, models=("ridge", "lgbm"), extra_train=Etr, extra_query=Eq)
    assert plus.shape == base.shape and not np.allclose(plus, base)
    lst = pf.fit_predict(tr, seq, [q.iloc[:20], q.iloc[20:]], models=("ridge",), extra_train=Etr, extra_query=[Eq.iloc[:20], Eq.iloc[20:]])
    assert [len(x) for x in lst] == [20, 40]
    with pytest.raises(ValueError, match="extra_query"):
        pf.fit_predict(tr, seq, q, models=("ridge",), extra_train=Etr)
    with pytest.raises(ValueError, match="without extra_train"):
        pf.fit_predict(tr, seq, q, models=("ridge",), extra_query=Eq)
    with pytest.raises(ValueError, match="one row per row"):
        pf.fit_predict(tr, seq, q, models=("ridge",), extra_train=Etr.iloc[:5], extra_query=Eq)
    with pytest.raises(ValueError, match="same columns"):
        pf.fit_predict(tr, seq, q, models=("ridge",), extra_train=Etr, extra_query=Eq.rename(columns={"zs": "zz"}))


def test_fit_predict_assays_extra_columns():
    t, seq = _toy()
    tr = pd.concat([t.iloc[:150].assign(assay_id="a"), t.iloc[100:250].assign(assay_id="b")], ignore_index=True)
    q = pd.concat([t.iloc[250:280].assign(assay_id="a"), t.iloc[250:290].assign(assay_id="b")], ignore_index=True)
    rng = np.random.default_rng(4)
    Etr = pd.DataFrame({"zs": rng.normal(size=len(tr))})
    Eq = pd.DataFrame({"zs": rng.normal(size=len(q))})
    p = pf.fit_predict_assays(tr, {"a": seq, "b": seq}, q, models=("ridge",), extra_train=Etr, extra_query=Eq)
    assert p.shape == (len(q),)
    with pytest.raises(ValueError, match="together"):
        pf.fit_predict_assays(tr, {"a": seq, "b": seq}, q, models=("ridge",), extra_train=Etr)


def test_fit_predict_assays_plm_flag(monkeypatch):
    t, seq = _toy()
    tr = pd.concat([t.iloc[:150].assign(assay_id="a"), t.iloc[100:250].assign(assay_id="b")], ignore_index=True)
    dev = pd.concat([t.iloc[250:270].assign(assay_id="a"), t.iloc[250:275].assign(assay_id="b")], ignore_index=True)
    ev = pd.concat([t.iloc[270:290].assign(assay_id="a"), t.iloc[275:300].assign(assay_id="b")], ignore_index=True)
    calls = []

    def fake(wild_type, positions=None, masked=True, model="esm2_650m"):
        calls.append(masked)
        return {k: np.stack([np.random.default_rng([7 if masked else 8, int(p)]).normal(size=20) for p in positions[k]]) for k in wild_type}
    monkeypatch.setattr(pp, "site_logprobs", fake)
    monkeypatch.setattr(pp, "available", lambda: True)
    w = {"a": seq, "b": seq}
    p = pf.fit_predict_assays(tr, w, [dev, ev], models=("ridge",), plm=True)
    assert [len(x) for x in p] == [len(dev), len(ev)] and calls == [True, False]
    cols = [pp.plm_features(pd.concat([x[["position", "wt_aa", "mut_aa", "assay_id"]]], ignore_index=True), w) for x in (tr, dev, ev)]
    q = pf.fit_predict_assays(tr, w, [dev, ev], models=("ridge",), extra_train=cols[0], extra_query=[cols[1], cols[2]])
    assert all(np.allclose(a, b) for a, b in zip(p, q))
    with pytest.raises(ValueError, match="cannot be combined"):
        pf.fit_predict_assays(tr, w, dev, models=("ridge",), plm=True, extra_train=cols[0], extra_query=cols[1])
    monkeypatch.setattr(pp, "available", lambda: False)
    with pytest.raises(RuntimeError, match="not available"):
        pf.fit_predict_assays(tr, w, dev, models=("ridge",), plm=True)
