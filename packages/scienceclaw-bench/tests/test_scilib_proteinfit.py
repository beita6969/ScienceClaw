"""scilib.proteinfit: parity with the adapter helpers, leakage-free position statistics, interface text, synthetic end-to-end."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scilib
from scilib import proteinfit as pf
from scienceclaw.bench.tasks import for31_proteingym as m
from scienceclaw.runtime.integrity import scan_code


def _synthetic_assay(seed: int, L: int = 70, n_train: int = 420, n_query: int = 110, tag: str = "A"):
    """A random protein whose fitness = burial-dependent position effect (smooth along the sequence) + residue effects."""
    rng = np.random.default_rng(seed)
    wt = "".join(rng.choice(list(pf.AA), L))
    pos_eff = np.convolve(rng.normal(0, 1, L + 6), np.ones(3) / 3, mode="same")[3:3 + L]
    hyd = np.array([pf._KD[pf.AAI[c]] for c in wt])
    rows = []
    for p in range(1, L + 1):
        for a in pf.AA:
            if a == wt[p - 1]:
                continue
            j = pf.AAI[a]
            eff = -abs(pos_eff[p - 1]) * (1.0 + 0.4 * (hyd[p - 1] > 1) * abs(pf._KD[j] - hyd[p - 1]) / 4) \
                - 1.5 * (a == "P") - 0.15 * abs(pf._VOL[j] - pf._VOL[pf.AAI[wt[p - 1]]]) / 30
            rows.append((f"{wt[p - 1]}{p}{a}", p, wt[p - 1], a, eff + rng.normal(0, 0.15)))
    df = pd.DataFrame(rows, columns=["mutant", "position", "wt_aa", "mut_aa", "DMS_score"])
    df.insert(0, "assay_id", tag)
    perm = rng.permutation(len(df))
    q, t = df.iloc[perm[:n_query]].reset_index(drop=True), df.iloc[perm[n_query:n_query + n_train]].reset_index(drop=True)
    return t, q, wt


def _site_mean_reference(train, query):
    means = train.groupby("position")["DMS_score"].mean()
    return query["position"].map(means).fillna(train["DMS_score"].mean()).to_numpy()


def test_metric_and_tables_match_the_adapter():
    rng = np.random.default_rng(0)
    for _ in range(10):
        a, b = rng.normal(size=30), rng.normal(size=30)
        b[:5] = b[5]                      # ties
        assert pf.spearman(a, b) == pytest.approx(m.spearman(a, b))
    assert pf.spearman(np.ones(6), np.arange(6)) == 0.0
    assert pf.AA == m.AA and (pf.BLOSUM62 == m.BLOSUM62).all()
    props = pf.aa_properties()
    assert list(props.index) == list(pf.AA) and np.isfinite(props.to_numpy()).all()
    assert props.loc["I", "hydropathy"] == m.KYTE_DOOLITTLE["I"] and props.loc["W", "volume"] == m.VOLUME_A3["W"]
    out = pf.mean_spearman([1, 2, 3, 4, 1, 2, 3, 4], [1, 2, 3, 4, 4, 3, 2, 1], ["a"] * 4 + ["b"] * 4)
    assert out["per_group"] == {"a": pytest.approx(1.0), "b": pytest.approx(-1.0)} and out["mean"] == pytest.approx(0.0)


def test_describe_lists_every_public_name():
    text = scilib.describe("proteinfit")
    for name in pf.__all__:
        assert name in text, name


def test_code_node_may_import_scilib():
    code = "from scilib.proteinfit import fit_predict_assays\n\ndef run(inputs, config):\n    return {}\n"
    assert scan_code(code) == []


def test_sequence_context_uses_only_the_wild_type():
    _, _, wt = _synthetic_assay(1)
    ctx = pf.sequence_context(wt)
    assert len(ctx) == len(wt) and list(ctx.index[:2]) == [1, 2] and np.isfinite(ctx.to_numpy()).all()
    assert ctx["wt_hydropathy"].iloc[0] == pytest.approx(pf._KD[pf.AAI[wt[0]]])
    assert ctx["rel_pos"].iloc[-1] == pytest.approx(1.0) and ctx["dist_term"].iloc[0] == 0


def test_position_statistics_never_see_the_rows_own_score():
    tr, _, wt = _synthetic_assay(2)
    base = pf.variant_features(tr, wt, tr, exclude_self=True)
    i = 17
    # swap the score of row i with the score of a row far away along the sequence: the assay mean / spread stay identical,
    # so only statistics that read row i's own score (or those of positions near it) can change
    far = int(np.argmax(np.abs(tr["position"].to_numpy() - tr.loc[i, "position"])))
    tr2 = tr.copy()
    tr2.loc[i, "DMS_score"], tr2.loc[far, "DMS_score"] = tr.loc[far, "DMS_score"], tr.loc[i, "DMS_score"]
    pert = pf.variant_features(tr2, wt, tr2, exclude_self=True)
    stat_cols = [c for c in base.columns if c.startswith(("site_", "cls", "own_class", "sim_", "near_hyd", "nb"))]
    same_pos = (tr["position"] == tr.loc[i, "position"]).to_numpy()
    # row i: unchanged (its own score is excluded); other rows of the same position: changed
    assert np.allclose(base.loc[i, stat_cols].to_numpy(dtype=float), pert.loc[i, stat_cols].to_numpy(dtype=float),
                       atol=1e-3, equal_nan=True)
    others = np.flatnonzero(same_pos & (np.arange(len(tr)) != i))
    assert len(others) and not np.allclose(base.loc[others, "site_mean"], pert.loc[others, "site_mean"])
    # without exclusion the row does see itself
    leak = pf.variant_features(tr2, wt, tr2, exclude_self=False)
    assert leak.loc[i, "site_mean"] != pytest.approx(base.loc[i, "site_mean"])


def test_site_matrix_hierarchical_mean():
    tr, _, wt = _synthetic_assay(3, n_train=200)
    sm = pf.SiteMatrix(tr, wt)
    cnt = sm.counts()
    assert cnt.sum() == len(tr) and sm.M.shape == (len(wt), 20)
    hm = sm.site_mean(shrink=1.0)
    assert hm.shape == (len(wt),) and np.isfinite(hm).all()
    empty = np.flatnonzero(cnt == 0)
    if len(empty):                                             # unseen position -> neighbour prior, not the raw mean
        assert hm[empty[0]] == pytest.approx(sm.neighbour_prior()[empty[0]])
    heavy = sm.site_mean(shrink=1e6)
    assert np.abs(heavy - sm.neighbour_prior()).max() < 1e-3


def test_fit_predict_beats_site_mean_and_is_deterministic():
    tr, q, wt = _synthetic_assay(4, L=50, n_train=300, n_query=80)
    p1 = pf.fit_predict(tr, wt, q)
    p2 = pf.fit_predict(tr, wt, q)
    assert np.array_equal(p1, p2) and len(p1) == len(q) and np.isfinite(p1).all()
    ref = pf.spearman(_site_mean_reference(tr, q), q["DMS_score"])
    got = pf.spearman(p1, q["DMS_score"])
    assert got > ref + 0.03, (got, ref)
    _, mem = pf.fit_predict(tr, wt, q, models=pf.MODELS, return_members=True)     # every model, one call
    assert set(mem) == set(pf.MODELS) and all(len(v) == len(q) and np.isfinite(v).all() for v in mem.values())
    two = pf.fit_predict(tr, wt, [q.iloc[:30], q.iloc[30:]], models=("ridge", "factor"))
    assert [len(x) for x in two] == [30, 50]
    with pytest.raises(ValueError, match="unknown model"):
        pf.fit_predict(tr, wt, q, models=("nope",))


def test_informative_errors():
    tr, q, wt = _synthetic_assay(5)
    with pytest.raises(ValueError, match="lacks column"):
        pf.fit_predict(tr.drop(columns=["DMS_score"]), wt, q)
    bad = q.copy()
    bad.loc[0, "position"] = len(wt) + 3
    with pytest.raises(ValueError, match="outside the wild-type"):
        pf.fit_predict(tr, wt, bad)
    with pytest.raises(ValueError, match="disagrees with the wild-type"):
        pf.fit_predict(tr, wt[::-1], q)
    with pytest.raises(ValueError, match="at least 5"):
        pf.fit_predict(tr.iloc[:3], wt, q)
    with pytest.raises(ValueError, match="dict of sequences"):
        pf.fit_predict(tr, {"A": wt}, q)


def test_fit_predict_assays_aligns_rows_and_pooling():
    ta, qa, wa = _synthetic_assay(6, tag="A", L=40, n_train=220, n_query=60)
    tb, qb, wb = _synthetic_assay(7, tag="B", L=35, n_train=220, n_query=50)
    train = pd.concat([ta, tb], ignore_index=True)
    query = pd.concat([qa, qb], ignore_index=True).sample(frac=1.0, random_state=0).reset_index(drop=True)   # interleaved
    wild = {"A": wa, "B": wb}
    pred = pf.fit_predict_assays(train, wild, query, models=("ridge", "factor"))
    assert pred.shape == (len(query),)
    res = pf.mean_spearman(pred, query["DMS_score"], query["assay_id"])
    assert res["mean"] > 0.5
    # per-assay results are what fit_predict gives on the assay's rows
    mask = (query["assay_id"] == "B").to_numpy()
    direct = pf.fit_predict(tb, wb, query[mask], models=("ridge", "factor"))
    assert np.allclose(pred[mask], direct)
    pooled = pf.fit_predict_assays(train, wild, query, models=("ridge",), pool=True)
    assert pooled.shape == pred.shape and pf.mean_spearman(pooled, query["DMS_score"], query["assay_id"])["mean"] > 0.5
    with pytest.raises(ValueError, match="no sequence for assay"):
        pf.fit_predict_assays(train, {"A": wa}, query)
    with pytest.raises(ValueError, match="no rows for query assay"):
        pf.fit_predict_assays(ta, wild, query)


def test_cross_validate_variants_and_positions():
    tr, _, wt = _synthetic_assay(8, n_train=300)
    a = pf.cross_validate(tr, wt, models=("ridge",), n_folds=3)
    b = pf.cross_validate(tr, wt, models=("ridge",), n_folds=3, by="position")
    assert len(a["per_fold"]) == 3 and -1 <= a["spearman"] <= 1 and a["spearman"] > 0.3
    assert len(b["per_fold"]) == 3 and -1 <= b["spearman"] <= 1
    with pytest.raises(ValueError, match="by must be"):
        pf.cross_validate(tr, wt, by="x")
