"""FoR31 ProteinGym substitutions adapter tests (skipped when the data are not available)."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from scienceclaw.bench.tasks import for31_proteingym as m

SPLITS = [("src", 7), ("val", 2), ("id", 4), ("ood", 4)]


def test_helpers():
    assert m.protein_of("BLAT_ECOLX_Jacquier_2013") == "BLAT_ECOLX"
    assert m.protein_of("A0A140D2T1_ZIKV_Sourisseau_2019") == "A0A140D2T1_ZIKV"
    assert m.protein_of("ANCSZ_Hobbs_2022") == "ANCSZ"
    assert (m.BLOSUM62 == m.BLOSUM62.T).all() and m.BLOSUM62[m.AA.index("W"), m.AA.index("W")] == 11
    f = m.substitution_features(pd.DataFrame({"wt_aa": ["A", "D"], "mut_aa": ["P", "K"]}))
    assert f.shape == (2, 9) and f.loc[0, "to_proline"] == 1.0 and f.loc[1, "charge_delta"] == 2.0
    assert m.spearman(np.arange(5), np.arange(5)) == pytest.approx(1.0)
    assert m.spearman(np.ones(5), np.arange(5)) == 0.0


@pytest.fixture(scope="module")
def adapter():
    a = m.Adapter()
    ok, why = a.available()
    if not ok:
        pytest.skip(why)
    return a


@pytest.fixture(scope="module")
def episodes(adapter):
    return {s: adapter.build_episodes(s, n, seed=11) for s, n in SPLITS}


def test_splits_protein_disjoint_and_deterministic(adapter, episodes):
    idx = adapter._index()
    items = {s: {i for e in eps for i in e.lineage["item_ids"]} for s, eps in episodes.items()}
    prots = {s: {idx[a]["protein"] for a in v} for s, v in items.items()}
    for a, b in combinations(items, 2):
        assert not items[a] & items[b], (a, b)
        assert not prots[a] & prots[b], (a, b)
    assert all("Tsuboyama" in a for a in items["ood"])
    assert not any("Tsuboyama" in a for s in ("src", "val", "id") for a in items[s])
    for e in episodes["id"] + episodes["ood"]:
        assert len(set(e.lineage["item_ids"])) == 16 and not e.lineage["reused_items"]
    again = adapter.build_episodes("val", 2, seed=11)
    assert [e.lineage["item_ids"] for e in again] == [e.lineage["item_ids"] for e in episodes["val"]]


def test_tools_expose_only_visible_data(adapter, episodes):
    e = episodes["id"][2]
    ev = e.tool("load_eval_inputs").fn({}, {})["table"]
    assert "DMS_score" not in ev.columns
    assert list(ev["item"].unique()) == list(range(16))
    assert list(ev.groupby("item")["assay_id"].first()) == e.lineage["item_ids"]
    tr = e.tool("load_train").fn({}, {})
    dv = e.tool("load_dev_inputs").fn({}, {})["table"]
    assert "DMS_score" in tr["table"].columns and "DMS_score" not in dv.columns
    for a in e.lineage["item_ids"]:
        q = set(ev.loc[ev.assay_id == a, "mutant"])
        t = set(tr["table"].loc[tr["table"].assay_id == a, "mutant"])
        d = set(dv.loc[dv.assay_id == a, "mutant"])
        assert q and t and not q & t and not q & d and not t & d
        wt = tr["wild_type"][a]
        sub = ev[ev.assay_id == a]
        assert all(wt[p - 1] == w for p, w in zip(sub.position, sub.wt_aa))
    feats = e.tool("substitution_features").fn({"table": ev}, {})["features"]
    assert len(feats) == len(ev) and np.isfinite(feats.to_numpy()).all()
    truth_dev = np.concatenate([adapter._item(a)["dev"]["DMS_score"].to_numpy() for a in e.lineage["item_ids"]])
    s = e.tool("score_dev").fn({"predictions": truth_dev}, {})
    assert s["mean_spearman"] == pytest.approx(1.0)


def test_evaluator_reference_oracle_malformed(adapter, episodes):
    details = []
    for e in (episodes["id"][0], episodes["ood"][0]):
        assays = e.lineage["item_ids"]
        ref = np.concatenate([m.Adapter.site_mean_predictions(adapter._item(a)["train"], adapter._item(a)["query"])
                              for a in assays])
        r = e.evaluate(ref, None)
        assert r.hard_ok() and not r.accepted and r.z == 0
        assert r.primary == pytest.approx(r.details["reference"]) and r.details["norm_score"] == pytest.approx(1.0)
        truth = np.concatenate([adapter._item(a)["query"]["DMS_score"].to_numpy() for a in assays])
        ro = e.evaluate(truth, None)
        assert ro.primary == pytest.approx(1.0) and ro.accepted and ro.z == 1
        details.append(ro.details)
        assert e.evaluate(list(truth), None).primary == pytest.approx(1.0)      # lists are accepted
        bad = e.evaluate(truth[:-1], None)
        assert not bad.h["output_shape"] and bad.z == 0 and bad.primary is None
        nan = truth.copy()
        nan[0] = np.nan
        bad2 = e.evaluate(nan, None)
        assert bad2.h["output_shape"] and not bad2.h["finite_values"] and bad2.z == 0
    assert adapter.pooled_metric(details) == pytest.approx(1.0)
    dup = adapter.pooled_metric([{"kind": "spearman_by_assay", "assays": ["a", "a", "b"], "spearman": [0.2, 0.4, 0.9]}])
    assert dup == pytest.approx((0.3 + 0.9) / 2)


def test_objective_documents_the_domain_library(adapter, episodes):
    ep = episodes["val"][0]
    assert "scilib.proteinfit" in ep.objective and "fit_predict" in ep.objective
