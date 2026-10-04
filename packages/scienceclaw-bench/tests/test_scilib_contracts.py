"""scilib.contracts: metric parity with the adapter, interface text, sandbox import, end-to-end on synthetic contracts."""
from __future__ import annotations

import numpy as np
import pytest

import scilib
from scilib import contracts as sc
from scienceclaw.bench.tasks import for48_contractnli as m
from scienceclaw.runtime.integrity import scan_code

FILLER = [
    "This Agreement is governed by the laws of the State of Delaware.",
    "Notices under this Agreement shall be given in writing to the addresses set out below.",
    "Each party represents that it has full authority to enter into this Agreement.",
    "No amendment of this Agreement is effective unless signed by both parties.",
    "The headings in this Agreement are for convenience only.",
    "This Agreement may be executed in counterparts, each of which is an original.",
    "Neither party may assign this Agreement without prior consent.",
    "Any dispute shall be resolved by the courts of the county where the Disclosing Party resides.",
    "Payment terms are set out in the separate services schedule.",
    "The parties are independent contractors and nothing creates a partnership.",
]
RETURN = ["Upon termination the Receiving Party shall return all confidential materials to the Disclosing Party.",
          "On written request the Recipient must destroy or return every document containing Confidential Information.",
          "At the end of the term all Confidential Information shall be returned or destroyed by the Receiving Party."]
RETURN_NOT = ["The Receiving Party is not required to return confidential materials after termination.",
              "The Recipient shall not be obliged to destroy or return any document containing Confidential Information."]
LICENSE = ["Nothing in this Agreement grants the Receiving Party any license or right in the Confidential Information.",
           "No license or other right to the proprietary information is granted to the Recipient hereby."]
HYPS = {"h-return": {"hypothesis": "Receiving Party shall return or destroy confidential materials upon termination.",
                     "short_description": "Return"},
        "h-license": {"hypothesis": "Agreement shall not grant Receiving Party any license to Confidential Information.",
                      "short_description": "License"},
        "h-never": {"hypothesis": "Receiving Party shall notify Disclosing Party of every subpoena.",
                    "short_description": "Notice"}}


def _synthetic(n_docs: int, seed: int, prefix: str = "d"):
    rng = np.random.default_rng(seed)
    docs, ann = [], []
    for i in range(n_docs):
        spans = [FILLER[j] for j in rng.choice(len(FILLER), 8, replace=False)]
        marks = {}
        for key, pool, pool_not in (("h-return", RETURN, RETURN_NOT), ("h-license", LICENSE, LICENSE)):
            if rng.random() < 0.75:
                neg = key == "h-return" and rng.random() < 0.3
                text = (pool_not if neg else pool)[rng.integers(len(pool_not if neg else pool))]
                marks[key] = (text, "Contradiction" if neg else "Entailment")
        for text, _ in marks.values():
            spans.insert(int(rng.integers(len(spans) + 1)), text)
        did = f"{prefix}{i}"
        docs.append({"doc_id": did, "span_texts": spans})
        for key in ("h-return", "h-license", "h-never"):
            if key in marks:
                ann.append({"doc_id": did, "hypothesis_key": key, "label": marks[key][1],
                            "evidence_spans": [spans.index(marks[key][0])]})
            else:
                ann.append({"doc_id": did, "hypothesis_key": key, "label": "NotMentioned", "evidence_spans": []})
    return docs, ann


def _pairs(docs, ann):
    return sc.annotation_pairs(docs, ann)


def test_metric_helpers_match_the_adapter():
    rng = np.random.default_rng(3)
    for _ in range(25):
        g = (rng.random(40) < 0.15).astype(int)
        if g.sum() == 0:
            g[0] = 1
        s = np.round(rng.random(40), 1)                      # ties on purpose
        assert sc.average_precision(g, s) == pytest.approx(m.average_precision(g, s))
        assert sc.precision_at_recall(g, s, 0.8) == pytest.approx(m.precision_at_recall(g, s, 0.8))
    assert np.isnan(sc.precision_at_recall(np.zeros(3, int), np.ones(3), 0.8))
    assert sc.gold_vector(5, [1, 3]).tolist() == [0, 1, 0, 1, 0]


def test_evaluate_predictions_aggregates_per_pair():
    gold = [np.array([0, 1, 0, 1]), np.array([1, 0])]
    preds = [{"label": "Entailment", "span_scores": [0.1, 0.9, 0.2, 0.8]},
             {"label": "Contradiction", "span_scores": [0.1, 0.9]}]
    r = sc.evaluate_predictions(preds, gold, ["Entailment", "Entailment"])
    assert r["map"] == pytest.approx(0.75) and r["nli_binary_accuracy"] == pytest.approx(0.5)
    assert r["p_at_r80"] == pytest.approx(np.mean([1.0, 0.5]))
    assert sc.evidence_map(gold, [p["span_scores"] for p in preds]) == pytest.approx(0.75)


def test_describe_lists_every_public_name():
    text = scilib.describe("contracts")
    for name in sc.__all__:
        assert name in text


def test_code_node_may_import_scilib():
    assert scan_code("from scilib.contracts import fit_predict\n\ndef run(inputs, config):\n    return {}\n") == []


def test_structure_features_and_hypothesis_similarity():
    texts = ["1. DEFINITIONS", "(a) the Receiving Party shall not disclose, copy or use the information;",
             "The Receiving Party shall return all Confidential Information upon termination."]
    F = sc.structure_features(texts)
    assert F.shape == (3, len(sc.STRUCT_COLUMNS))
    col = {c: j for j, c in enumerate(sc.STRUCT_COLUMNS)}
    assert F[0, col["heading_like"]] == 1 and F[1, col["list_marker"]] == 1 and F[1, col["n_negations"]] >= 1
    docs, _ = _synthetic(6, 0)
    feat = sc.Featurizer(min_df=1).fit(docs)
    F = feat.transform(docs)
    n_total = sum(len(d["span_texts"]) for d in docs)
    q = feat.query_scores(F, HYPS["h-return"]["hypothesis"])
    assert set(q) == {"cos_uni", "cos_char", "bm25", "overlap", "idf_cov"} and all(len(v) == n_total for v in q.values())
    flat = [t for d in docs for t in d["span_texts"]]
    top = int(np.argmax(q["bm25"]))
    assert any(w in flat[top].lower() for w in ("return", "destroy"))


@pytest.fixture(scope="module")
def fitted():
    tr, ann = _synthetic(24, 0)
    te, ann_te = _synthetic(10, 1, prefix="t")
    items, gold, labels = _pairs(te, ann_te)
    items = [{**it, "hypothesis": HYPS[it["hypothesis_key"]]["hypothesis"]} for it in items]
    model = sc.ContractModel(n_folds=3, n_estimators=60).fit(tr, ann, HYPS)
    return model, tr, ann, te, items, gold, labels


def test_fit_predict_output_format_and_planted_evidence(fitted):
    model, tr, ann, te, items, gold, labels = fitted
    out = model.predict(items, te)
    assert len(out) == len(items)
    for it, p in zip(items, out):
        assert p["label"] in ("Entailment", "Contradiction")
        s = np.asarray(p["span_scores"])
        assert s.shape == (it["n_spans"],) and np.all(np.isfinite(s)) and s.min() >= 0 and s.max() <= 1
    r = sc.evaluate_predictions(out, gold, labels)
    assert r["map"] > 0.8 and r["nli_binary_accuracy"] > 0.8
    ref = np.mean([sc.average_precision(g, np.random.default_rng(0).random(len(g))) for g in gold])
    assert r["map"] > ref + 0.3


def test_fit_predict_is_deterministic_and_matches_the_class(fitted):
    model, tr, ann, te, items, gold, labels = fitted
    a = sc.fit_predict(tr, ann, HYPS, items[:6], te, n_folds=3, n_estimators=60)
    b = sc.fit_predict(tr, ann, HYPS, items[:6], te, n_folds=3, n_estimators=60)
    assert a == b
    assert [p["span_scores"] for p in a] == [p["span_scores"] for p in model.predict(items[:6], te)]


def test_hypothesis_without_training_evidence_still_gets_lexical_scores(fitted):
    model, tr, ann, te, items, gold, labels = fitted
    d = te[0]
    subpoena = d["span_texts"] + ["The Receiving Party shall notify the Disclosing Party promptly of any subpoena."]
    it = {"doc_id": "x", "hypothesis_key": "h-never", "hypothesis": HYPS["h-never"]["hypothesis"], "n_spans": len(subpoena)}
    (p,) = model.predict([it], [{"doc_id": "x", "span_texts": subpoena}])
    s = np.asarray(p["span_scores"])
    assert s.shape == (len(subpoena),) and int(np.argmax(s)) == len(subpoena) - 1
    assert p["label"] in ("Entailment", "Contradiction")


def test_cross_validate_reports_metrics_per_hypothesis():
    docs, ann = _synthetic(18, 2)
    r = sc.cross_validate(docs, ann, HYPS, n_folds=3, n_estimators=40)
    assert r["n_pairs"] == sum(a["label"] != "NotMentioned" for a in ann)
    assert 0.5 < r["map"] <= 1.0 and set(r["per_hypothesis_map"]) == {"h-return", "h-license"}
    assert 0.0 <= r["nli_binary_accuracy"] <= 1.0 and len(r["aps"]) == r["n_pairs"]


def test_informative_errors(fitted):
    model, tr, ann, te, items, gold, labels = fitted
    with pytest.raises(ValueError, match="doc_id"):
        model.predict([{"doc_id": "missing", "hypothesis_key": "h-return", "n_spans": 3}], te)
    with pytest.raises(ValueError, match="n_spans"):
        model.predict([{**items[0], "n_spans": items[0]["n_spans"] + 1}], te)
    with pytest.raises(ValueError, match="hypothesis_key"):
        model.predict([{"doc_id": te[0]["doc_id"], "n_spans": 3}], te)
    with pytest.raises(ValueError, match="no annotated"):
        sc.ContractModel().fit(tr, [a for a in ann if a["label"] == "NotMentioned"], HYPS)
    with pytest.raises(ValueError, match="span_texts"):
        sc.ContractModel().fit([{"doc_id": "a"}], ann, HYPS)


# ------------------------------------------------------------------------------------------------ pretrained-model features
def _fake_textenc(monkeypatch):
    """Deterministic stand-ins for the three scilib.textenc calls (word-overlap based)."""
    import zlib
    from scilib import textenc as te

    def bag(t):
        v = np.zeros(1024, dtype=np.float32)
        for w in sc.tokenize(t):
            v[zlib.crc32(w.encode()) % 1024] += 1
        return v

    def embed(texts, model="bge_large_en", query=False):
        E = np.stack([bag(t) for t in texts]) if len(texts) else np.zeros((0, 1024), np.float32)
        return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-6)

    def relevance(passages, queries, model="bge_reranker_large"):
        P, Q = embed(passages), embed(queries)
        return (P @ Q.T * 10 - 3).astype(np.float32)

    def nli(premises, hypotheses, model="nli_deberta_v3_large"):
        r = relevance(premises, hypotheses)
        lg = np.stack([-r, r, np.zeros_like(r)], axis=-1)
        return (lg - np.logaddexp.reduce(lg, axis=-1, keepdims=True)).astype(np.float32)

    monkeypatch.setattr(te, "embed", embed)
    monkeypatch.setattr(te, "relevance", relevance)
    monkeypatch.setattr(te, "nli", nli)


def test_plm_groups_are_validated():
    assert sc.ContractModel().plm == ()
    assert sc.ContractModel(plm=True).plm == sc.PLM_GROUPS
    assert sc.ContractModel(plm=["nli"]).plm == ("nli",)
    with pytest.raises(ValueError, match="plm groups"):
        sc.ContractModel(plm=["nope"])


def test_plm_features_fit_predict(monkeypatch):
    _fake_textenc(monkeypatch)
    tr, ann = _synthetic(18, 0)
    te_docs, ann_te = _synthetic(6, 1, prefix="t")
    items, gold, labels = _pairs(te_docs, ann_te)
    items = [{**it, "hypothesis": HYPS[it["hypothesis_key"]]["hypothesis"]} for it in items]
    base = sc.ContractModel(n_folds=3, n_estimators=40).fit(tr, ann, HYPS)
    full = sc.ContractModel(n_folds=3, n_estimators=40, plm=True).fit(tr, ann, HYPS)
    assert full.n_full_ > base.n_full_ and full.n_free_ > base.n_free_
    names = full.feature_names_
    for col in ("rr", "rr_rank", "nli_ent", "nli_ev_dmax", "emb_cos", "e1", "e1_rank", "eknn_max", "eknn_top3"):
        assert col in names
    out = full.predict(items, te_docs)
    for it, p in zip(items, out):
        s = np.asarray(p["span_scores"])
        assert s.shape == (it["n_spans"],) and np.all(np.isfinite(s)) and s.min() >= 0 and s.max() <= 1
    again = sc.ContractModel(n_folds=3, n_estimators=40, plm=True).fit(tr, ann, HYPS).predict_scores(items, te_docs)
    assert all(np.array_equal(a, b) for a, b in zip(full.predict_scores(items, te_docs), again))
    assert sc.evaluate_predictions(out, gold, labels)["map"] > 0.8
    sub = sc.ContractModel(n_folds=3, n_estimators=40, plm=["rerank"]).fit(tr, ann, HYPS)
    assert "rr" in sub.feature_names_ and "nli_ent" not in sub.feature_names_ and "e1" not in sub.feature_names_
    # a hypothesis text without training pairs is scored from the free (hypothesis-text) features
    new = [{"doc_id": te_docs[0]["doc_id"], "hypothesis_key": "h-new", "hypothesis": "Receiving Party must return or destroy materials."}]
    s = sub.predict_scores(new, te_docs)[0]
    assert s.shape == (len(te_docs[0]["span_texts"]),) and np.all(np.isfinite(s))
