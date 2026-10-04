"""FoR42 PhysioNet/CinC 2019 sepsis adapter tests (skipped when the data are not available)."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from scienceclaw.bench.tasks import for42_sepsis as m
from scienceclaw.core.trace import Trace

SPLITS = [("src", 7), ("val", 2), ("id", 4), ("ood", 4)]


def test_vectorized_utility_equals_official_loop():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(1, 80))
        labels = np.zeros(n, dtype=int)
        if rng.random() < 0.5:
            labels[int(rng.integers(0, n)):] = 1
        preds = (rng.random(n) < rng.random()).astype(int)
        assert m.prediction_utility(labels, preds) == pytest.approx(m.compute_prediction_utility_official(labels, preds))
        b = m.best_predictions(labels)
        assert m.prediction_utility(labels, b) == pytest.approx(m.compute_prediction_utility_official(labels, b))


def test_known_answers():
    assert m.prediction_utility(np.zeros(10, int), np.ones(10, int)) == pytest.approx(-0.5)
    labels = np.zeros(30, int)
    labels[20:] = 1                       # t_sepsis = 26; optimal window [14, 29]
    best = m.best_predictions(labels)
    assert best.tolist() == [0] * 14 + [1] * 16
    o, bst, ina = m.utility_triplet(labels, best)
    assert o == pytest.approx(bst) and ina < 0
    assert m.normalized_utility(o, bst, ina) == pytest.approx(1.0)
    assert m.normalized_utility(ina, bst, ina) == pytest.approx(0.0)


@pytest.fixture(scope="module")
def adapter():
    a = m.Adapter()
    ok, why = a.available()
    if not ok:
        pytest.skip(why)
    return a


@pytest.fixture(scope="module")
def episodes(adapter):
    return {s: adapter.build_episodes(s, n, seed=7) for s, n in SPLITS}


def _items(eps):
    return {i for e in eps for i in e.lineage["item_ids"]}


def test_splits_disjoint_stratified_deterministic(adapter, episodes):
    items = {s: _items(eps) for s, eps in episodes.items()}
    for a, b in combinations(items, 2):
        assert not items[a] & items[b], (a, b)
    pools = adapter._pools()
    visible = set(pools["train"]) | set(pools["dev"])
    assert not visible & set().union(*items.values())
    assert not set(pools["train"]) & set(pools["dev"])
    stays = adapter._stays()
    for s, eps in episodes.items():
        for e in eps:
            ids = e.lineage["item_ids"]
            assert len(ids) == len(set(ids)) == 16 and e.n_items == 16
            assert sum(stays[p]["septic"] for p in ids) == 2
            assert {stays[p]["hospital"] for p in ids} == ({"B"} if s == "ood" else {"A"})
            assert e.lineage["ood_kind"] == ("cross_hospital_within_dataset" if s == "ood" else None)
    again = adapter.build_episodes("id", 4, seed=7)
    assert [e.lineage["item_ids"] for e in again] == [e.lineage["item_ids"] for e in episodes["id"]]
    assert [e.id for e in again] == [e.id for e in episodes["id"]]
    other = adapter.build_episodes("id", 4, seed=8)
    assert _items(other) <= set(pools["id"])                 # pools do not depend on the build seed
    rep = adapter.build_episodes("rep", 2, seed=7)
    assert [e.lineage["item_ids"] for e in rep] == [e.lineage["item_ids"] for e in episodes["src"][:2]]
    assert all(e.split == "rep" for e in rep)


def test_tools_expose_only_visible_data(adapter, episodes):
    e = episodes["ood"][1]
    ev = e.tool("load_eval_inputs").fn({}, {})
    assert "SepsisLabel" not in ev["table"].columns
    assert ev["patient_ids"] == e.lineage["item_ids"]
    assert ev["table"].groupby("patient_id", sort=False).size().loc[ev["patient_ids"]].tolist() == ev["n_hours"]
    tr = e.tool("load_train").fn({}, {})
    assert "SepsisLabel" in tr["table"].columns
    assert not set(tr["table"]["patient_id"]) & set(ev["patient_ids"])
    dv = e.tool("load_dev_inputs").fn({}, {})
    assert "SepsisLabel" not in dv["table"].columns
    sd = e.tool("score_dev").fn({"predictions": [np.zeros(h, int) for h in dv["n_hours"]]}, {})
    assert sd["normalized_utility"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        e.tool("score_dev").fn({"predictions": [np.zeros(1, int)]}, {})
    pub = e.public_view()
    assert "SepsisLabel is 1" in pub["objective"] and "item_ids" not in str(pub)


def test_evaluator_reference_oracle_malformed(adapter, episodes):
    details = []
    for e in (episodes["id"][0], episodes["ood"][0]):
        pids = e.lineage["item_ids"]
        r = e.evaluate(adapter.reference_predictions(pids), None)
        assert r.completed and r.hard_ok()
        assert r.primary == pytest.approx(r.details["reference"])
        assert not r.accepted and r.z == 0 and r.details["norm_score"] == pytest.approx(1.0)
        oracle = [m.best_predictions(adapter._labels(p)) for p in pids]
        ro = e.evaluate(oracle, None)
        assert ro.primary == pytest.approx(1.0) and ro.accepted and ro.z == 1
        details.append(ro.details)
        zeros = e.evaluate([np.zeros(h, int) for h in e.tool("load_eval_inputs").fn({}, {})["n_hours"]], None)
        assert zeros.primary == pytest.approx(0.0)
        bad_shape = e.evaluate([np.zeros(3, int)] * 16, None)
        assert not bad_shape.h["output_structure"] and bad_shape.z == 0 and bad_shape.primary is None
        assert bad_shape.details["pooled_payload"]["fallback"]
        bad_vals = e.evaluate([2 * o for o in oracle], None)
        assert bad_vals.h["output_structure"] and not bad_vals.h["binary_values"] and bad_vals.z == 0
    assert adapter.pooled_metric(details) == pytest.approx(1.0)
    assert adapter.pooled_metric([]) is None


# ------------------------------------------------------------------------------------------------ LEAK-1: causality
def _stay_lengths(n=8):
    return [30 + 4 * i for i in range(n)]


def test_positional_only_flags_length_and_position_shortcuts():
    lens = _stay_lengths()

    def last(k):
        return [np.r_[np.zeros(n - k, int), np.ones(k, int)] for n in lens]

    def first(k):
        return [np.r_[np.ones(k, int), np.zeros(n - k, int)] for n in lens]

    assert m.positional_only(last(12))[0]                                   # "flag the last 12 hours"
    assert m.positional_only(last(1))[0]
    assert m.positional_only(first(6))[0]                                   # start-frame position rule
    assert m.positional_only([np.ones(n, int) for n in lens])[0]            # flag everything
    assert not m.positional_only([np.zeros(n, int) for n in lens])[0]       # never predicting is not positional
    mixed = [p if i % 3 == 0 else np.zeros_like(p) for i, p in enumerate(last(12))]
    assert not m.positional_only(mixed)[0]                                  # position is split between stays: patient-specific
    assert not m.positional_only(last(12)[:m.POSITION_MIN_STAYS - 1])[0]    # too few stays to say anything
    ok, msg = m.positional_only(last(12))
    assert "distance to the end" in msg


def test_probe_keeps_are_deterministic_strict_and_hidden_hash_based():
    pids = [f"p{i:06d}" for i in range(40)]
    hours = [int(h) for h in np.random.default_rng(0).integers(9, 120, size=40)]
    a = m.probe_keeps(7, pids, hours)
    assert a == m.probe_keeps(7, pids, hours) and a != m.probe_keeps(8, pids, hours)
    for kp, n in zip(a, hours):
        assert 1 <= kp <= n - 1                                             # strictly shorter than the full stay
        assert kp >= min(m.PROBE_MIN_ROWS, n - 1)
        if n >= 20:
            assert 0.35 * n <= kp <= 0.85 * n
    assert m.probe_keeps(7, ["x", "y"], [1, 5]) == [1, 4]                   # a single-row stay cannot be cut


def test_prefix_mismatch_counts_only_the_prefix_hours():
    full = [np.array([0, 1, 1, 0, 1]), np.array([1, 1, 0, 0])]
    assert m.prefix_mismatch(full, [np.array([0, 1, 1]), np.array([1, 1])]) == (0, 5)
    assert m.prefix_mismatch(full, [np.array([0, 0, 1]), np.array([0, 1])]) == (2, 5)
    assert m.prefix_mismatch(full, [np.array([], int), np.array([], int)]) == (0, 0)


def test_acceptance_needs_positive_utility(adapter, episodes):
    neg = 0
    for s, eps in episodes.items():
        if s == "src":
            continue
        for e in eps:
            ref = e.evaluate(adapter.reference_predictions(e.lineage["item_ids"]), None).details["reference"]
            zeros = e.evaluate([np.zeros(h, int) for h in e.tool("load_eval_inputs").fn({}, {})["n_hours"]], None)
            assert zeros.primary == pytest.approx(0.0) and not zeros.accepted and zeros.z == 0
            assert zeros.details["acceptance_floor"] == 0.0
            if ref < -m.ACCEPT_MARGIN:                                      # would have been accepted without the floor
                neg += 1
                assert zeros.primary > ref + m.ACCEPT_MARGIN
    assert neg >= 1, "expected at least one episode whose reference utility is below -ACCEPT_MARGIN"
    assert "max(reference" in episodes["id"][0].acceptance


def _run_rule(derived, rule):
    """Stand-in for the submitted graph: apply ``rule(list of per-stay frames)`` on the derived episode's rows."""
    ev = derived.tool("load_eval_inputs").fn({}, {})
    df, hours = ev["table"], ev["n_hours"]
    frames, pos = [], 0
    for n in hours:
        frames.append(df.iloc[pos:pos + n])
        pos += n
    return [np.asarray(a, dtype=int) for a in rule(frames)], Trace()


def _causal_rule(frames):                       # per-hour rule on LOCF heart rate: uses rows 0..t only
    return [(g["HR"].ffill().fillna(0.0).to_numpy() > 90).astype(int) for g in frames]


def _look_ahead_rule(frames):                   # flags the last 12 hours of every other stay: uses the stay length
    return [np.r_[np.zeros(max(len(g) - 12, 0), int), np.ones(min(12, len(g)), int)] if i % 2 == 0
            else np.zeros(len(g), int) for i, g in enumerate(frames)]


def test_causal_prefix_probe_accepts_causal_and_rejects_look_ahead(adapter, episodes):
    e = episodes["id"][0]
    y_causal, _ = _run_rule(e, _causal_rule)
    tr = Trace()
    out = e.run_probes(y_causal, tr, lambda d, name: _run_rule(d, _causal_rule))
    assert out["causal_prefix"]["ok"], out
    res = e.evaluate(y_causal, tr)
    assert res.h["causal_prefix"] and res.h["not_positional_only"] and res.hard_ok()
    assert "0/" in res.h_msgs["causal_prefix"] or "prefix hours changed" in res.h_msgs["causal_prefix"]

    y_bad, _ = _run_rule(e, _look_ahead_rule)
    tr2 = Trace()
    out2 = e.run_probes(y_bad, tr2, lambda d, name: _run_rule(d, _look_ahead_rule))
    assert not out2["causal_prefix"]["ok"], out2
    res2 = e.evaluate(y_bad, tr2)
    assert res2.h["not_positional_only"]                                    # split between stays: only the probe sees it
    assert not res2.h["causal_prefix"] and res2.z == 0 and not res2.hard_ok()
    assert tr2.to_dict()["probes"]["causal_prefix"]["ok"] is False

    # the same graph crashing on the shortened stays fails the probe too
    def crash(d, name):
        raise RuntimeError("needs the full stay")

    assert not e.run_probes(y_causal, Trace(), crash)["causal_prefix"]["ok"]


def test_probe_derived_episode_is_a_pure_truncation(adapter, episodes):
    e = episodes["val"][0]
    y = [np.zeros(h, int) for h in e.tool("load_eval_inputs").fn({}, {})["n_hours"]]
    (probe,) = e.probe_specs(y)
    assert probe.name == "causal_prefix" and e.probe_specs(None) == [] and e.probe_specs("garbage") == []
    full = e.tool("load_eval_inputs").fn({}, {})
    seen = {}

    def runner(derived, name):
        seen["ev"] = derived.tool("load_eval_inputs").fn({}, {})
        seen["derived"] = derived
        return [np.zeros(h, int) for h in seen["ev"]["n_hours"]], Trace()

    out = e.run_probes(y, Trace(), runner)
    assert out["causal_prefix"]["ok"]
    cut, d = seen["ev"], seen["derived"]
    assert cut["patient_ids"] == full["patient_ids"]
    assert all(1 <= c < n for c, n in zip(cut["n_hours"], full["n_hours"]))
    assert len(cut["table"]) == sum(cut["n_hours"]) and "SepsisLabel" not in cut["table"].columns
    starts_full = np.r_[0, np.cumsum(full["n_hours"])[:-1]]
    starts_cut = np.r_[0, np.cumsum(cut["n_hours"])[:-1]]
    for i, (c, sf, sc) in enumerate(zip(cut["n_hours"], starts_full, starts_cut)):
        a = full["table"].iloc[sf:sf + c].reset_index(drop=True)
        b = cut["table"].iloc[sc:sc + c].reset_index(drop=True)
        assert a.equals(b), i
    assert d._evaluate is None and d._probes is None and d.id.endswith("#causal_prefix")
    assert d.required_output is e.required_output or d.required_output == e.required_output
    ok_shape, _ = d.constraints[0].check([np.zeros(c, int) for c in cut["n_hours"]], None)
    assert ok_shape and not d.constraints[0].check(y, None)[0]              # expects the cut lengths, not the full ones


def test_causal_constraint_visibility_and_default(adapter, episodes):
    e = episodes["id"][1]
    pids = e.lineage["item_ids"]
    pub = e.public_view()
    joined = " ".join(pub["constraints"])
    assert "not_positional_only" in joined and "causal_prefix" not in joined and "probe" not in str(pub).lower()
    assert "n_hours[i] only sets the length" in pub["objective"]
    hidden = {c.name for c in e.constraints if not c.visible}
    assert hidden == {"causal_prefix"}
    r = e.evaluate(adapter.reference_predictions(pids), None)               # never probed -> passes
    assert r.h["causal_prefix"] and r.h_msgs["causal_prefix"] == "not probed"
    r2 = e.evaluate(adapter.reference_predictions(pids), Trace())
    assert r2.h["causal_prefix"]


def test_positional_shortcut_is_rejected_by_the_visible_constraint(adapter, episodes):
    for e in (episodes["id"][0], episodes["ood"][1]):
        hours = e.tool("load_eval_inputs").fn({}, {})["n_hours"]
        last12 = [np.r_[np.zeros(max(n - 12, 0), int), np.ones(min(12, n), int)] for n in hours]
        r = e.evaluate(last12, None)
        assert not r.h["not_positional_only"] and r.z == 0 and not r.hard_ok()
        first6 = [np.r_[np.ones(6, int), np.zeros(n - 6, int)] for n in hours]
        assert not e.evaluate(first6, None).h["not_positional_only"]
        ref = e.evaluate(adapter.reference_predictions(e.lineage["item_ids"]), None)
        assert ref.h["not_positional_only"]                                 # the logistic-regression reference is patient-specific
        malformed = e.evaluate([np.zeros(3, int)] * 16, None)
        assert malformed.h_msgs["not_positional_only"].startswith("n/a")


def test_objective_documents_the_domain_library_and_states_the_data_facts(adapter, episodes):
    import re
    e = episodes["src"][0]
    obj = e.public_view()["objective"]
    assert "scilib.sepsis" in obj and "fit_predict" in obj
    assert "EtCO2 0% (always NaN)" in obj                                    # missing-rate facts from the visible training stays
    assert re.search(r"HR \d+%", obj) and "ICULOS 100%" in obj
    assert "contain 30 septic stays" in obj and "evaluation stays contain" not in obj   # visible composition only
    assert "rows 0..t" in obj
    specs = {s.name: s.description for s in e.tools}
    for name in ("load_eval_inputs", "load_dev_inputs", "score_dev"):
        assert "rows 0..t" in specs[name] or "same causal rule" in specs[name]
    # nothing that states or hints the reference, the acceptance margin or a preferred recipe
    texts = [obj, *specs.values()]
    dv = e.tool("load_dev_inputs").fn({}, {})
    sd = e.tool("score_dev").fn({"predictions": [np.zeros(h, int) for h in dv["n_hours"]]}, {})
    assert set(sd) == {"normalized_utility", "n_stays"}                      # score_dev returns the plain dev score only
    texts.append(str(sd))
    for t in texts:
        low = t.lower()
        for word in ("reference", "baseline", "margin", "accept", "best recipe", "you should"):
            assert word not in low, (word, t[:80])
