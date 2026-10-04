"""bench.metrics: MacroSR, pooled scores, PI (ratio vs linear), ranks, Friedman/Nemenyi, sign test, bootstrap."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from scienceclaw.bench import metrics as M
from scienceclaw.bench.tasks.toy import ToyAdapter


def test_macro_sr_hand_computed():
    z = {"a": [1, 0, 1, 1], "b": [0, 0], "c": []}
    assert M.macro_sr(z) == pytest.approx((0.75 + 0.0) / 2)
    assert M.macro_sr({"a": [1], "b": [1, 1, 0, 0]}) == pytest.approx(0.75)   # macro, not micro (= 3/5)
    assert math.isnan(M.macro_sr({}))
    sr = M.success_rates(z)
    assert sr["a"] == 0.75 and math.isnan(sr["c"])


def test_group_macro():
    vals = {"d1": 1.0, "d2": 0.0, "d3": 0.5, "d4": float("nan")}
    groups = {"d1": "F1", "d2": "F1", "d3": "F2", "d4": "F2"}
    assert M.group_macro(vals, groups) == {"F1": 0.5, "F2": 0.5}


def test_pooled_scores_inline_path_and_missing(tmp_path):
    ad = ToyAdapter()
    (tmp_path / "p.json").write_text(json.dumps({"y_true": [0.0, 0.0], "y_pred": [2.0, 2.0], "y_ref": [0, 0]}))
    results = {"TOY": [
        {"pooled_payload": {"y_true": [0.0, 0.0], "y_pred": [1.0, 1.0], "y_ref": [0.0, 0.0]}},
        {"pooled_payload": "p.json"},                               # relative path -> base_dir
        {"details": {"pooled_payload": None}},                       # no output: excluded
    ]}
    out = M.pooled_scores(results, {"TOY": ad}, base_dir=tmp_path)
    assert out["TOY"] == pytest.approx(math.sqrt((1 + 1 + 4 + 4) / 4))
    assert M.pooled_coverage(results, tmp_path) == {"TOY": (2, 3)}
    assert M.pooled_scores({"TOY": [{}]}, {"TOY": ad}) == {"TOY": None}
    with pytest.raises(KeyError):
        M.pooled_scores({"X": []}, {})


def test_pi_ratio_vs_linear_on_lower_is_better():
    # d_min: RMSE (lower better), reference 2.0; d_max: accuracy (higher better), reference 0.5
    scores = {"A0": {"d_min": 2.0, "d_max": 0.5},
              "B": {"d_min": 1.0, "d_max": 0.6},
              "C": {"d_min": 4.0, "d_max": 0.5}}
    dirs = {"d_min": "min", "d_max": "max"}
    ratio = M.performance_index(scores, "A0", dirs)
    linear = M.performance_index(scores, "A0", dirs, normalization="linear")
    assert ratio.per_discipline["B"]["d_min"] == pytest.approx(200.0)      # 100 * 2 / 1
    assert linear.per_discipline["B"]["d_min"] == pytest.approx(150.0)     # 100 * (2 - 1/2)
    assert ratio.per_discipline["C"]["d_min"] == pytest.approx(50.0)       # 100 * 2 / 4
    assert linear.per_discipline["C"]["d_min"] == pytest.approx(0.0)       # 100 * (2 - 4/2)
    assert ratio.per_discipline["B"]["d_max"] == linear.per_discipline["B"]["d_max"] == pytest.approx(120.0)
    assert ratio["A0"] == linear["A0"] == pytest.approx(100.0)
    assert ratio["B"] == pytest.approx((200 + 120) / 2)
    assert linear["B"] == pytest.approx((150 + 120) / 2)
    assert ratio["C"] == pytest.approx((50 + 100) / 2)
    assert isinstance(ratio, dict) and ratio.reference == "A0" and ratio.normalization == "ratio"
    assert ratio.to_dict()["per_discipline"]["B"]["d_min"] == pytest.approx(200.0)


def test_pi_missing_and_excluded():
    scores = {"ref": {"a": 1.0, "b": 0.0, "c": None}, "m": {"a": None, "b": 1.0, "c": 3.0}}
    dirs = {"a": "max", "b": "min", "c": "max"}
    pi = M.performance_index(scores, "ref", dirs)
    assert pi.disciplines == ["a"]                     # b: ratio needs s_ref > 0; c: no reference score
    assert set(pi.excluded) == {"b", "c"}
    assert pi["m"] == 0.0                              # missing score earns PI_d = 0
    assert math.isnan(M.performance_index(scores, "ref", dirs, missing="skip")["m"])
    with pytest.raises(KeyError):
        M.performance_index(scores, "nope", dirs)
    assert M.pi_value(0.0, 2.0, "min") == 1000.0      # perfect error score capped
    assert M.pi_value(-0.5, 0.5, "max", "linear") == pytest.approx(-100.0)


def test_average_ranks_with_ties_and_missing():
    scores = {"A": {"d1": 0.9, "d2": 1.0}, "B": {"d1": 0.9, "d2": 2.0}, "C": {"d1": 0.5, "d2": 0.5}}
    dirs = {"d1": "max", "d2": "min"}
    r = M.average_ranks(scores, dirs)
    assert r == {"A": pytest.approx(1.75), "B": pytest.approx(2.25), "C": pytest.approx(2.0)}
    t = M.rank_table(scores, dirs)
    assert t["d1"] == {"A": 1.5, "B": 1.5, "C": 3.0}
    # missing scores tie for last place
    r2 = M.average_ranks({"A": {"d": 1.0}, "B": {"d": None}, "C": {}}, {"d": "max"})
    assert r2 == {"A": 1.0, "B": 2.5, "C": 2.5}


def test_friedman_hand_computed():
    scores = {"A": {"d1": 0.9, "d2": 1.0}, "B": {"d1": 0.9, "d2": 2.0}, "C": {"d1": 0.5, "d2": 0.5}}
    fr = M.friedman_test(scores, {"d1": "max", "d2": "min"})
    # chi2_F = 12N/(k(k+1)) * (sum R_j^2 - k(k+1)^2/4) = 24/12 * (12.125 - 12) = 0.25
    assert fr["statistic"] == pytest.approx(0.25) and fr["k"] == 3 and fr["n"] == 2
    from scipy.stats import chi2
    assert fr["p_value"] == pytest.approx(chi2.sf(0.25, 2))
    # Iman-Davenport: (N-1) chi / (N(k-1) - chi) = 0.25 / 3.75
    assert fr["F"] == pytest.approx(0.25 / 3.75)


def test_nemenyi_cd_matches_demsar_table():
    assert M.nemenyi_cd(2, 1) == pytest.approx(1.960 * math.sqrt(2 * 3 / 6), abs=2e-3)
    assert M.nemenyi_cd(3, 23) == pytest.approx(2.343 * math.sqrt(12 / 138), abs=2e-3)
    assert M.nemenyi_cd(4, 10) == pytest.approx(2.569 * math.sqrt(20 / 60), abs=2e-3)
    assert M.nemenyi_cd(3, 23, alpha=0.10) == pytest.approx(2.052 * math.sqrt(12 / 138), abs=2e-3)


def test_sign_test():
    dirs = {f"d{i}": "max" for i in range(5)} | {"e": "min", "t": "max"}
    a = {"d0": 2, "d1": 2, "d2": 2, "d3": 0, "d4": None, "e": 1.0, "t": 5}
    b = {"d0": 1, "d1": 1, "d2": 1, "d3": 1, "d4": 3, "e": 2.0, "t": 5}
    st = M.sign_test(a, b, dirs)
    assert (st["wins"], st["losses"], st["ties"], st["n"]) == (4, 1, 1, 5)
    assert st["p_value"] == pytest.approx(12 / 32)


def test_bootstrap_ci():
    assert M.bootstrap_ci([2.0] * 10) == (2.0, 2.0)
    assert M.bootstrap_ci([3.0]) == (3.0, 3.0)
    assert all(math.isnan(x) for x in M.bootstrap_ci([]))
    vals = [0.0, 1.0] * 50
    lo, hi = M.bootstrap_ci(vals, n=4000, seed=1)
    assert lo < 0.5 < hi and 0.3 < lo and hi < 0.7
    assert M.bootstrap_ci(vals, n=500, seed=3) == M.bootstrap_ci(vals, n=500, seed=3)
    lo_m, hi_m = M.bootstrap_ci(vals, n=500, stat=np.median)
    assert 0.0 <= lo_m <= hi_m <= 1.0
