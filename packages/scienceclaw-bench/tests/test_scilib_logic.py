"""scilib.logic: batched Z3 driver (synthetic QF_NIA scripts; fast)."""
from __future__ import annotations

import pytest

import scilib
from scilib import logic

HEAD = "(set-logic QF_NIA)\n(declare-fun x () Int)\n(declare-fun y () Int)\n"
SAT = HEAD + "(assert (= (* x y) 91))\n(assert (> x 1))\n(assert (> y 1))\n(check-sat)\n"
UNSAT = HEAD + "(assert (< (* x x) 0))\n(check-sat)\n"
# factoring a 40-bit product: not decided within a tiny resource limit
HARD = HEAD + "(assert (= (* x y) 1000036000099))\n(assert (> x 1))\n(assert (> y 1))\n(check-sat)\n"


def test_module_is_described():
    d = scilib.describe("logic")
    assert "scilib.logic" in d and "decide(" in d and "fallback" in d


def test_check_statuses_and_errors():
    assert logic.check(SAT, 10)["status"] == "sat"
    assert logic.check(UNSAT, 10)["status"] == "unsat"
    bad = logic.check("(assert (", 5)
    assert bad["status"] == "unknown" and bad["reason"].startswith("error:")
    rej = logic.check(SAT, 5, params={"no.such_param": 1})
    assert rej["status"] == "unknown" and rej["reason"].startswith("error:")
    lim = logic.check(HARD, 30, rlimit=1)
    assert lim["status"] == "unknown" and lim["reason"] and not lim["reason"].startswith("error:")


def test_solve_all_keeps_order_and_threads():
    r = logic.solve_all([SAT, UNSAT, SAT, UNSAT], threads=2, timeout_s=10)
    assert [x["status"] for x in r] == ["sat", "unsat", "sat", "unsat"]
    assert logic.solve_all([]) == []


def test_majority_status():
    assert logic.majority_status(["sat", "unsat", "unsat"]) == "unsat"
    assert logic.majority_status(["sat", "unsat"]) == "sat"


def test_decide_fallback_and_report():
    qs = [SAT, HARD, UNSAT]
    r = logic.decide(qs, budget_s=20, max_rounds=1, first_rlimit=1, first_timeout_s=5)
    assert r["status"][0] == "unknown" and r["fallback_label"] == "unknown"  # rlimit 1 decides nothing
    r = logic.decide(qs, train_status=["unsat", "unsat", "sat"], budget_s=20, max_rounds=2, first_rlimit=1,
                     factor=10 ** 7, first_timeout_s=5)
    assert r["status"][0] == "sat" and r["status"][2] == "unsat" and r["stage"][0] == 1
    assert r["fallback_label"] == "unsat"          # majority of train_status
    assert r["labels"][0] == "sat" and r["labels"][2] == "unsat"
    assert len(r["labels"]) == len(r["seconds"]) == len(r["reason"]) == 3
    assert r["n_decided"] == sum(s != "unknown" for s in r["status"])
    if r["status"][1] == "unknown":
        assert r["labels"][1] == "unsat"
    assert logic.decide(qs, train_status=["sat"], fallback="unknown", budget_s=20, max_rounds=1,
                        first_rlimit=1)["labels"][1] == "unknown"


def test_decide_budget_and_arguments():
    r = logic.decide([HARD] * 3, budget_s=3, first_rlimit=10 ** 12, first_timeout_s=60, max_rounds=2)
    assert r["wall_s"] < 12
    with pytest.raises(ValueError):
        logic.decide([SAT], fallback="majority")
    with pytest.raises(ValueError):
        logic.decide([SAT], fallback="maybe")
