"""FoR46 HumanEval -> MBPP adapter: v2 roles, splits, leakage, evaluator (reference / oracle), sandbox, constraints."""
from __future__ import annotations

import json

import pytest

from scienceclaw.bench.tasks import for46_code as m

ADAPTER = m.Adapter()
OK, WHY = ADAPTER.available()
pytestmark = pytest.mark.skipif(not OK, reason=f"FoR46 data unavailable: {WHY}")

COUNTS = {"src": 7, "val": 2, "id": 4, "ood": 4}          # the default config (rounds=7, n_val=2, n_id=4, n_ood=4)
V2 = pytest.mark.skipif(not OK or ADAPTER._data.get().roles != m.ROLES_V2, reason="needs data-team reconstructed_v2")


@pytest.fixture(scope="module")
def episodes():
    return {s: ADAPTER.build_episodes(s, n, seed=1000 + i) for i, (s, n) in enumerate(COUNTS.items())}


def _ids(eps):
    return [i for e in eps for i in e.lineage["item_ids"]]


# ------------------------------------------------------------------------------------------------ roles
def test_pool_sizes_and_capacity():
    pools = ADAPTER._data.get().pools
    assert {k: len(v) for k, v in pools.items()} == {"src": 36, "val": 64, "id": 64, "ood": 64, "ood_extra": 64}
    assert [ADAPTER.max_disjoint_episodes(s) for s in ("src", "val", "id", "ood")] == [2, 4, 4, 8]
    assert [ADAPTER.max_disjoint_episodes(s, 8) for s in ("src", "val", "id", "ood")] == [4, 8, 8, 16]
    assert all(i.startswith("HumanEval/") for k in ("src", "val", "id") for i in pools[k])
    assert all(i.startswith("MBPP/") for k in ("ood", "ood_extra") for i in pools[k])
    allv = [i for v in pools.values() for i in v]
    assert len(allv) == len(set(allv)) == 292                 # role pools are pairwise disjoint


@V2
def test_v2_manifest_and_role_files_agree_with_pools():
    data = ADAPTER._data.get()
    root = ADAPTER.root / m.DATASET_DIR / m.ROLES_V2
    man = json.loads((root / "sampling-manifest.json").read_text(encoding="utf-8"))["partitions"]
    role_of = {"src": "source_train", "val": "validation", "id": "heldout_id", "ood": "heldout_ood",
               "ood_extra": "heldout_extra"}
    for pool, role in role_of.items():
        rows = [json.loads(x) for x in (root / man[role]["data_file"]).read_text(encoding="utf-8").splitlines() if x]
        ids = [r["task_id"] if isinstance(r["task_id"], str) else f"MBPP/{int(r['task_id'])}" for r in rows]
        assert ids == data.pools[pool] == [f"MBPP/{int(x)}" if pool.startswith("ood") else str(x)
                                           for x in man[role]["selected_ids"]], role
        assert man[role]["count"] == len(ids)
        for r, iid in zip(rows, ids):                          # the raw dataset files carry the same content
            p = data.problems[iid]
            if p.dataset == "HumanEval":
                assert (r["prompt"], r["test"], r["entry_point"], r["canonical_solution"]) == (
                    p.prompt, p.hidden_test, p.entry_point, p.canonical), iid
            else:
                setup, tests = m.hidden_tests(p)
                assert r["test_list"] == tests and "\n".join(r.get("test_imports") or []) == setup, iid
                assert r["code"] == p.canonical and r["prompt"].strip() == p.prompt, iid
                assert 11 <= int(iid.split("/")[1]) <= 510      # MBPP official test partition
    # D_rep ("replication") is the already-seen source prefix, never a pool of new items
    assert [str(x) for x in man["replication"]["selected_ids"]] == data.pools["src"]
    assert man["replication"]["data_file"] == "retention.jsonl"
    assert data.receipt["roles_version"] == m.ROLES_V2 and len(data.receipt["roles_manifest_sha256"]) == 64


def test_role_selection_fallbacks_and_malformed_manifest(tmp_path):
    data = ADAPTER._data.get()
    d = tmp_path / "ds"
    (d / m.ROLES_V2).mkdir(parents=True)
    pools, label, _info = m._select_roles(d, data.problems, None)          # nothing there -> hash ranking
    assert label == m.ROLES_HASH and [len(pools[k]) for k in ("src", "val", "id", "ood", "ood_extra")] == [36, 64, 64, 64, 64]
    with pytest.raises(ValueError, match="requested"):
        m._select_roles(d, data.problems, "v2")
    p = data.pools

    def write(parts):
        (d / m.ROLES_V2 / "sampling-manifest.json").write_text(json.dumps({"partitions": parts}))

    def part(ids, mbpp=False):
        return {"selected_ids": [i.split("/")[1] if mbpp else i for i in ids], "count": len(ids)}

    good = {"source_train": part(p["src"]), "validation": part(p["val"]), "heldout_id": part(p["id"]),
            "heldout_ood": part(p["ood"], True), "heldout_extra": part(p["ood_extra"], True)}
    write(good)
    pools, label, info = m._select_roles(d, data.problems, None)
    assert label == m.ROLES_V2 and pools == {k: p[k] for k in ("src", "val", "id", "ood", "ood_extra")}
    assert len(info["roles_manifest_sha256"]) == 64
    write({**good, "heldout_extra": part(p["ood_extra"][:-1] + [p["ood"][0]], True)})     # overlapping roles
    with pytest.raises(ValueError, match="overlap"):
        m._select_roles(d, data.problems, None)
    write({**good, "heldout_ood": {**part(p["ood"], True), "count": 3}})                 # count mismatch
    with pytest.raises(ValueError, match="malformed"):
        m._select_roles(d, data.problems, None)
    write({**good, "validation": part(p["val"][:-1] + ["HumanEval/9999"])})               # unknown problem
    with pytest.raises(ValueError, match="unknown"):
        m._select_roles(d, data.problems, None)
    write({k: v for k, v in good.items() if k != "heldout_id"})                          # missing role
    with pytest.raises(ValueError, match="malformed"):
        m._select_roles(d, data.problems, None)
    with pytest.raises(ValueError, match="roles must be"):
        m._select_roles(d, data.problems, "v3")


def test_hash_and_v1_role_fallbacks_build_disjoint_episodes():
    for roles in ("hash", "v1"):
        a = m.Adapter(roles=roles)
        try:
            ok, why = a.available()
        except Exception as ex:                                # pragma: no cover
            pytest.fail(f"{roles}: {ex}")
        if not ok:
            pytest.skip(f"{roles}: {why}")
        eps = {s: a.build_episodes(s, n, seed=5) for s, n in COUNTS.items()}
        sets = {s: set(_ids(e)) for s, e in eps.items()}
        for x in sets:
            for y in sets:
                if x < y:
                    assert not (sets[x] & sets[y]), (roles, x, y)
        assert eps["id"][0].lineage["roles"] == a._data.get().roles


# ------------------------------------------------------------------------------------------------ splits
def test_counts_and_disjointness(episodes):
    for s, eps in episodes.items():
        assert len(eps) == COUNTS[s]
        for e in eps:
            assert e.n_items == 16 and len(e.lineage["item_ids"]) == 16
            assert len(set(e.lineage["item_ids"])) == 16            # no repeat inside an episode
    sets = {s: set(_ids(eps)) for s, eps in episodes.items()}
    for a in sets:
        for b in sets:
            if a < b:
                assert not (sets[a] & sets[b]), (a, b)
    for s in ("val", "id", "ood"):                                    # held-out splits never repeat items
        assert len(_ids(episodes[s])) == len(sets[s])
    assert all(i.startswith("HumanEval/") for s in ("src", "val", "id") for i in sets[s])
    assert all(i.startswith("MBPP/") for i in sets["ood"])
    assert all(e.lineage["ood_kind"] == "cross_dataset" for e in episodes["ood"])
    pools = ADAPTER._data.get().pools
    for s in ("src", "val", "id"):
        assert sets[s] <= set(pools[s])


def test_all_capacity_episodes_are_disjoint_across_splits():
    """The largest disjoint plan (val 4, id 4, ood 8) plus the 7 default source episodes share no problem."""
    plan = {"src": 7, "val": 4, "id": 4, "ood": 8}
    eps = {s: ADAPTER.build_episodes(s, n, seed=11) for s, n in plan.items()}
    for s in ("val", "id", "ood"):
        ids = _ids(eps[s])
        assert len(ids) == len(set(ids)) == plan[s] * 16
    sets = {s: set(_ids(e)) for s, e in eps.items()}
    for a in sets:
        for b in sets:
            if a < b:
                assert not (sets[a] & sets[b]), (a, b)
    assert len(sets["val"]) == 64 and len(sets["id"]) == 64 and len(sets["ood"]) == 128


def test_source_pool_is_recycled_for_seven_rounds():
    eps = ADAPTER.build_episodes("src", 7, seed=3)
    pool = set(ADAPTER._data.get().pools["src"])
    assert len(pool) == 36 and all(set(e.lineage["item_ids"]) <= pool for e in eps)
    assert all(len(set(e.lineage["item_ids"])) == 16 for e in eps)
    assert [e.lineage["src_cycle"] for e in eps] == [0, 0, 1, 1, 2, 2, 3]
    for c in range(3):                                               # the 2 episodes of a cycle are item-disjoint
        a, b = eps[2 * c].lineage["item_ids"], eps[2 * c + 1].lineage["item_ids"]
        assert not set(a) & set(b)
    assert all(e.lineage["src_items_recycled"] for e in eps)
    assert ADAPTER.build_episodes("id", 1, seed=3)[0].lineage["src_cycle"] is None
    small = ADAPTER.build_episodes("src", 4, seed=3, items_per_episode=9)      # 4 disjoint episodes of 9 = all 36
    assert len(set(_ids(small))) == 36


def test_split_plan_builds_with_default_and_extended_configs():
    """SplitPlan (lineage check, unique episode ids, rep copies) accepts rounds=7 and the largest disjoint plan."""
    from types import SimpleNamespace

    from scienceclaw.bench.splits import SplitPlan
    for rounds, nv, ni, no in ((7, 2, 4, 4), (2, 2, 2, 2), (7, 4, 4, 8)):
        cfg = SimpleNamespace(disciplines=["FoR46"], seed=20260928, items_per_episode=16, rounds=rounds,
                              n_val=nv, n_id=ni, n_ood=no)
        plan = SplitPlan.build(cfg, {"FoR46": ADAPTER})
        got = {s: len(plan.episodes[s]["FoR46"]) for s in ("src", "val", "id", "ood", "rep")}
        assert got == {"src": rounds, "val": nv, "id": ni, "ood": no, "rep": rounds}
        ids = [e.id for s in ("src", "val", "id", "ood") for e in plan.episodes[s]["FoR46"]]
        assert len(ids) == len(set(ids))
        reused = [w for w in plan.warnings if "reused by several episodes" in w]
        assert all("/src:" in w for w in reused)                       # only the recycled source split repeats items
        assert bool(reused) == (rounds > 2)


def test_ood_uses_role_pool_then_reserve(episodes):
    pools = ADAPTER._data.get().pools
    eps = ADAPTER.build_episodes("ood", 8, seed=1003)
    assert [e.lineage["ood_layer"] for e in eps] == ["role"] * 4 + ["reserve"] * 4
    assert set(_ids(eps[:4])) == set(pools["ood"]) and set(_ids(eps[4:])) == set(pools["ood_extra"])
    assert [e.lineage["item_ids"] for e in eps[:4]] == [e.lineage["item_ids"] for e in episodes["ood"]]
    assert all(e.lineage["ood_layer"] == "role" for e in episodes["ood"])
    with pytest.raises(m.PoolExhausted):
        ADAPTER.build_episodes("ood", 9, seed=1)


def test_determinism_and_prefix_stability():
    a = ADAPTER.build_episodes("id", 4, seed=7)
    b = ADAPTER.build_episodes("id", 4, seed=7)
    c = ADAPTER.build_episodes("id", 2, seed=7)
    assert [e.lineage["item_ids"] for e in a] == [e.lineage["item_ids"] for e in b]
    assert [e.lineage["item_ids"] for e in a[:2]] == [e.lineage["item_ids"] for e in c]
    assert [e.id for e in a] == [e.id for e in b]
    d = ADAPTER.build_episodes("id", 4, seed=8)
    assert [e.lineage["item_ids"] for e in a] != [e.lineage["item_ids"] for e in d]
    for split, big, small in (("src", 7, 3), ("ood", 8, 5), ("ood", 8, 4), ("val", 4, 1)):     # incl. reserve boundary
        x = ADAPTER.build_episodes(split, big, seed=9)
        y = ADAPTER.build_episodes(split, small, seed=9)
        assert [e.lineage["item_ids"] for e in x[:small]] == [e.lineage["item_ids"] for e in y], (split, small)
        assert [e.id for e in x[:small]] == [e.id for e in y]


def test_pool_exhaustion_is_reported():
    for split, n in (("id", 5), ("val", 5), ("ood", 9)):
        with pytest.raises(m.PoolExhausted):
            ADAPTER.build_episodes(split, n, seed=1)
    with pytest.raises(m.PoolExhausted):
        ADAPTER.build_episodes("src", 1, seed=1, items_per_episode=37)
    with pytest.raises(ValueError):
        ADAPTER.build_episodes("id", 1, seed=1, items_per_episode=0)
    with pytest.raises(ValueError):
        ADAPTER.build_episodes("rep", 1, seed=1)


# ------------------------------------------------------------------------------------------------ visibility
def test_tools_expose_no_hidden_tests(episodes):
    data = ADAPTER._data.get()
    for s in ("src", "ood"):
        e = episodes[s][0]
        out = e.tool("load_eval_inputs").fn({}, {})
        blob = json.dumps(out)
        assert len(out["problems"]) == e.n_items
        assert all(set(p) == {"prompt", "entry_point", "visible_tests", "kind"} for p in out["problems"])
        for iid in e.lineage["item_ids"]:
            p = data.problems[iid]
            assert p.canonical.strip() not in blob
            if p.dataset == "HumanEval":
                assert "def check(" not in blob and "candidate(" not in blob
            else:
                _setup, hidden = m.hidden_tests(p)
                for t in hidden[1:]:
                    if t not in p.visible_tests:
                        assert t not in blob
        assert "check(" not in json.dumps([t.signature() for t in e.tools])
    for iid in ADAPTER._data.get().pools["ood"] + ADAPTER._data.get().pools["ood_extra"]:
        assert len(data.problems[iid].visible_tests) == 1              # MBPP: only the first assert is public


def _oracle(e):
    data = ADAPTER._data.get()
    return [data.problems[i].canonical for i in e.lineage["item_ids"]]


def test_oracle_and_reference(episodes):
    data = ADAPTER._data.get()
    reserve = ADAPTER.build_episodes("ood", 5, seed=1003)[4]
    for e in (episodes["id"][0], episodes["ood"][0], episodes["val"][0], episodes["src"][0], reserve):
        y = _oracle(e)
        ev = e.evaluate(y, None)
        assert ev.primary == 1.0 and ev.hard_ok() and ev.accepted and ev.z == 1
        assert ev.details["norm_score"] == pytest.approx(min(10.0, 1.0 / max(ev.details["reference"], 0.1)))
        assert ev.details["pooled_payload"]["passed"] == [1] * 16
        ref_y = [m.memorizer_code(data.problems[i]) for i in e.lineage["item_ids"]]
        rv = e.evaluate(ref_y, None)
        assert rv.primary == pytest.approx(ev.details["reference"])
        assert not rv.accepted and rv.z == 0
        dev = e.dev_evaluate(y)
        assert dev["visible_pass_rate"] == 1.0
        tv = e.tool("run_visible_tests").fn({"codes": y}, {"test_timeout_s": 5})
        assert tv["visible_pass_rate"] == 1.0 and all(tv["passed"])


def test_empty_and_wrong_solutions_score_zero(episodes):
    for e in (episodes["id"][0], episodes["ood"][0]):
        ev = e.evaluate([""] * 16, None)
        assert ev.primary == 0.0 and not ev.accepted and ev.z == 0 and not ev.hard_ok()
        assert ev.details["pooled_payload"]["passed"] == [0] * 16
        ev = e.evaluate(["pass\n"] * 16, None)
        assert ev.primary == 0.0 and ev.details["pooled_payload"]["passed"] == [0] * 16


def test_wrong_code_fails_hidden_tests(episodes):
    e = episodes["id"][1]
    data = ADAPTER._data.get()
    y = _oracle(e)
    p0 = data.problems[e.lineage["item_ids"][0]]
    y[0] = f"def {p0.entry_point}(*a, **k):\n    raise RuntimeError('nope')\n"
    y[1] = f"def {data.problems[e.lineage['item_ids'][1]].entry_point}(*a, **k):\n    while True:\n        pass\n"
    ev = e.evaluate(y, None)
    assert ev.primary == pytest.approx(14 / 16)
    assert ev.details["pooled_payload"]["passed"][:2] == [0, 0]
    assert ev.hard_ok()


def test_constraints_fire(episodes):
    e = episodes["val"][0]
    y = _oracle(e)
    h, _ = e.check_constraints(y[:-1], None)
    assert not h["output_format"]
    h, _ = e.check_constraints(y[:-1] + [3], None)
    assert not h["output_format"]
    bad = list(y)
    bad[0] = "def broken(:\n"
    h, msgs = e.check_constraints(bad, None)
    assert h["output_format"] and not h["implemented"] and "syntax" in msgs["implemented"]
    ev = e.evaluate(y[:-1], None)
    assert ev.primary == 0.0 and not ev.hard_ok() and ev.z == 0
    assert ev.details["pooled_payload"]["passed"] == [0] * 16


def test_pooled_metric():
    pay = [{"item_ids": ["a", "b"], "passed": [1, 0]}, {"item_ids": ["c", "d"], "passed": [1, 1]},
           {"item_ids": ["e"], "passed": None}]
    assert ADAPTER.pooled_metric(pay) == pytest.approx(3 / 5)
    assert ADAPTER.pooled_metric([]) is None


def test_visible_example_extraction():
    prompt = ('def f(x, y):\n    """Add.\n    f(1, 2) ➞ 3\n    f([1], [2]) == [1, 2]\n'
              '    f(a, b) => 3\n    """\n')
    tests, pairs = m.humaneval_visible(prompt, "f")
    assert tests == ["assert f(1, 2) == 3", "assert f([1], [2]) == [1, 2]"]
    assert [p[:2] for p in pairs] == [("(1, 2)", "3"), ("([1], [2])", "[1, 2]")]
    doc = 'def g(s):\n    """\n    >>> g("ab")\n    \'ba\'\n    """\n'
    assert m.humaneval_visible(doc, "g")[0] == ["assert g(\"ab\") == 'ba'"]


# ------------------------------------------------------------------------------------------------ sandbox
def test_sandbox_hardening():
    r = m.run_program("import os\nFILES = sorted(os.listdir('.'))\n", ["assert FILES == []"], 5)
    assert r["passed"]                                             # no job / result file in the working directory
    fake = ("import os, json\nopen('result.json', 'w').write(json.dumps({'passed': True, 'compiled': True, "
            "'tests': []}))\nos._exit(0)\n")
    assert not m.run_program(fake, ["assert False"], 5)["passed"]
    r = m.run_program("import sys\nsys.exit(0)\n", ["assert True"], 5)
    assert not r["passed"] and "SystemExit" in (r["error"] or "")
    r = m.run_program("x = 1\n", ["while True:\n    pass", "assert x == 1"], 1)
    assert not r["passed"] and r["tests"][0][0] is False and r["tests"][1][0] is True


def test_sandbox_blocks_network_and_bounds_resources():
    net = ("import socket\nRES = []\n"
           "for f in (lambda: socket.create_connection(('127.0.0.1', 9), 1),\n"
           "          lambda: socket.socket().connect(('127.0.0.1', 9)),\n"
           "          lambda: socket.getaddrinfo('example.com', 80)):\n"
           "    try:\n        f()\n        RES.append('open')\n"
           "    except OSError as e:\n        RES.append('blocked')\n")
    r = m.run_program(net, ["assert RES == ['blocked'] * 3, RES"], 5)
    assert r["passed"] and r["sandbox"] in ("seatbelt", "none")
    r = m.run_program("import os\nos.system('echo hi')\n", ["assert True"], 5)     # destructive os calls are disabled
    assert not r["passed"]
    r = m.run_program("x = 1\n", ["while True:\n    pass"], 1)                    # infinite loop -> per-test timeout
    assert not r["passed"] and r["tests"] and r["tests"][0][0] is False
    if r["sandbox"] == "seatbelt":                                                 # OS boundary (macOS): no writes
        w = "try:\n    open('/tmp/.sc46_escape_probe', 'w').close()\n    W = 'wrote'\nexcept OSError:\n    W = 'denied'\n"
        assert m.run_program(w, ["assert W == 'denied', W"], 5)["passed"]
        assert m.run_program("open('local.txt', 'w').write('x')\n", ["assert True"], 5)["passed"]


def test_sandbox_can_be_disabled_by_env(monkeypatch):
    monkeypatch.setattr(m, "_seatbelt_state", None)
    monkeypatch.setenv(m._SEATBELT_ENV, "0")
    r = m.run_program("x = 1\n", ["assert x == 1"], 5)
    assert r["passed"] and r["sandbox"] == "none"
    monkeypatch.setattr(m, "_seatbelt_state", None)         # restore the probe for the following tests


def test_objective_documents_the_domain_library(episodes):
    for split in ("src", "ood"):
        obj = episodes[split][0].objective
        assert "scilib.codegen" in obj
        for name in ("build_prompts", "sanitize", "repair_prompts", "select_best"):
            assert name in obj
        assert "6 s" in obj                                   # the hidden per-statement limit is documented
