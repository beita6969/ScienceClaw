"""FoR44 ACIC 2016 adapter.

Real data: the data team's ``reconstructed_v1`` delivery (192 official-generator whole simulations, roles source /
validation / evaluation of 64 each, setting-disjoint). The adapter maps source -> src (+ dev), validation -> val,
evaluation -> id; ood is empty unless ``ood_mode="step_assignment"``. Real-data tests skip when the delivery is missing.
Layout handling, role assignment and the OOD proxy are also exercised on tiny synthetic fixtures (test scaffolding only,
never benchmark data).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scienceclaw.bench.tasks.for44_acic import (
    DATASET_DIR, IID_SCENARIOS, OOD_SCENARIOS, PARAMETERS_2016, ROLES, Adapter, design_matrix, discover, ols_effect,
    read_sim, role_of, scenario_of,
)

N_UNITS = 120


def test_parameter_table_and_scenario_split():
    assert len(PARAMETERS_2016) == 77 and len(OOD_SCENARIOS) == 32 and len(IID_SCENARIOS) == 45
    assert PARAMETERS_2016[1] == {"model.trt": "linear", "root.trt": 0.35, "overlap.trt": "one-term",
                                  "model.rsp": "linear", "alignment": 0.75, "te.hetero": "high"}
    assert PARAMETERS_2016[77]["model.trt"] == "step" and 9 in OOD_SCENARIOS and 17 in OOD_SCENARIOS


def _settings(items) -> set[int]:
    return {scenario_of(i) for i in items}


def _assert_disjoint_pools(parts: dict[str, list[str]]) -> None:
    """Item-disjoint over all pools; setting-disjoint between src/val/id/ood and between dev and val/id/ood."""
    names = ("dev", "src", "val", "id", "ood")
    for i, u in enumerate(names):
        for v in names[i + 1:]:
            assert not set(parts[u]) & set(parts[v]), (u, v)
    for u, v in (("src", "val"), ("src", "id"), ("val", "id"), ("src", "ood"), ("val", "ood"), ("id", "ood"),
                 ("dev", "val"), ("dev", "id"), ("dev", "ood")):
        assert not _settings(parts[u]) & _settings(parts[v]), (u, v)


# ------------------------------------------------------------------------------------------------ real delivery
@pytest.fixture(scope="module")
def real():
    a = Adapter()
    files = discover(a.root)
    if not any(role_of(f) for f in files.values()):
        pytest.skip(f"no data-team role folders with potential outcomes under {a.root}")
    return a


def _role_items(a: Adapter) -> dict[str, list[str]]:
    files = a._files.get()
    return {r: sorted(i for i, f in files.items() if role_of(f) == r) for r in ROLES}


def test_real_available_and_pools_follow_delivered_roles(real):
    ok, why = real.available()
    assert ok, why
    roles, parts = _role_items(real), real._parts.get()
    assert len(parts["dev"]) == real.n_dev
    assert set(parts["dev"]) | set(parts["src"]) == set(roles["source"])          # source -> src (+ dev)
    assert set(parts["val"]) == set(roles["validation"])                          # validation -> val
    assert set(parts["id"]) == set(roles["evaluation"])                           # evaluation -> id
    assert parts["ood"] == []                                                     # the delivery defines no OOD role
    _assert_disjoint_pools(parts)                                                 # roles are setting-disjoint
    assert real.build_episodes("ood", 4, 1) == []
    assert "ood empty" in why and "'ood': 0" in why


def test_real_reference_and_truth_match_data_team_R_outputs(real):
    csv = real.root / "reconstructed_v1" / "baseline-per-simulation.csv"
    if not csv.exists():
        pytest.skip("data-team baseline table missing")
    b = pd.read_csv(csv).set_index("instance_id")
    X = real._X.get()
    for iid in sorted(real._files.get())[:6]:
        sim = real._sim(iid)
        assert ols_effect(X, sim.z, sim.y) == pytest.approx(b.loc[iid, "estimated_ate"], abs=1e-8)   # R lm(y ~ .)
        assert float(np.mean(sim.mu1 - sim.mu0)) == pytest.approx(b.loc[iid, "true_ate"], abs=1e-10)
        assert sim.sd_y == pytest.approx(b.loc[iid, "response_sd"], abs=1e-10)


def test_real_episodes_lineage_and_trivial_solutions(real):
    eps = {s: real.build_episodes(s, 2, 5, items_per_episode=4) for s in ("src", "val", "id")}
    dev_ids = set(real._parts.get()["dev"])
    owner: dict[str, str] = {}
    setting_owner: dict[int, str] = {}
    for s, es in eps.items():
        for ep in es:
            assert ep.split == s and ep.lineage["pool"] == "iid" and ep.lineage["ood_kind"] is None
            assert ep.lineage["data_team_role"] == {"src": "source", "val": "validation", "id": "evaluation"}[s]
            assert set(ep.lineage["dev_item_ids"]) == dev_ids and not dev_ids & set(ep.lineage["item_ids"])
            for i in ep.lineage["item_ids"]:
                assert owner.setdefault(i, s) == s                                # item-disjoint across splits
                assert setting_owner.setdefault(scenario_of(i), s) == s           # setting-disjoint across splits
    again = real.build_episodes("id", 2, 5, items_per_episode=4)
    assert [e.lineage["item_ids"] for e in again] == [e.lineage["item_ids"] for e in eps["id"]]

    ep = eps["id"][0]
    t = {x.name: x for x in ep.tools}
    assert t["load_covariates"].fn({}, {})["covariates"].shape == (4802, 58)
    ev = t["load_eval_inputs"].fn({}, {})
    Z, Y = ev["treatment"], ev["outcome"]
    assert set(ev) == {"treatment", "outcome"} and Z.shape == (4, 4802) and Y.shape == (4, 4802)
    assert t["load_dev_inputs"].fn({}, {})["dev_treatment"].shape == (real.n_dev, 4802)

    # trivial reference solution 1: OLS y ~ z + X (the adapter's reference) computed only from visible data
    X = design_matrix(t["load_covariates"].fn({}, {})["covariates"])
    ols = np.array([ols_effect(X, Z[j], Y[j]) for j in range(4)])
    r_ols = ep.evaluate(ols, None)
    assert r_ols.primary == pytest.approx(r_ols.details["reference"]) and r_ols.details["norm_score"] == pytest.approx(1.0)
    assert not r_ols.accepted and 0.0 < r_ols.primary < 1.0
    # trivial solution 2: naive difference of means (confounded) -> finite, positive, normalized score in (0, 10]
    dm = np.array([Y[j][Z[j] == 1].mean() - Y[j][Z[j] == 0].mean() for j in range(4)])
    r_dm = ep.evaluate(dm, None)
    assert np.isfinite(r_dm.primary) and 0.0 < r_dm.details["norm_score"] <= 10.0 and r_dm.h["effect_scale"]
    # all-zero estimates are worse than OLS; the hidden truth scores 0 (clipped normalized score 10) and is accepted
    r_zero = ep.evaluate(np.zeros(4), None)
    assert r_zero.primary > r_ols.primary and r_zero.details["norm_score"] < 1.0
    pp = r_zero.details["pooled_payload"]
    assert pp["tau_ref"] == pytest.approx(ols.tolist(), abs=1e-8)
    oracle = ep.evaluate(np.asarray(pp["satt"]), None)
    assert oracle.primary == pytest.approx(0.0) and oracle.details["norm_score"] == 10.0 and oracle.accepted
    assert real.pooled_metric([r_ols.details["pooled_payload"], r_dm.details["pooled_payload"]]) > 0.0
    assert ep.evaluate(np.zeros(3), None).h["output_shape"] is False


def test_real_step_assignment_proxy_ood(real):
    a = Adapter(ood_mode="step_assignment")
    ok, why = a.available()
    assert ok, why
    roles, parts = _role_items(a), a._parts.get()
    step_all = {i for v in roles.values() for i in v if scenario_of(i) in OOD_SCENARIOS}
    assert step_all and set(parts["ood"]) == step_all                            # every step sim of every role
    for s in ("dev", "src", "val", "id"):
        assert not _settings(parts[s]) & set(OOD_SCENARIOS)
    assert set(parts["val"]) == set(roles["validation"]) - step_all
    assert set(parts["id"]) == set(roles["evaluation"]) - step_all
    _assert_disjoint_pools(parts)
    eps = a.build_episodes("ood", 2, 5, items_per_episode=4)
    assert len(eps) == 2
    for ep in eps:
        assert ep.lineage["pool"] == "ood" and ep.lineage["ood_kind"] == "proxy_within_dataset"
        assert ep.lineage["ood_mode"] == "step_assignment" and set(ep.lineage["item_ids"]) <= step_all
    assert Adapter(ood_mode="none").build_episodes("ood", 2, 5, items_per_episode=4) == []
    with pytest.raises(ValueError):
        Adapter(ood_mode="bogus")


# ------------------------------------------------------------------------------------------------ synthetic fixtures
def _draw(x: pd.DataFrame, rng: np.random.Generator, shift: float):
    x1, x3 = x["x_1"].to_numpy(float), x["x_3"].to_numpy(float)
    z = (rng.random(len(x)) < 1 / (1 + np.exp(-(x1 + shift)))).astype(int)
    mu0 = 1.0 + x1 + 0.5 * (x["x_2"] == "B")
    mu1 = mu0 + 2.0 + 0.5 * x3
    y0, y1 = mu0 + rng.normal(0, 1, len(x)), mu1 + rng.normal(0, 1, len(x))
    return z, y0, y1, mu0.to_numpy(), mu1.to_numpy()


def _write_sim(path: Path, x: pd.DataFrame, rng: np.random.Generator, shift: float, dgp_layout: bool = False) -> None:
    z, y0, y1, mu0, mu1 = _draw(x, rng, shift)
    path.parent.mkdir(parents=True, exist_ok=True)
    if dgp_layout:
        pd.DataFrame({"z": z, "y": np.where(z == 1, y1, y0), "y.0": y0, "y.1": y1, "mu.0": mu0, "mu.1": mu1}).to_csv(path, index=False)
    else:
        pd.DataFrame({"z": z, "y0": y0, "y1": y1, "mu0": mu0, "mu1": mu1}).to_csv(path, index=False)


def _write_team_sim(d: Path, x: pd.DataFrame, rng: np.random.Generator, shift: float) -> None:
    """Data-team layout: observed.csv.gz (subject_id, z, y) + oracle.csv.gz (y0, y1, mu0, mu1, tau)."""
    z, y0, y1, mu0, mu1 = _draw(x, rng, shift)
    d.mkdir(parents=True)
    sid = np.arange(1, len(x) + 1)
    pd.DataFrame({"subject_id": sid, "z": z, "y": np.where(z == 1, y1, y0)}).to_csv(d / "observed.csv.gz", index=False)
    pd.DataFrame({"subject_id": sid, "y0": y0, "y1": y1, "mu0": mu0, "mu1": mu1, "tau": mu1 - mu0}).to_csv(
        d / "oracle.csv.gz", index=False)


def _covariates(rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({"x_1": rng.normal(size=N_UNITS), "x_2": rng.choice(["A", "B", "C"], N_UNITS),
                         "x_3": rng.normal(size=N_UNITS), "x_4": rng.integers(0, 5, N_UNITS)})


def _team_root(base: Path, roles: dict[str, tuple[int, ...]], reps: int, seed: int = 0) -> Path:
    """<base>/for44-acic-2016/reconstructed_v1/<role>/acic2016-pPP-rRRR/ + covariates.csv.gz."""
    rng = np.random.default_rng(seed)
    top = base / DATASET_DIR / "reconstructed_v1"
    top.mkdir(parents=True)
    x = _covariates(rng)
    x.to_csv(top / "covariates.csv.gz", index=False)
    for k, (role, ps) in enumerate(roles.items()):
        for p in ps:
            for r in range(1, reps + 1):                       # distinct replicate numbers per role: ids never collide
                _write_team_sim(top / role / f"acic2016-p{p:02d}-r{k * reps + r:03d}", x, rng, 0.0)
    return base


TEAM_ROLES = {"source": (1, 2, 3, 9, 17), "validation": (4, 5, 10, 48), "evaluation": (6, 7, 11, 49)}   # 9/17/48/49: step


@pytest.fixture(scope="module")
def team_root(tmp_path_factory):
    return _team_root(tmp_path_factory.mktemp("acic_team"), TEAM_ROLES, reps=8)


def test_team_layout_roles_map_to_splits_and_ood_is_empty(team_root):
    a = Adapter(data_root=str(team_root), n_dev=4)
    ok, why = a.available()
    assert ok, why
    parts = a._parts.get()
    assert {k: len(v) for k, v in parts.items()} == {"dev": 4, "src": 36, "val": 32, "id": 32, "ood": 0}
    assert _settings(parts["dev"]) | _settings(parts["src"]) == set(TEAM_ROLES["source"])
    assert _settings(parts["val"]) == set(TEAM_ROLES["validation"])
    assert _settings(parts["id"]) == set(TEAM_ROLES["evaluation"])
    _assert_disjoint_pools(parts)
    eps = {s: a.build_episodes(s, 2, 7, items_per_episode=4) for s in ("src", "val", "id")}
    for s, es in eps.items():
        assert len(es) == 2
        for ep in es:
            assert set(ep.lineage["item_ids"]) <= set(parts[s]) and ep.n_items == 4
    assert a.build_episodes("ood", 3, 7, items_per_episode=4) == []
    again = Adapter(data_root=str(team_root), n_dev=4).build_episodes("val", 2, 7, items_per_episode=4)
    assert [e.lineage["item_ids"] for e in again] == [e.lineage["item_ids"] for e in eps["val"]]
    with pytest.raises(ValueError, match="supports only"):
        a.build_episodes("id", 9, 7, items_per_episode=4)                            # 32 sims / 4 = 8 episodes max


def test_team_layout_step_assignment_ood_overrides_roles(team_root):
    a = Adapter(data_root=str(team_root), n_dev=4, ood_mode="step_assignment")
    ok, why = a.available()
    assert ok, why
    parts = a._parts.get()
    assert {k: len(v) for k, v in parts.items()} == {"dev": 4, "src": 20, "val": 24, "id": 24, "ood": 32}
    assert _settings(parts["ood"]) == {9, 17, 48, 49}
    assert not (_settings(parts["dev"]) | _settings(parts["src"]) | _settings(parts["val"]) | _settings(parts["id"])) & {9, 17, 48, 49}
    _assert_disjoint_pools(parts)
    ep = a.build_episodes("ood", 1, 3, items_per_episode=4)[0]
    assert ep.lineage["ood_kind"] == "proxy_within_dataset" and ep.lineage["data_team_role"] is None
    t = {x.name: x for x in ep.tools}
    assert t["load_eval_inputs"].fn({}, {})["treatment"].shape == (4, N_UNITS)


def test_team_layout_minimum_plan_and_role_contract(tmp_path, team_root):
    ok, why = Adapter(data_root=str(team_root), n_dev=10**6).available()
    assert ok is False and "too few" in why and "dev" in why
    ok, why = Adapter(data_root=str(team_root), n_dev=4, ood_mode="step_assignment").available()
    assert ok
    # a setting occurring in two roles breaks the lineage contract -> not available (no silent leakage)
    bad = _team_root(tmp_path, {"source": (1, 2), "validation": (3,), "evaluation": (2,)}, reps=2)
    ok, why = Adapter(data_root=str(bad), n_dev=1).available()
    assert ok is False and "setting-disjoint" in why


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory):
    """Loose layouts without role folders: official release (p 1, 3, 5, 10) + dgp exports (p 9, 17 = step)."""
    root = tmp_path_factory.mktemp("acic_root")
    base = root / DATASET_DIR
    rng = np.random.default_rng(0)
    x = _covariates(rng)
    (base / "data_cf_all").mkdir(parents=True)
    x.to_csv(base / "data_cf_all" / "x.csv", index=False)
    for p in (1, 3, 5, 10):
        for s in range(1, 31):
            _write_sim(base / "data_cf_all" / str(p) / f"zymu_{s}.csv", x, rng, 0.0)
    for p in (9, 17):
        for s in range(1, 9):
            _write_sim(base / "dgp_exports" / f"p{p}_s{s}.csv", x, rng, 1.0, dgp_layout=True)
    return root


def test_fixture_discovery_layouts_and_reader(fixture_root):
    files = discover(fixture_root / DATASET_DIR)
    assert len(files) == 4 * 30 + 2 * 8
    assert all(role_of(f) is None for f in files.values())
    sim = read_sim(files["acic2016-p09-r001"])
    assert sim.z.shape == (N_UNITS,) and np.isfinite(sim.satt) and sim.sd_y > 0
    sim2 = read_sim(files["acic2016-p01-r001"])                    # y reconstructed from y0/y1 and z
    assert np.isfinite(sim2.y).all()


def test_fixture_loose_layout_assigns_whole_settings_to_roles(fixture_root):
    a = Adapter(data_root=str(fixture_root), n_dev=4)
    parts = a._parts.get()
    assert sum(len(v) for v in parts.values()) == 4 * 30 + 2 * 8 and parts["ood"] == []
    _assert_disjoint_pools(parts)                                   # settings hashed to roles, never split
    again = Adapter(data_root=str(fixture_root), n_dev=4)._parts.get()
    assert again == parts
    ok, why = Adapter(data_root=str(fixture_root), n_dev=10**6).available()
    assert ok is False and "too few" in why


def test_fixture_episodes_tools_evaluator(fixture_root):
    a = Adapter(data_root=str(fixture_root), n_dev=4, ood_mode="step_assignment")
    ok, why = a.available()
    parts = a._parts.get()
    if not ok:                                                     # a role with < 1 full episode of 16 in this hash draw
        assert "too few" in why
    assert set(parts["ood"]) == {i for i in a._files.get() if scenario_of(i) in (9, 17)}
    _assert_disjoint_pools(parts)
    eps = {s: a.build_episodes(s, 1, 7, items_per_episode=4) for s in ("src", "val", "id", "ood")}
    owner = {}
    for s, es in eps.items():
        for ep in es:
            for i in ep.lineage["item_ids"]:
                assert owner.setdefault(i, s) == s
            assert not set(ep.lineage["dev_item_ids"]) & set(ep.lineage["item_ids"])
    again = Adapter(data_root=str(fixture_root), n_dev=4, ood_mode="step_assignment").build_episodes(
        "src", 1, 7, items_per_episode=4)
    assert [e.lineage["item_ids"] for e in again] == [e.lineage["item_ids"] for e in eps["src"]]
    ep = eps["ood"][0]
    t = {x.name: x for x in ep.tools}
    ev = t["load_eval_inputs"].fn({}, {})
    assert set(ev) == {"treatment", "outcome"} and ev["treatment"].shape == (4, N_UNITS)
    assert t["load_covariates"].fn({}, {})["covariates"].shape == (N_UNITS, 4)
    dev = t["score_dev"].fn({"dev_estimates": np.full(4, 2.0)}, {})
    assert dev["dev_normalized_rmse"] >= 0 and dev["dev_reference_normalized_rmse"] >= 0
    pp = ep.evaluate(np.zeros(4), None).details["pooled_payload"]
    ref = ep.evaluate(np.asarray(pp["tau_ref"]), None)
    assert ref.primary == pytest.approx(ref.details["reference"]) and not ref.accepted
    oracle = ep.evaluate(np.asarray(pp["satt"]), None)
    assert oracle.primary == pytest.approx(0.0) and oracle.accepted and oracle.z == 1
    assert a.pooled_metric([oracle.details["pooled_payload"]]) == pytest.approx(0.0)
    huge = ep.evaluate(np.asarray(pp["satt"]) + 100.0, None)
    assert huge.h["effect_scale"] is False and huge.z == 0
    assert ep.evaluate(np.zeros(3), None).h["output_shape"] is False


def test_objective_documents_the_domain_library(fixture_root):
    a = Adapter(data_root=str(fixture_root), n_dev=4, ood_mode="step_assignment")
    ep = a.build_episodes("src", 1, 7, items_per_episode=4)[0]
    assert "scilib.causal" in ep.objective and "estimate_effects" in ep.objective


def test_ols_effect_recovers_linear_effect():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 3))
    z = (rng.random(500) < 0.4).astype(int)
    y = 1.5 * z + X @ np.array([1.0, -2.0, 0.5]) + rng.normal(0, 0.1, 500)
    assert ols_effect(X, z, y) == pytest.approx(1.5, abs=0.05)
