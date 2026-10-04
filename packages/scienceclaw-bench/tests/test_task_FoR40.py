"""FoR40 DCASE 2024 Task 2 adapter on the data team's ``reconstructed_v3`` delivery.

Real-data tests (skipped when the delivery is absent or unusable) check the frozen role files, the split sizes, the
empty OOD split under ``SplitPlan``, item disjointness, the visible-data rules (source-domain normal fit support only),
a hand-written reference solution through ``episode.evaluate`` and that ``pooled_metric`` reproduces the data team's
official-evaluator primaries of the full native cohorts. A small synthetic v3-style fixture (test scaffolding only)
covers the fail-closed checks (tampered / missing files, overlapping roles, label contradictions, frozen hashes).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.stats import hmean
from sklearn.metrics import roc_auc_score

from scienceclaw.bench.registry import DATA_ROOT
from scienceclaw.bench.splits import SplitPlan
from scienceclaw.bench.tasks.for40_dcase import (
    DATA_VERSION, DATASET_DIR, EPS, FROZEN_SHA256, ROLE_FILES, STRATA, SUPPORT_FILE, Adapter, DCASEDataError,
    PoolExhausted, load_design, log_mel, official_scores, read_clip,
)
from scienceclaw.config import BenchConfig

# Data team receipts (runtime/dcase_native_v3/README.md, full native cohorts, 3-feature standardized-distance baseline)
NATIVE_BASELINE_PRIMARY = {"src": 0.432146086632, "val": 0.512055022739, "id": 0.484808528273}
ROLE_SIZES = {"src": 210, "val": 210, "id": 140}
HANDLE = re.compile(r"^dcase2024t2/v3/[A-Za-z]+/[0-9a-f]{64}$")


# ------------------------------------------------------------------------------------------------ pure functions
def test_official_scores_formula():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    d = np.array(["source", "source", "target", "target", "source", "target", "source", "target"])
    s = np.array([0.1, 0.4, 0.3, 0.9, 0.8, 0.2, 0.7, 0.95])
    out = official_scores(y, d, s)
    src = (d == "source") | (y != 0)
    tgt = (d == "target") | (y != 0)
    assert out["auc_source"] == pytest.approx(roc_auc_score(y[src], s[src]))
    assert out["auc_target"] == pytest.approx(roc_auc_score(y[tgt], s[tgt]))
    assert out["pauc"] == pytest.approx(roc_auc_score(y, s, max_fpr=0.1))
    assert out["official_score"] == pytest.approx(hmean([out["auc_source"], out["auc_target"], out["pauc"]]))
    perfect = official_scores(y, d, y.astype(float))
    assert perfect["official_score"] == pytest.approx(1.0)
    worst = official_scores(y, d, -y.astype(float))
    assert worst["official_score"] == pytest.approx(hmean([EPS, EPS, worst["pauc"]]))


def test_pooled_diagnostics_keeps_agent_old_reference_and_strong_reference_separate():
    a = Adapter()
    payload = [{"machine": "fan", "y_true": [0, 0, 1, 1],
                "domain": ["source", "target", "source", "target"],
                "y_score": [0.1, 0.2, 0.8, 0.9],
                "y_ref": [0.2, 0.3, 0.7, 0.8],
                "y_strong_ref": [0.15, 0.25, 0.75, 0.85]}]
    d = a.pooled_diagnostics(payload)
    assert d["n_episodes"] == 1 and d["n_items"] == 4 and d["n_machines"] == 1
    assert d["pooled_primary"] == pytest.approx(official_scores(
        np.array(payload[0]["y_true"]), np.array(payload[0]["domain"]), np.array(payload[0]["y_score"]))["official_score"])
    assert d["pooled_reference"] == pytest.approx(official_scores(
        np.array(payload[0]["y_true"]), np.array(payload[0]["domain"]), np.array(payload[0]["y_ref"]))["official_score"])
    assert d["pooled_strong_reference"] == pytest.approx(official_scores(
        np.array(payload[0]["y_true"]), np.array(payload[0]["domain"]), np.array(payload[0]["y_strong_ref"]))["official_score"])


def test_log_mel_shape():
    x = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
    S = log_mel(x, 16000, 1024, 512, 64)
    assert S.shape == (1 + 16000 // 512, 64) and np.isfinite(S).all()
    assert S.mean(axis=0).argmax() < 20                       # 440 Hz lands in a low mel band


def test_real_root_constant():
    assert Adapter().root == Path(DATA_ROOT) / DATASET_DIR


# ------------------------------------------------------------------------------------------------ trivial solution
def _features(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    spec = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(y.size, 1 / 16000)
    return np.array([np.log(np.sqrt(np.mean(y * y)) + 1e-12), np.sum(freq * spec) / (np.sum(spec) + 1e-12) / 8000,
                     np.mean(np.signbit(y[1:]) != np.signbit(y[:-1]))])


def _trivial_scores(train_waves: list, eval_waves: list) -> np.ndarray:
    """Hand-written reference solution: mean squared standardized distance of (log RMS, spectral centroid, ZCR)
    to the normal training clips (the data team's public baseline)."""
    ft = np.array([_features(w) for w in train_waves])
    fe = np.array([_features(w) for w in eval_waves])
    mu, scale = ft.mean(axis=0), np.maximum(ft.std(axis=0), 1e-8)
    return np.mean(((fe - mu) / scale) ** 2, axis=1)


def _solve_with_tools(ep) -> np.ndarray:
    t = {x.name: x for x in ep.tools}
    tr, ev = t["load_train"].fn({}, {}), t["load_eval_inputs"].fn({}, {})
    return _trivial_scores([w[:n] for w, n in zip(tr["waveforms"], tr["lengths"])],
                           [w[:n] for w, n in zip(ev["waveforms"], ev["lengths"])])


# ------------------------------------------------------------------------------------------------ real local delivery
@pytest.fixture(scope="module")
def real():
    a = Adapter()
    ok, why = a.available()
    if not ok:
        pytest.skip(why)
    return a


@pytest.fixture(scope="module")
def plan(real):
    return {s: real.build_episodes(s, n, 20260928 + k, items_per_episode=16)
            for k, (s, n) in enumerate({"src": 7, "val": 7, "id": 7}.items())}


def test_available_and_capacity(real):
    ok, why = real.available()
    assert ok and "capacity" in why and "OOD pool empty" in why
    assert {s: real.capacity(s, 16) for s in ("src", "val", "id", "ood")} == {"src": 7, "val": 7, "id": 7, "ood": 0}
    assert {s: real.capacity(s, 8) for s in ("src", "val", "id")} == {"src": 14, "val": 14, "id": 14}
    assert real.capacity("src", 24) == 0 and real.capacity("id", 20) == 7
    counts = real.stratum_counts()
    assert counts["ood"] == {} and set(counts["src"]) == set(counts["val"]) == set(counts["id"])
    assert all(v == [10, 10, 5, 5] for s in ("src", "val") for v in counts[s].values())
    assert all(v == [5, 5, 5, 5] for v in counts["id"].values())
    assert Adapter(required_items=24).available()[0] is False


def test_design_roles_frozen_and_disjoint(real):
    d = real._design.get()
    assert d.file_sha256 == FROZEN_SHA256
    assert {s: sum(len(c.clips) for c in cohorts.values()) for s, cohorts in d.cohorts.items()} == ROLE_SIZES
    assert all(len(c) == 7 for c in d.cohorts.values())
    assert sum(len(v) for v in d.support.values()) == 168 and {len(v) for v in d.support.values()} == {24}
    assert all(c.domain == "source" and c.label == 0 for v in d.support.values() for c in v)
    probes = [c for cohorts in d.cohorts.values() for co in cohorts.values() for c in co.clips.values()]
    support = [c for v in d.support.values() for c in v]
    assert len(probes) == 560 and len({c.item_id for c in probes + support}) == 728
    assert len({c.pcm_sha256 for c in probes + support}) == 728 and len({c.sha256 for c in probes + support}) == 728
    assert all(HANDLE.match(c.item_id) for c in probes + support)


def test_every_file_matches_its_recorded_hash(real):
    assert real.verify_files() == 728


def test_ood_is_empty_and_splitplan_handles_it(real):
    assert real.build_episodes("ood", 4, 20260928) == []
    assert real.build_episodes("ood", 1, 7, items_per_episode=8) == []
    with pytest.raises(ValueError):
        real.build_episodes("rep", 1, 7)
    cfg = BenchConfig(disciplines=["FoR40"])                                  # paper plan: 7 / 2 / 4 / 4
    sp = SplitPlan.build(cfg, {"FoR40": real})
    assert {s: len(sp.episodes[s]["FoR40"]) for s in ("src", "val", "id", "ood")} == {"src": 7, "val": 2, "id": 4, "ood": 0}
    assert len(sp.episodes["rep"]["FoR40"]) == 7
    assert any("FoR40/ood: adapter returned 0 of 4" in w for w in sp.warnings)
    assert not any("reused" in w for w in sp.warnings)
    assert sp.manifest()["sha256"] == SplitPlan.build(cfg, {"FoR40": real}).manifest()["sha256"]
    other = SplitPlan.build(BenchConfig(disciplines=["FoR40"], seed=3), {"FoR40": real})    # SplitPlan verifies overlap
    assert {s: len(other.episodes[s]["FoR40"]) for s in ("src", "val", "id", "ood")} == {"src": 7, "val": 2, "id": 4, "ood": 0}


def test_episodes_disjoint_stratified_deterministic(real, plan):
    d = real._design.get()
    owner: dict[str, str] = {}
    for s, eps in plan.items():
        assert len(eps) == 7 and {e.lineage["machine"] for e in eps} == set(d.cohorts[s])     # each machine once
        for ep in eps:
            ids = ep.lineage["item_ids"]
            m = ep.lineage["machine"]
            assert ep.n_items == len(ids) == 16 == len(set(ids)) and ep.split == s
            clips = [d.cohorts[s][m].clips[i] for i in ids]
            assert sorted((c.domain, c.label) for c in clips) == sorted(list(STRATA) * 4)
            assert not set(ids) & set(ep.lineage["train_item_ids"])
            assert ep.lineage["train_item_ids"] == [c.item_id for c in d.support[m]] and len(ids) == 16
            assert ep.lineage["ood_kind"] is None and ep.lineage["pool"] == "iid"
            for i in ids:
                assert owner.setdefault(i, s) == s
    assert len(owner) == 3 * 7 * 16
    again = real.build_episodes("src", 3, 20260928, items_per_episode=16)                 # prefix-stable, deterministic
    assert [e.lineage["item_ids"] for e in again] == [e.lineage["item_ids"] for e in plan["src"][:3]]
    assert [e.id for e in again] == [e.id for e in plan["src"][:3]]
    other = real.build_episodes("src", 7, 99, items_per_episode=16)
    assert [e.lineage["item_ids"] for e in other] != [e.lineage["item_ids"] for e in plan["src"]]
    eight = real.build_episodes("val", 14, 5, items_per_episode=8)                        # 2 episodes per cohort
    assert len({i for e in eight for i in e.lineage["item_ids"]}) == 14 * 8
    with pytest.raises(PoolExhausted):
        real.build_episodes("val", 15, 5, items_per_episode=8)
    with pytest.raises(PoolExhausted):
        real.build_episodes("src", 8, 5)
    with pytest.raises(ValueError):
        real.build_episodes("src", 1, 5, items_per_episode=6)


def test_visible_data_rules_and_no_label_leak(real, plan, monkeypatch):
    ep = plan["val"][0]
    t = {x.name: x for x in ep.tools}
    assert set(t) == {"load_train", "load_eval_inputs", "log_mel_spectrogram", "audio_embedding"} and ep._dev_evaluate is None
    tr, ev = t["load_train"].fn({}, {}), t["load_eval_inputs"].fn({}, {})
    assert set(tr) == {"waveforms", "lengths", "domains", "attributes", "sample_rate"}
    assert set(ev) == {"waveforms", "lengths", "sample_rate"} and ev["waveforms"].shape[0] == 16
    assert tr["waveforms"].shape[0] == 24 and set(tr["domains"]) == {"source"} and all(isinstance(a, str) for a in tr["attributes"])
    assert tr["sample_rate"] == ev["sample_rate"] == 16000.0
    # the fit support is not among the evaluation waveforms (PCM-level check)
    fp = {hashlib.sha256(np.round(w[:n] * 32768).astype("<i2").tobytes()).hexdigest()
          for w, n in zip(ev["waveforms"], ev["lengths"])}
    tp = {hashlib.sha256(np.round(w[:n] * 32768).astype("<i2").tobytes()).hexdigest()
          for w, n in zip(tr["waveforms"], tr["lengths"])}
    d = real._design.get()
    assert not fp & tp and fp <= {c.pcm_sha256 for c in d.cohorts["val"][ep.lineage["machine"]].clips.values()}
    S = t["log_mel_spectrogram"].fn({"waveforms": ev["waveforms"][:2]}, {"n_mels": 32, "sample_rate": ev["sample_rate"]})
    assert S["log_mel"].shape[0] == 2 and S["log_mel"].shape[2] == 32
    with pytest.raises(ValueError):
        t["log_mel_spectrogram"].fn({"waveforms": ev["waveforms"][:2]}, {"n_mels": 4})
    # The pretrained route is optional and additive; a deterministic stub checks its
    # ToolSpec contract without requiring GPU weights in the CPU test environment.
    def fake_embed(waveforms, lengths, sample_rate, model, kind):
        assert model == "ast_audioset" and kind == "pooled" and sample_rate == 16000.0
        return np.arange(len(waveforms) * 3, dtype=np.float32).reshape(len(waveforms), 3) + 1

    monkeypatch.setattr("scilib.audioenc.embed", fake_embed)
    E = t["audio_embedding"].fn({"waveforms": ev["waveforms"][:2], "lengths": ev["lengths"][:2],
                                  "sample_rate": ev["sample_rate"]}, {})
    assert E["embeddings"].shape == (2, 3) and np.isfinite(E["embeddings"]).all()
    # nothing the policy can see carries a file name or label
    visible = json.dumps([ep.objective, ep.lineage["item_ids"], ep.lineage["train_item_ids"], [x.description for x in ep.tools]])
    assert not re.search(r"\.wav|_test_|_train_|section_\d\d", visible) and all(HANDLE.match(i) for i in ep.lineage["item_ids"])
    assert "no anomalous and no target-domain training clips" in ep.objective


def test_reference_solution_scores_through_evaluate(real, plan):
    payloads = []
    for ep in (plan["val"][0], plan["val"][1], plan["id"][0]):
        y = _solve_with_tools(ep)
        r = ep.evaluate(y, None)
        assert r.hard_ok() and r.primary is not None and 0.0 < r.primary <= 1.0
        assert set(r.metrics) >= {"auc_source", "auc_target", "pauc", "official_score", "reference_official_score"}
        assert np.isfinite(r.details["norm_score"]) and set(r.details) >= {"reference", "norm_score", "pooled_payload"}
        assert r.accepted == bool(r.primary >= r.details["reference"] + 0.02)
        pay = r.details["pooled_payload"]
        assert pay["machine"] == ep.lineage["machine"] and pay["item_ids"] == ep.lineage["item_ids"]
        assert r.primary == pytest.approx(official_scores(pay["y_true"], pay["domain"], pay["y_score"])["official_score"])
        # the reference itself scores the reference and is not accepted; an oracle is perfect and accepted
        ref = ep.evaluate(np.asarray(pay["y_ref"]), None)
        assert ref.primary == pytest.approx(r.details["reference"]) and not ref.accepted
        oracle = ep.evaluate(np.asarray(pay["y_true"], dtype=float), None)
        assert oracle.primary == pytest.approx(1.0) and oracle.accepted and oracle.hard_ok()
        payloads.append(pay)
    pooled = real.pooled_metric(payloads)
    assert pooled is not None and 0.0 < pooled <= 1.0
    one_machine = [p for p in payloads if p["machine"] == payloads[0]["machine"]]
    assert real.pooled_metric(one_machine) == pytest.approx(official_scores(
        one_machine[0]["y_true"], one_machine[0]["domain"], one_machine[0]["y_score"])["official_score"])
    ep = plan["val"][0]
    assert ep.evaluate(np.zeros(15), None).h["output_shape"] is False
    assert ep.evaluate(np.r_[np.inf, np.zeros(15)], None).h["finite"] is False
    bad = ep.evaluate("nonsense", None)
    assert bad.primary is None and not bad.accepted and real.pooled_metric([bad.details["pooled_payload"]]) is not None
    assert real.pooled_metric([]) is None


def test_pooled_metric_reproduces_native_official_primaries(real):
    """Full native cohorts (all 7 machines, every clip) with the data team's baseline: same primaries as their
    unchanged official evaluator, to 12 digits (also re-verifies every WAV sha256 through read_clip)."""
    d = real._design.get()
    for split, expected in NATIVE_BASELINE_PRIMARY.items():
        payloads = []
        for machine, cohort in sorted(d.cohorts[split].items()):
            ids = sorted(cohort.clips)
            clips = [cohort.clips[i] for i in ids]
            scores = _trivial_scores([read_clip(c) for c in d.support[machine]], [read_clip(c) for c in clips])
            payloads.append({"machine": machine, "y_true": [c.label for c in clips], "domain": [c.domain for c in clips],
                             "y_score": scores.tolist()})
        assert real.pooled_metric(payloads) == pytest.approx(expected, abs=1e-10), split


# ------------------------------------------------------------------------------------------------ synthetic fixture
MACHINES = ("fan", "valve")
PER_STRATUM = 4          # probe clips per stratum and cohort -> 2 episodes of 8 clips per cohort
N_SUPPORT = 5


def _write_wav(path: Path, seed: int, anomalous: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.05, 4000) + (0.3 * np.sin(np.arange(4000) * 0.9) if anomalous else 0.0)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), 16000, pcm)
    return pcm


def _make_delivery(root: Path) -> Path:
    """A miniature v3 delivery below ``root/for40-dcase2024-task2``; returns the v3 directory."""
    base = root / DATASET_DIR
    v3 = base / DATA_VERSION
    (v3 / "roles").mkdir(parents=True)
    counter = iter(range(1, 10_000))

    def record(machine: str, dom: str, label: int, train: bool, j: int, folder: str) -> dict:
        k = next(counter)
        split, cond = ("train" if train else "test"), ("anomaly" if label else "normal")
        name = f"section_00_{dom}_{split}_{cond}_{j:04d}_attr{machine}_{j}.wav"
        p = base / folder / "data" / machine / split / name
        pcm = _write_wav(p, k, bool(label))
        rec = {"bytes": p.stat().st_size, "channels": 1, "domain": dom, "frames": 4000, "id": f"{machine}/{split}/{name}",
               "label": label, "machine": machine, "path": f"/elsewhere/datasets/{DATASET_DIR}/{folder}/data/{machine}/{split}/{name}",
               "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(), "public_id": hashlib.sha256(f"h{k}".encode()).hexdigest(),
               "sample_rate": 16000, "section": "00", "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        if not train:
            rec["condition"] = cond
        return rec

    for role, offset in (("source", 0), ("val", 100), ("id", 200)):
        cohorts = []
        for machine in MACHINES:
            recs = []
            for r, (dom, lab) in enumerate(STRATA):
                for j in range(PER_STRATUM):
                    recs.append(record(machine, dom, lab, False, offset + r * PER_STRATUM + j, "reconstructed_v2"))
            cohorts.append({"episode_id": f"FoR40-v3/{role}/{machine}/section00/both-domains", "machine": machine,
                            "native_unit": "both-domain machine-section cohort", "payload_sha256": "ab" * 32,
                            "probe_records": recs, "section": "00",
                            "stratum_counts": {f"{d}|{y}": PER_STRATUM for d, y in STRATA}})
        (v3 / "roles" / f"{role}.json").write_text(json.dumps(cohorts))
    support = [record(m, "source", 0, True, j, "reconstructed_v1") for m in MACHINES for j in range(N_SUPPORT)]
    (v3 / SUPPORT_FILE).write_text(json.dumps(support))
    return v3


def _adapter(root: Path, **kw) -> Adapter:
    kw.setdefault("required_plan", {"src": 4, "val": 2, "id": 2})
    kw.setdefault("required_items", 8)
    return Adapter(data_root=str(root), verify_frozen=False, **kw)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("dcase_v3")
    _make_delivery(root)
    return root


def _edit(path: Path, fn) -> None:
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data))


def test_synthetic_pipeline_and_support_semantics(synthetic):
    a = _adapter(synthetic)
    ok, why = a.available()
    assert ok, why
    assert a.capacity("src", 8) == 4 and a.capacity("src", 16) == 2 and a.capacity("ood", 8) == 0
    plan = {s: a.build_episodes(s, n, 3, items_per_episode=8) for s, n in {"src": 4, "val": 4, "id": 4, "ood": 4}.items()}
    assert plan["ood"] == [] and all(len(plan[s]) == 4 for s in ("src", "val", "id"))
    owner: dict[str, str] = {}
    for s in ("src", "val", "id"):
        for ep in plan[s]:
            for i in ep.lineage["item_ids"]:
                assert owner.setdefault(i, s) == s
            assert len(ep.lineage["train_item_ids"]) == N_SUPPORT
    assert not {i for ep in plan["src"] for i in ep.lineage["train_item_ids"]} & set(owner)
    ep = plan["val"][0]
    t = {x.name: x for x in ep.tools}
    tr = t["load_train"].fn({}, {})
    assert tr["waveforms"].shape == (N_SUPPORT, 4000) and set(tr["domains"]) == {"source"}
    assert tr["attributes"][0].startswith("attr") and t["load_eval_inputs"].fn({}, {})["waveforms"].shape == (8, 4000)
    pay = ep.evaluate(np.zeros(8), None).details["pooled_payload"]
    assert ep.evaluate(np.asarray(pay["y_true"], float), None).primary == pytest.approx(1.0)
    assert ep.lineage["item_ids"] == pay["item_ids"] and ep.lineage["ood_kind"] is None
    # two episodes of one machine pool their items in pooled_metric
    same = [e for e in plan["src"] if e.lineage["machine"] == plan["src"][0].lineage["machine"]]
    assert len(same) == 2
    pays = [e.evaluate(np.asarray(e.evaluate(np.zeros(8), None).details["pooled_payload"]["y_true"], float), None)
            .details["pooled_payload"] for e in same]
    assert a.pooled_metric(pays) == pytest.approx(1.0)
    again = _adapter(synthetic).build_episodes("src", 4, 3, items_per_episode=8)
    assert [e.lineage["item_ids"] for e in again] == [e.lineage["item_ids"] for e in plan["src"]]


def test_synthetic_frozen_hashes_and_small_plan(synthetic):
    ok, why = Adapter(data_root=str(synthetic), required_plan={"src": 4, "val": 2, "id": 2}, required_items=8).available()
    assert not ok and "frozen" in why                                     # role files differ from the pinned sha256
    ok, why = _adapter(synthetic, required_items=16, required_plan={"src": 3}).available()
    assert not ok and "src: 2 of 3 episodes" in why
    ok, why = _adapter(synthetic, required_plan={"src": 1, "ood": 1}).available()
    assert not ok and "ood: 0 of 1" in why
    assert not Adapter(data_root=str(synthetic / "nowhere")).available()[0]


def test_synthetic_tampered_wav_fails_closed(tmp_path):
    _make_delivery(tmp_path)
    a = _adapter(tmp_path)
    victim = sorted((tmp_path / DATASET_DIR / "reconstructed_v2" / "data").rglob("*_test_*.wav"))[0]
    clip = next(c for c in a._design.get().clips() if c.path == victim)
    assert read_clip(clip).size == 4000 and a.verify_files() == 2 * 3 * 4 * PER_STRATUM + len(MACHINES) * N_SUPPORT
    wavfile.write(str(victim), 16000, np.zeros(4000, dtype=np.int16))          # same size, different content
    assert a.available()[0]                                                    # size check alone does not notice
    with pytest.raises(DCASEDataError, match="sha256"):
        read_clip(clip)
    with pytest.raises(DCASEDataError, match="sha256"):
        a.verify_files()


def test_synthetic_missing_and_wrong_size_files(tmp_path):
    _make_delivery(tmp_path)
    victim = sorted((tmp_path / DATASET_DIR / "reconstructed_v1" / "data").rglob("*.wav"))[0]
    victim.unlink()
    ok, why = _adapter(tmp_path).available()
    assert not ok and "missing or have the wrong size" in why


@pytest.mark.parametrize("what", ["dup_handle", "cross_role_pcm", "label_contradiction", "target_support", "bad_counts",
                                  "unknown_support_machine", "path_escape"])
def test_synthetic_invalid_designs_are_rejected(tmp_path, what):
    v3 = _make_delivery(tmp_path)
    src, val = v3 / "roles" / "source.json", v3 / "roles" / "val.json"
    if what == "dup_handle":                       # a val clip reuses a source handle
        first = json.loads(src.read_text())[0]["probe_records"][0]["public_id"]
        _edit(val, lambda d: d[0]["probe_records"][0].__setitem__("public_id", first))
        needle = "not unique"
    elif what == "cross_role_pcm":                 # same audio in two roles
        pcm = json.loads(src.read_text())[0]["probe_records"][0]["pcm_sha256"]
        _edit(val, lambda d: d[0]["probe_records"][0].__setitem__("pcm_sha256", pcm))
        needle = "not unique"
    elif what == "label_contradiction":
        _edit(src, lambda d: d[0]["probe_records"][0].__setitem__("label", 1 - d[0]["probe_records"][0]["label"]))
        needle = "contradict"
    elif what == "target_support":
        _edit(v3 / SUPPORT_FILE, lambda d: d[0].__setitem__("domain", "target"))
        needle = "contradict"
    elif what == "bad_counts":
        _edit(src, lambda d: d[0]["stratum_counts"].__setitem__("source|0", 99))
        needle = "stratum counts"
    elif what == "unknown_support_machine":
        _edit(v3 / SUPPORT_FILE, lambda d: [r.__setitem__("machine", "gearbox") for r in d if r["machine"] == "valve"])
        needle = "no source_fit_support"
    else:
        _edit(src, lambda d: d[0]["probe_records"][0].__setitem__("path", "/x/" + DATASET_DIR + "/../../etc/passwd"))
        needle = "suspicious path"
    with pytest.raises(DCASEDataError, match=needle):
        load_design(tmp_path / DATASET_DIR, verify_frozen=False)
    ok, why = _adapter(tmp_path).available()
    assert not ok and "unusable" in why


def test_role_file_constants_cover_all_roles():
    assert set(ROLE_FILES) == {"src", "val", "id"} and set(FROZEN_SHA256) == set(ROLE_FILES.values()) | {SUPPORT_FILE}


# ------------------------------------------------------------------------------------------------ domain library (scilib.anomsound)
def test_objective_documents_the_domain_library(plan):
    from test_adapter_visible_text import scan_acceptance, scan_recipe, visible_text

    ep = plan["val"][0]
    for name in ("scilib.anomsound", "score_clips", "fit_predict", "component_scores", "rank_average", "log_mel_list",
                 "probe_report"):
        assert name in ep.objective
    text = visible_text(ep)
    assert not scan_recipe(text) and not scan_acceptance(text)
    low = ep.objective.lower()
    assert "reference" not in low and "accepted" not in low and "baseline" not in low
    # nothing about the composition of the evaluation clips (how many are anomalous / target-domain) is stated
    assert not re.search(r"\b(\d+|four|eight|half|most|all)\s+(of the \d+\s+)?(clips\s+)?(are\s+)?(anomal|target|source)", low)


def test_domain_library_round_trip_through_the_tools(real, plan):
    """The library consumes the tools' arrays as they are (zero-padded rows + lengths) and returns a valid answer;
    its nearest-training-clip component reproduces the adapter's own log-mel statistics (same front end)."""
    from scilib import anomsound as an

    for ep in (plan["src"][0], plan["id"][1]):
        t = {x.name: x for x in ep.tools}
        tr, ev = t["load_train"].fn({}, {}), t["load_eval_inputs"].fn({}, {})
        y = an.score_clips(tr, ev)
        assert y.shape == (16,) and np.isfinite(y).all() and y.min() > 0 and y.max() <= 1
        assert np.array_equal(y, an.score_clips(tr, ev))
        res = ep.evaluate(y, None)
        assert res.hard_ok() and res.primary is not None and 0.0 < res.primary <= 1.0
        tm = an.log_mel_list(tr["waveforms"], tr["lengths"], tr["sample_rate"])
        em = an.log_mel_list(ev["waveforms"], ev["lengths"], ev["sample_rate"])
        comps = an.component_scores(tm, em)
        assert set(comps) == set(an.COMPONENTS)
        ref = ep.evaluate(comps["nn_train"], None)
        assert ref.primary == pytest.approx(res.details["reference"], abs=1e-6)
        # the tool's own (padded) log-mel output, trimmed through the library, gives the same descriptors
        lm = t["log_mel_spectrogram"].fn({"waveforms": ev["waveforms"]}, {"sample_rate": ev["sample_rate"]})["log_mel"]
        trimmed = an.trim_mels(lm, ev["lengths"])
        assert np.allclose(an.clip_descriptors(trimmed, "ms"), an.clip_descriptors(em, "ms"), atol=1e-3)
