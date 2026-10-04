"""SplitPlan: lineage disjointness, determinism, round-major source stream, D_rep, manifest hashing."""
from __future__ import annotations

import json

import pytest

from scienceclaw.bench.splits import (LineageOverlapError, SplitPlan, discipline_order, load_adapters,
                                      manifest_hash, split_seed)
from scienceclaw.bench.task import Episode
from scienceclaw.bench.tasks.toy import ToyAdapter
from scienceclaw.config import BenchConfig
from scienceclaw.core.schema import PortSchema


class FakeAdapter:
    """Minimal adapter: item ids drawn from a shared pool so that overlap can be provoked."""

    def __init__(self, code: str, family: str = "Life & health", overlap: bool = False) -> None:
        self.discipline, self.family, self.name = code, family, f"fake-{code}"
        self.metric, self.direction, self.task_type = "acc", "max", "classification"
        self.overlap = overlap
        self.calls: list[tuple] = []

    def available(self):
        return True, "fake"

    def build_episodes(self, split, n, seed, items_per_episode=16):
        self.calls.append((split, n, seed, items_per_episode))
        base = 0 if self.overlap else {"src": 0, "val": 1000, "id": 2000, "ood": 3000}[split]
        eps = []
        for k in range(n):
            ids = [f"{self.discipline}:{base + k * items_per_episode + i}" for i in range(items_per_episode)]
            eps.append(Episode(id=f"{self.discipline}-{split}-{k}-{seed % 1000}", discipline=self.discipline,
                               family=self.family, split=split, task_type="classification", objective="o",
                               required_output=PortSchema("array"), tools=[], constraints=[],
                               lineage={"item_ids": ids, "seed": seed}, n_items=items_per_episode))
        return eps

    def pooled_metric(self, per_episode):
        return None


def _cfg(**kw):
    base = dict(disciplines=[], items_per_episode=4, rounds=3, n_val=1, n_id=2, n_ood=2, seed=123)
    base.update(kw)
    return BenchConfig(**base)


def _adapters():
    return {"TOY": ToyAdapter(), "FoR49": FakeAdapter("FoR49", "Engineering & computing"),
            "FoR31": FakeAdapter("FoR31")}


def test_order_and_seeds():
    assert discipline_order({"TOY", "FoR49", "FoR31", "ZZZ"}) == ["FoR31", "FoR49", "TOY", "ZZZ"]
    s = {split_seed(1, d, sp) for d in ("FoR31", "FoR49") for sp in ("src", "val", "id", "ood")}
    assert len(s) == 8
    assert split_seed(1, "FoR31", "src") == split_seed(1, "FoR31", "src")
    ads = _adapters()
    plan = SplitPlan.build(_cfg(), ads)
    assert plan.order == ["FoR31", "FoR49", "TOY"]
    seeds = [c[2] for c in ads["FoR31"].calls]
    assert len(set(seeds)) == 4 and ads["FoR31"].calls[0][:2] == ("src", 3)
    assert [c[1] for c in ads["FoR31"].calls] == [3, 1, 2, 2]


def test_disjointness_verified_and_overlap_raises():
    plan = SplitPlan.build(_cfg(), _adapters())
    for d in plan.order:
        owners = {}
        for split in ("src", "val", "id", "ood"):
            for ep in plan.episodes[split][d]:
                for i in ep.lineage["item_ids"]:
                    assert owners.setdefault(i, split) == split
    bad = {"FoR31": FakeAdapter("FoR31", overlap=True)}
    with pytest.raises(LineageOverlapError):
        SplitPlan.build(_cfg(), bad)


def test_missing_item_ids_and_wrong_split_raise():
    class NoIds(FakeAdapter):
        def build_episodes(self, split, n, seed, items_per_episode=16):
            eps = super().build_episodes(split, n, seed, items_per_episode)
            for e in eps:
                e.lineage = {}
            return eps

    with pytest.raises(ValueError, match="item_ids"):
        SplitPlan.build(_cfg(), {"FoR31": NoIds("FoR31")})

    class WrongSplit(FakeAdapter):
        def build_episodes(self, split, n, seed, items_per_episode=16):
            eps = super().build_episodes(split, n, seed, items_per_episode)
            for e in eps:
                e.split = "id"
            return eps

    with pytest.raises(ValueError, match="split"):
        SplitPlan.build(_cfg(), {"FoR31": WrongSplit("FoR31")})


def test_source_stream_round_major_and_rep():
    plan = SplitPlan.build(_cfg(), _adapters())
    stream = plan.source_stream()
    assert [r for r, _ in stream] == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert [ep.discipline for _, ep in stream[:3]] == ["FoR31", "FoR49", "TOY"]
    assert all(ep.split == "src" for _, ep in stream)
    assert plan.rep_until(0) == []
    rep2 = plan.rep_until(2)
    assert len(rep2) == 6 and all(e.split == "rep" for e in rep2)
    src_by_id = {ep.id: (r, ep) for r, ep in stream}
    for e in rep2:
        r, src = src_by_id[e.id]
        assert e.lineage["rep_of"] == e.id and e.lineage["round"] == r <= 2
        assert e.lineage["item_ids"] == src.lineage["item_ids"]
    assert src_by_id[rep2[0].id][1].split == "src"          # copies do not mutate the source episodes
    assert len(plan.rep_until(99)) == 9 and plan.source_round(stream[4][1].id) == 2


def test_restriction_to_listed_disciplines():
    plan = SplitPlan.build(_cfg(disciplines=["TOY"]), _adapters())
    assert plan.order == ["TOY"]
    with pytest.raises(KeyError):
        SplitPlan.build(_cfg(disciplines=["FoR52"]), _adapters())


def test_manifest_deterministic_hash_and_save(tmp_path):
    m1 = SplitPlan.build(_cfg(), _adapters()).manifest()
    m2 = SplitPlan.build(_cfg(), _adapters()).manifest()
    assert m1 == m2 and m1["sha256"] == manifest_hash(m1) and len(m1["sha256"]) == 64
    assert SplitPlan.build(_cfg(seed=124), _adapters()).manifest()["sha256"] != m1["sha256"]
    assert m1["splits"]["src"]["TOY"][0]["round"] == 1
    assert set(m1["splits"]) == {"src", "val", "id", "ood"}
    assert len(m1["rep"]["episodes"]) == 9
    plan = SplitPlan.build(_cfg(), _adapters())
    p = tmp_path / "splits.json"
    plan.save(p)
    loaded = SplitPlan.load_manifest(p)
    assert loaded == m1
    plan.verify_against(loaded)
    tampered = json.loads(p.read_text())
    tampered["splits"]["id"]["TOY"][0]["id"] = "X"
    p.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="sha256"):
        SplitPlan.load_manifest(p)
    with pytest.raises(ValueError):
        SplitPlan.build(_cfg(seed=5), _adapters()).verify_against(loaded)


def test_load_adapters_toy_and_strict():
    ads = load_adapters(BenchConfig(disciplines=["TOY"]))
    assert list(ads) == ["TOY"]
    with pytest.raises(RuntimeError):
        load_adapters(BenchConfig(disciplines=["NOPE"]))
