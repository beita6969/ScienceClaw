"""Lineage-disjoint benchmark splits D_src / D_val / D_ID / D_OOD / D_rep and frozen, hashed manifests (Sec. 4).

Protocol (DESIGN §7):

* Disciplines are processed in a fixed order: registry order (FoR30 … FoR52), then ``TOY``, then any other
  adapter codes sorted alphabetically.
* Per discipline the adapter builds ``rounds`` source episodes, ``n_val`` validation, ``n_id`` held-out ID and
  ``n_ood`` held-out OOD episodes. Each (discipline, split) gets its own deterministic seed
  :func:`split_seed` = first 32 bits of sha256(f"{seed}|{discipline}|{split}").
* Adapters guarantee lineage disjointness; the plan *verifies* it: the evaluation item ids
  (``episode.lineage["item_ids"]``) of different splits of the same discipline must not intersect, otherwise
  :class:`LineageOverlapError` is raised. Duplicate item ids across episodes of the *same* split are recorded
  as manifest warnings. Episode ids must be unique across the whole plan.
* Rounds are 1-indexed: round r processes source episode index r-1 of every discipline; snapshot A_r is the
  program after round r (A_0 = initial program). :meth:`SplitPlan.source_stream` yields ``(r, episode)``
  round-major in the fixed discipline order.
* D_rep: frozen copies of source episodes (same items, ``split="rep"``, same episode id,
  ``lineage["rep_of"]`` = source id, ``lineage["round"]`` = its source round). ``rep_until(r)`` returns the
  copies of all source episodes of rounds <= r (those already seen by A_r).
* :meth:`SplitPlan.manifest` lists episode ids + lineage per split and discipline (plus seeds and adapter
  metadata) and its sha256 over the canonical JSON (sorted keys, compact separators) of everything except the
  ``sha256`` field itself.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .registry import BY_CODE, DISCIPLINES
from .task import Episode, TaskAdapter

BUILT_SPLITS = ("src", "val", "id", "ood")
MANIFEST_SCHEMA = "scienceclaw.splits/v1"
TOY_CODE = "TOY"


class LineageOverlapError(ValueError):
    """Raised when two splits of one discipline share evaluation items (lineage leak)."""


def split_seed(seed: int, discipline: str, split: str) -> int:
    """Deterministic, distinct seed per (global seed, discipline, split)."""
    return int(hashlib.sha256(f"{seed}|{discipline}|{split}".encode()).hexdigest()[:8], 16)


def discipline_order(codes: list[str] | set[str] | dict) -> list[str]:
    """Registry order (FoR30..FoR52), then TOY, then remaining codes sorted."""
    codes = set(codes)
    reg = [d.code for d in DISCIPLINES if d.code in codes]
    rest = sorted(codes - set(reg) - {TOY_CODE})
    return reg + ([TOY_CODE] if TOY_CODE in codes else []) + rest


def _json_default(o: Any) -> Any:
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=str)
    if isinstance(o, tuple):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default, ensure_ascii=False)


def manifest_hash(manifest: dict) -> str:
    """sha256 of the canonical JSON of the manifest without its ``sha256`` field."""
    body = {k: v for k, v in manifest.items() if k != "sha256"}
    return hashlib.sha256(canonical_json(body).encode()).hexdigest()


def _item_ids(ep: Episode) -> list[str]:
    ids = (ep.lineage or {}).get("item_ids")
    if ids is None:
        raise ValueError(f"episode {ep.id} ({ep.discipline}/{ep.split}) has no lineage['item_ids']; adapters must "
                         "record the evaluation item ids so that split disjointness can be verified")
    return [str(i) for i in ids]


def _build_capped(ad, code: str, split: str, n: int, seed: int, ipe: int, warnings: list[str]) -> list[Episode]:
    """``ad.build_episodes`` for ``n`` episodes, backing off to the largest count the data pool supports.

    Adapters raise a ``PoolExhausted`` (ValueError) when the delivered data cannot supply ``n`` item-disjoint episodes
    (e.g. FoR44 supports 3 source episodes of 16 items). Draws are prefix-stable, so the first k episodes equal those of
    the full request; the shortfall is recorded in ``warnings`` and the round schedule simply skips the missing rounds.
    """
    for k in range(n, 0, -1):
        try:
            eps = list(ad.build_episodes(split, k, seed, items_per_episode=ipe))
        except ValueError as ex:
            if type(ex).__name__ != "PoolExhausted":
                raise
            last = str(ex)
            continue
        if k < n:
            warnings.append(f"{code}/{split}: data pool supports only {k} of {n} requested episodes ({last[:160]})")
        return eps
    if n > 0:
        warnings.append(f"{code}/{split}: data pool supports no episode ({last[:160]})")
    return []


@dataclass
class SplitPlan:
    episodes: dict[str, dict[str, list[Episode]]]          # split -> discipline -> episodes
    order: list[str]                                       # fixed discipline order
    seed: int = 0
    items_per_episode: int = 16
    rounds: int = 0
    counts: dict[str, int] = field(default_factory=dict)   # requested n per split
    seeds: dict[str, dict[str, int]] = field(default_factory=dict)   # discipline -> split -> seed
    adapters_info: dict[str, dict] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, bench_cfg: Any, adapters: dict[str, TaskAdapter]) -> "SplitPlan":
        """Build all splits for ``adapters`` (restricted to ``bench_cfg.disciplines`` when non-empty)."""
        wanted = list(getattr(bench_cfg, "disciplines", None) or [])
        if wanted:
            missing = [c for c in wanted if c not in adapters]
            if missing:
                raise KeyError(f"bench.disciplines lists {missing} but no adapter was provided for them")
            pool = {c: adapters[c] for c in wanted}
        else:
            pool = dict(adapters)
        if not pool:
            raise ValueError("SplitPlan.build: no adapters")
        order = discipline_order(pool)
        seed = int(bench_cfg.seed)
        ipe = int(bench_cfg.items_per_episode)
        counts = {"src": int(bench_cfg.rounds), "val": int(bench_cfg.n_val), "id": int(bench_cfg.n_id),
                  "ood": int(bench_cfg.n_ood)}
        episodes: dict[str, dict[str, list[Episode]]] = {s: {} for s in (*BUILT_SPLITS, "rep")}
        seeds: dict[str, dict[str, int]] = {}
        info: dict[str, dict] = {}
        warnings: list[str] = []
        seen_ids: dict[str, str] = {}
        for code in order:
            ad = pool[code]
            info[code] = {k: getattr(ad, k, None) for k in ("name", "family", "metric", "direction", "task_type")}
            seeds[code] = {}
            owner: dict[str, str] = {}                       # item id -> split (this discipline)
            for split in BUILT_SPLITS:
                n = counts[split]
                s = split_seed(seed, code, split)
                seeds[code][split] = s
                eps = _build_capped(ad, code, split, n, s, ipe, warnings)
                reused: list[str] = []
                if len(eps) < n:
                    warnings.append(f"{code}/{split}: adapter returned {len(eps)} of {n} requested episodes")
                for ep in eps:
                    if ep.split != split:
                        raise ValueError(f"adapter {code} returned episode {ep.id} with split {ep.split!r} for {split!r}")
                    if ep.discipline != code:
                        raise ValueError(f"adapter {code} returned episode {ep.id} of discipline {ep.discipline!r}")
                    if ep.id in seen_ids:
                        raise ValueError(f"duplicate episode id {ep.id} ({seen_ids[ep.id]} and {code}/{split})")
                    seen_ids[ep.id] = f"{code}/{split}"
                    local: set[str] = set()
                    for iid in _item_ids(ep):
                        prev = owner.get(iid)
                        if prev is not None and prev != split:
                            raise LineageOverlapError(f"{code}: item {iid!r} appears in splits {prev!r} and {split!r}")
                        if prev == split and iid not in local:
                            reused.append(iid)
                        owner[iid] = split
                        local.add(iid)
                if reused:
                    uniq = list(dict.fromkeys(reused))
                    warnings.append(f"{code}/{split}: {len(uniq)} item ids reused by several episodes of the split "
                                    f"(e.g. {uniq[:3]})")
                episodes[split][code] = eps
            reps = []
            for r, ep in enumerate(episodes["src"][code], start=1):
                rep = dataclasses.replace(ep, split="rep",
                                          lineage={**copy.deepcopy(ep.lineage), "rep_of": ep.id, "round": r})
                reps.append(rep)
            episodes["rep"][code] = reps
        return cls(episodes=episodes, order=order, seed=seed, items_per_episode=ipe, rounds=counts["src"],
                   counts=counts, seeds=seeds, adapters_info=info, warnings=warnings)

    # ---------------------------------------------------------------- access
    @property
    def disciplines(self) -> list[str]:
        return list(self.order)

    def split_episodes(self, split: str) -> list[Episode]:
        """All episodes of a split, flattened in discipline order (episode order within a discipline kept)."""
        if split == "rep":
            return self.rep_until(self.rounds)
        by_d = self.episodes.get(split, {})
        return [ep for code in self.order for ep in by_d.get(code, [])]

    def source_stream(self) -> list[tuple[int, Episode]]:
        """(round, episode) for rounds 1..R, round-major, fixed discipline order within a round."""
        out: list[tuple[int, Episode]] = []
        for r in range(1, self.rounds + 1):
            for code in self.order:
                eps = self.episodes["src"].get(code, [])
                if r - 1 < len(eps):
                    out.append((r, eps[r - 1]))
        return out

    def source_round(self, episode_id: str) -> int | None:
        for r, ep in self.source_stream():
            if ep.id == episode_id:
                return r
        return None

    def rep_until(self, round_: int) -> list[Episode]:
        """Frozen copies (split 'rep') of the source episodes of rounds <= ``round_`` (round-major order)."""
        out: list[Episode] = []
        for r in range(1, min(int(round_), self.rounds) + 1):
            for code in self.order:
                reps = self.episodes["rep"].get(code, [])
                if r - 1 < len(reps):
                    out.append(reps[r - 1])
        return out

    def get(self, episode_id: str, split: str | None = None) -> Episode:
        splits = [split] if split else [*BUILT_SPLITS, "rep"]
        for s in splits:
            for eps in self.episodes.get(s, {}).values():
                for ep in eps:
                    if ep.id == episode_id:
                        return ep
        raise KeyError(f"episode {episode_id!r} not in plan" + (f" split {split!r}" if split else ""))

    def discipline_of(self, episode_id: str) -> str | None:
        for s in BUILT_SPLITS:
            for code, eps in self.episodes.get(s, {}).items():
                if any(ep.id == episode_id for ep in eps):
                    return code
        return None

    # -------------------------------------------------------------- manifest
    def manifest(self) -> dict:
        splits: dict[str, dict[str, list[dict]]] = {}
        for split in BUILT_SPLITS:
            splits[split] = {}
            for code in self.order:
                rows = []
                for i, ep in enumerate(self.episodes[split].get(code, [])):
                    row = {"id": ep.id, "n_items": ep.n_items, "family": ep.family, "lineage": ep.lineage}
                    if split == "src":
                        row["round"] = i + 1
                    rows.append(row)
                splits[split][code] = rows
        m = {
            "schema": MANIFEST_SCHEMA,
            "seed": self.seed,
            "items_per_episode": self.items_per_episode,
            "rounds": self.rounds,
            "counts": self.counts,
            "disciplines": list(self.order),
            "adapters": self.adapters_info,
            "split_seeds": self.seeds,
            "splits": splits,
            "rep": {"of": "src", "rule": "rep_until(r) = frozen copies of src episodes of rounds <= r",
                    "episodes": [{"id": ep.id, "round": ep.lineage.get("round")} for ep in self.rep_until(self.rounds)]},
            "warnings": list(self.warnings),
        }
        # normalise through JSON so the in-memory manifest equals the saved one
        m = json.loads(canonical_json(m))
        m["sha256"] = manifest_hash(m)
        return m

    def save(self, path: str | Path) -> dict:
        m = self.manifest()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(m, indent=1, sort_keys=True, ensure_ascii=False, default=_json_default))
        return m

    @staticmethod
    def load_manifest(path: str | Path, verify: bool = True) -> dict:
        m = json.loads(Path(path).read_text())
        if verify and m.get("sha256") != manifest_hash(m):
            raise ValueError(f"split manifest {path} is corrupted: sha256 mismatch")
        return m

    def verify_against(self, manifest: dict) -> None:
        """Raise if this (rebuilt) plan differs from a frozen manifest."""
        mine = self.manifest()
        if mine["sha256"] != manifest.get("sha256"):
            diff = [s for s in BUILT_SPLITS if mine["splits"].get(s) != manifest.get("splits", {}).get(s)]
            raise ValueError(f"rebuilt split plan does not match the frozen manifest (sha {mine['sha256'][:12]} vs "
                             f"{str(manifest.get('sha256'))[:12]}; differing splits: {diff or 'metadata'})")


# ---------------------------------------------------------------------------------------------- adapter loading
def _instantiate(module: str, data_root: str | None) -> TaskAdapter:
    mod = importlib.import_module(f"scienceclaw.bench.tasks.{module}")
    cls = mod.Adapter
    params = inspect.signature(cls).parameters
    takes_root = "data_root" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    return cls(data_root=data_root) if (takes_root and data_root) else cls()


def adapter_status(bench_cfg: Any = None) -> list[dict]:
    """One row per registry discipline (+ TOY): module present?, data available?, reason."""
    root = getattr(bench_cfg, "data_root", None) if bench_cfg is not None else None
    rows = []
    for d in DISCIPLINES:
        row = {"code": d.code, "family": d.family, "dataset": d.dataset, "metric": d.metric, "direction": d.direction,
               "module": d.module, "implemented": False, "available": False, "reason": ""}
        try:
            ad = _instantiate(d.module, root)
        except ModuleNotFoundError as ex:
            row["reason"] = f"adapter module missing ({ex.name})"
            rows.append(row)
            continue
        except Exception as ex:  # adapter constructor failed: report, do not crash the table
            row["implemented"] = True
            row["reason"] = f"adapter init failed: {type(ex).__name__}: {ex}"
            rows.append(row)
            continue
        row["implemented"] = True
        try:
            ok, why = ad.available()
        except Exception as ex:
            ok, why = False, f"available() raised {type(ex).__name__}: {ex}"
        row["available"], row["reason"] = bool(ok), str(why)
        rows.append(row)
    rows.append({"code": TOY_CODE, "family": "Engineering & computing", "dataset": "synthetic regression (offline)",
                 "metric": "RMSE", "direction": "min", "module": "toy", "implemented": True, "available": True,
                 "reason": "synthetic"})
    return rows


def load_adapters(bench_cfg: Any, strict: bool | None = None) -> dict[str, TaskAdapter]:
    """Instantiate the adapters a run needs.

    ``bench_cfg.disciplines`` non-empty: exactly those codes (``"TOY"`` allowed); a missing/unavailable adapter
    raises (``strict`` defaults to True). Empty: every available registry discipline (TOY is never added
    implicitly); unavailable ones are skipped (``strict`` defaults to False).
    """
    wanted = list(getattr(bench_cfg, "disciplines", None) or [])
    strict = bool(wanted) if strict is None else strict
    root = getattr(bench_cfg, "data_root", None)
    codes = wanted or [d.code for d in DISCIPLINES]
    out: dict[str, TaskAdapter] = {}
    problems: list[str] = []
    for code in codes:
        if code == TOY_CODE:
            from .tasks.toy import ToyAdapter
            out[code] = ToyAdapter()
            continue
        if code not in BY_CODE:
            problems.append(f"{code}: unknown discipline code")
            continue
        try:
            ad = _instantiate(BY_CODE[code].module, root)
        except ModuleNotFoundError as ex:
            problems.append(f"{code}: adapter module missing ({ex.name})")
            continue
        ok, why = ad.available()
        if not ok:
            problems.append(f"{code}: unavailable ({why})")
            continue
        out[code] = ad
    if strict and problems:
        raise RuntimeError("cannot load requested adapters: " + "; ".join(problems))
    return out
