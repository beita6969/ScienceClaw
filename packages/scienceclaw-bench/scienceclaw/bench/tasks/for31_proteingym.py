"""FoR31 Biological sciences — ProteinGym v1.3 DMS substitutions (mean Spearman across assays).

Item
    One deep-mutational-scanning (DMS) assay of the ProteinGym substitution benchmark, restricted to single amino-acid
    substitutions: a fixed set of held-out *query* variants (up to 128) whose DMS scores must be ranked, plus up to
    1024 visible labelled training variants and up to 32 dev variants of the same assay (all three sets disjoint,
    fixed by ``pool_seed``). The per-item metric is the Spearman correlation between predictions and DMS scores on
    the query variants; an episode of 16 items reports the mean over its assays (the ProteinGym headline metric is
    Spearman per assay averaged over assays). Design note: an item is an assay's query-variant set (not a single
    variant), because Spearman is only defined per assay.
IID / OOD
    IID: the 153 functional DMS assays (activity, binding, expression, organismal fitness; many labs).
    OOD: the 64 Tsuboyama et al. 2023 mega-scale cDNA-display proteolysis *stability* assays — a different source
    study and assay technology on 64 small domains that share no UniProt entry with the IID assays
    (``lineage["ood_kind"] = "cross_source_within_benchmark"``).
Splits (fixed by ``pool_seed``; protein-disjoint)
    IID assays grouped by UniProt entry name and allocated to id (64 assays), val (32) and src (remainder).
    Assays need >= 160 single substitutions to be eligible.
Metric (D_V)
    Spearman rank correlation (scipy.stats.spearmanr, average ranks for ties) per assay; constant predictions give
    an undefined correlation, scored 0. Episode primary = mean over items; pooled = mean over assays (an assay seen in
    several episodes is averaged within the assay first).
Reference baseline
    Site-mean: each query variant gets the mean training DMS score at its position (training-mean of the assay if
    the position has no training variant). Acceptance: primary > reference + ``ACCEPT_MARGIN``.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import scilib
from ...core.schema import PortSchema
from ..registry import DATA_ROOT
from ..task import ConstraintSpec, Episode, EvalResult, ToolSpec
from ._life_health_common import (PROTOCOL_SEED, Memo, allocate_groups, compose_episodes, default_budget,
                                  default_cache_dir, effective_items, extract_payloads, ids_hash, make_result, rng_for,
                                  tmp_path_for, unit_constraint)

CODE = "FoR31"
DATASET_DIR = "for31-proteingym-substitutions"
AA = "ARNDCQEGHILKMFPSTWYV"
MIN_SINGLES = 160
N_QUERY, N_DEV, N_TRAIN = 128, 32, 1024
IID_TARGETS = [("id", 64), ("val", 32)]          # src = remainder
OOD_TARGET = 64
ACCEPT_MARGIN = 0.03                             # absolute Spearman above the site-mean reference
_MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")

# BLOSUM62 (NCBI), rows/cols in AA order
_B62 = """
 4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
 0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
 0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
 1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
 0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
 0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""
BLOSUM62 = np.array([[int(v) for v in line.split()] for line in _B62.strip().splitlines()], dtype=int)
KYTE_DOOLITTLE = dict(zip(AA, [1.8, -4.5, -3.5, -3.5, 2.5, -3.5, -3.5, -0.4, -3.2, 4.5, 3.8, -3.9, 1.9, 2.8, -1.6,
                               -0.8, -0.7, -0.9, -1.3, 4.2]))
VOLUME_A3 = dict(zip(AA, [88.6, 173.4, 114.1, 111.1, 108.5, 143.8, 138.4, 60.1, 153.2, 166.7, 166.7, 168.6, 162.9,
                          189.9, 112.7, 89.0, 116.1, 227.8, 193.6, 140.0]))   # Zamyatnin 1972
CHARGE = {a: (1.0 if a in "KR" else -1.0 if a in "DE" else 0.0) for a in AA}
POLAR = set("RNDQEHKSTY")


def protein_of(assay: str) -> str:
    """UniProt entry name from a ProteinGym assay id: leading all-caps/digit tokens ('BLAT_ECOLX_Jacquier_2013')."""
    out = []
    for t in assay.split("_"):
        if re.fullmatch(r"[A-Z0-9]+", t):
            out.append(t)
        else:
            break
    return "_".join(out) or assay


def substitution_features(table: pd.DataFrame) -> pd.DataFrame:
    """Per-variant physico-chemical descriptors of the substitution wt_aa -> mut_aa (row order preserved)."""
    wt = table["wt_aa"].astype(str).to_numpy()
    mt = table["mut_aa"].astype(str).to_numpy()
    ai = {a: i for i, a in enumerate(AA)}
    bl = np.array([BLOSUM62[ai[w], ai[m]] if w in ai and m in ai else np.nan for w, m in zip(wt, mt)], dtype=float)

    def prop(d: dict, s: np.ndarray) -> np.ndarray:
        return np.array([d.get(x, np.nan) for x in s], dtype=float)

    return pd.DataFrame({
        "blosum62": bl,
        "hydropathy_wt": prop(KYTE_DOOLITTLE, wt), "hydropathy_mut": prop(KYTE_DOOLITTLE, mt),
        "hydropathy_delta": prop(KYTE_DOOLITTLE, mt) - prop(KYTE_DOOLITTLE, wt),
        "volume_delta_A3": prop(VOLUME_A3, mt) - prop(VOLUME_A3, wt),
        "charge_delta": prop(CHARGE, mt) - prop(CHARGE, wt),
        "polarity_change": np.array([float((w in POLAR) != (m in POLAR)) for w, m in zip(wt, mt)]),
        "to_proline": (mt == "P").astype(float), "from_glycine": (wt == "G").astype(float),
    }, index=table.index)


def spearman(pred: np.ndarray, true: np.ndarray) -> float:
    from scipy.stats import spearmanr

    pred, true = np.asarray(pred, dtype=float), np.asarray(true, dtype=float)
    if len(pred) < 3 or np.all(pred == pred[0]) or np.all(true == true[0]):
        return 0.0
    r = spearmanr(pred, true).statistic
    return float(r) if np.isfinite(r) else 0.0


class Adapter:
    discipline = CODE
    name = "ProteinGym substitutions"
    family = "Life & health"
    metric = "mean Spearman"
    direction = "max"
    task_type = "protein_variant_effect_prediction"

    def __init__(self, data_root: str | None = None, cache_dir: str | None = None, pool_seed: int = PROTOCOL_SEED,
                 **_: Any) -> None:
        self.base = Path(data_root or os.environ.get("SCIENCECLAW_DATA_ROOT", DATA_ROOT)) / DATASET_DIR
        self.root = self.base / "data" / "DMS_ProteinGym_substitutions"
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir(CODE)
        self.pool_seed = int(pool_seed)
        self._memo = Memo()

    # ---------------------------------------------------------------- data
    def _files(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(f for f in self.root.glob("*.csv") if not f.name.startswith("._"))

    def available(self) -> tuple[bool, str]:
        files = self._files()
        if len(files) < 200:
            return False, f"expected 217 ProteinGym v1.3 assay CSVs under {self.root}; found {len(files)}"
        try:
            pools = self._pools()
        except (ValueError, OSError, KeyError) as ex:
            return False, f"cannot build pools: {type(ex).__name__}: {ex}"
        return True, f"{len(files)} assays; " + ", ".join(f"{k}={len(v)}" for k, v in pools.items())

    def _read_singles(self, f: Path) -> tuple[pd.DataFrame, str]:
        """Single-substitution rows (mutant, position, wt_aa, mut_aa, DMS_score) and the wild-type sequence."""
        df = pd.read_csv(f, usecols=["mutant", "DMS_score"])
        df = df.dropna(subset=["mutant", "DMS_score"])
        m = df["mutant"].astype(str).str.extract(_MUT_RE)
        ok = m.notna().all(axis=1) & m[0].isin(list(AA)) & m[2].isin(list(AA))
        df, m = df[ok], m[ok]
        out = pd.DataFrame({"mutant": df["mutant"].astype(str).to_numpy(), "position": m[1].astype(int).to_numpy(),
                            "wt_aa": m[0].to_numpy(), "mut_aa": m[2].to_numpy(),
                            "DMS_score": df["DMS_score"].astype(float).to_numpy()})
        out = out[out["wt_aa"] != out["mut_aa"]].drop_duplicates("mutant", keep="first").reset_index(drop=True)
        wt = ""
        if len(out):
            first = out.iloc[0]
            for chunk in pd.read_csv(f, usecols=["mutant", "mutated_sequence"], chunksize=2000):
                hit = chunk[chunk["mutant"].astype(str) == first["mutant"]]
                if len(hit):
                    seq = list(str(hit.iloc[0]["mutated_sequence"]))
                    p = int(first["position"]) - 1
                    if 0 <= p < len(seq) and seq[p] == first["mut_aa"]:
                        seq[p] = first["wt_aa"]
                        wt = "".join(seq)
                    break
        if wt:   # drop rows inconsistent with the wild type (none expected in v1.3)
            cons = np.array([0 < p <= len(wt) and wt[p - 1] == w for p, w in zip(out["position"], out["wt_aa"])])
            out = out[cons].reset_index(drop=True)
        return out, wt

    def _index(self) -> dict[str, dict]:
        """assay -> {"protein", "tsuboyama", "n_singles", "wt_len"}; JSON cache keyed by file names and sizes."""
        def build() -> dict[str, dict]:
            files = self._files()
            key = ids_hash([f"{f.name}:{f.stat().st_size}" for f in files])
            cache = self.cache_dir / f"index_{key}.json"
            if cache.exists():
                return json.loads(cache.read_text())
            idx = {}
            for f in files:
                singles, wt = self._assay(f.stem, f)
                idx[f.stem] = {"protein": protein_of(f.stem), "tsuboyama": "Tsuboyama" in f.stem,
                               "n_singles": int(len(singles)), "wt_len": len(wt)}
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = tmp_path_for(cache)
            tmp.write_text(json.dumps(idx, indent=0))
            os.replace(tmp, cache)
            return idx
        return self._memo.get("index", build)

    def _assay(self, assay: str, f: Path | None = None) -> tuple[pd.DataFrame, str]:
        """Parsed singles + WT of one assay (parquet/JSON cache under cache_dir/assays)."""
        def load() -> tuple[pd.DataFrame, str]:
            src = f or (self.root / f"{assay}.csv")
            d = self.cache_dir / "assays"
            pq, js = d / f"{assay}.parquet", d / f"{assay}.json"
            size = src.stat().st_size
            if pq.exists() and js.exists():
                meta = json.loads(js.read_text())
                if meta.get("source_bytes") == size:
                    return pd.read_parquet(pq), meta["wild_type"]
            singles, wt = self._read_singles(src)
            d.mkdir(parents=True, exist_ok=True)
            tp, tj = tmp_path_for(pq), tmp_path_for(js)
            singles.to_parquet(tp, index=False)
            os.replace(tp, pq)
            tj.write_text(json.dumps({"wild_type": wt, "source_bytes": size}))
            os.replace(tj, js)
            return singles, wt
        return self._memo.get(f"assay:{assay}", load)

    def _pools(self) -> dict[str, list[str]]:
        def build() -> dict[str, list[str]]:
            idx = self._index()
            elig = {a: v for a, v in idx.items() if v["n_singles"] >= MIN_SINGLES and v["wt_len"] > 0}
            iid: dict[str, list[str]] = {}
            for a, v in sorted(elig.items()):
                if not v["tsuboyama"]:
                    iid.setdefault(v["protein"], []).append(a)
            ood_all = sorted(a for a, v in elig.items() if v["tsuboyama"])
            iid_prot, ood_prot = set(iid), {elig[a]["protein"] for a in ood_all}
            ood_all = [a for a in ood_all if elig[a]["protein"] not in iid_prot]     # protein-disjoint OOD
            pools = allocate_groups(iid, IID_TARGETS, "src", [CODE, self.pool_seed, "iid"])
            og = {a: [a] for a in ood_all}
            pools["ood"] = allocate_groups(og, [("ood", OOD_TARGET)], "ood_unused", [CODE, self.pool_seed, "ood"])["ood"]
            if len(pools["id"]) < 16 or len(pools["src"]) < 16 or len(pools["val"]) < 16 or len(pools["ood"]) < 16:
                raise ValueError("not enough eligible assays: " + ", ".join(f"{k}={len(v)}" for k, v in pools.items()))
            if iid_prot & ood_prot:
                raise ValueError("IID and OOD proteins overlap")
            return pools
        return self._memo.get(f"pools:{self.pool_seed}", build)

    def _item(self, assay: str) -> dict:
        """Fixed disjoint query / dev / train variant sets of one assay (independent of the episode seed)."""
        def build() -> dict:
            singles, wt = self._assay(assay)
            n = len(singles)
            nq, nd = min(N_QUERY, int(0.4 * n)), min(N_DEV, int(0.1 * n))
            nt = min(N_TRAIN, n - nq - nd)
            perm = rng_for(CODE, self.pool_seed, assay, "variants").permutation(n)
            q, d, t = perm[:nq], perm[nq:nq + nd], perm[nq + nd:nq + nd + nt]
            return {"query": singles.iloc[np.sort(q)].reset_index(drop=True),
                    "dev": singles.iloc[np.sort(d)].reset_index(drop=True),
                    "train": singles.iloc[np.sort(t)].reset_index(drop=True), "wild_type": wt}
        return self._memo.get(f"item:{self.pool_seed}:{assay}", build)

    # ---------------------------------------------------------------- reference and scoring
    @staticmethod
    def site_mean_predictions(train: pd.DataFrame, query: pd.DataFrame) -> np.ndarray:
        means = train.groupby("position")["DMS_score"].mean()
        glob = float(train["DMS_score"].mean())
        return query["position"].map(means).fillna(glob).to_numpy(dtype=float)

    @staticmethod
    def _coerce(y: Any, n: int) -> tuple[np.ndarray | None, str]:
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.to_numpy()
        try:
            a = np.asarray(y, dtype=float)
        except (TypeError, ValueError):
            return None, "y must be a numeric 1-D array"
        a = a.reshape(-1) if a.ndim == 2 and 1 in a.shape else a
        if a.ndim != 1 or a.shape[0] != n:
            return None, f"y must have shape ({n},), got {a.shape}"
        if not np.all(np.isfinite(a)):
            return None, "y contains non-finite values"
        return a, "ok"

    def _score(self, assays: list[str], key: str, pred: np.ndarray) -> dict:
        rhos, pos = [], 0
        for a in assays:
            truth = self._item(a)[key]["DMS_score"].to_numpy()
            rhos.append(spearman(pred[pos:pos + len(truth)], truth))
            pos += len(truth)
        return {"rhos": rhos, "mean": float(np.mean(rhos)) if rhos else 0.0}

    def pooled_metric(self, per_episode: list[dict]) -> float | None:
        """Mean Spearman over assays (per-assay values of repeated assays are averaged first)."""
        pays = extract_payloads(per_episode, "spearman_by_assay")
        by: dict[str, list[float]] = {}
        for p in pays:
            for a, r in zip(p["assays"], p["spearman"]):
                by.setdefault(a, []).append(float(r))
        return float(np.mean([np.mean(v) for v in by.values()])) if by else None

    # ---------------------------------------------------------------- episodes
    def build_episodes(self, split: str, n: int, seed: int, items_per_episode: int = 16) -> list[Episode]:
        if split not in ("src", "val", "id", "ood", "rep"):
            raise ValueError(f"unknown split {split!r}")
        ok, why = self.available()
        if not ok:
            raise RuntimeError(f"{CODE} unavailable: {why}")
        base = "src" if split == "rep" else split
        pool = self._pools()[base]
        k = effective_items(split, len(pool), n, int(items_per_episode))
        groups, reused = compose_episodes(pool, n, k, [CODE, self.pool_seed, base, seed, items_per_episode])
        eps = [self._episode(split, base, seed, j, assays, reused) for j, assays in enumerate(groups)]
        for e in eps:
            e.lineage["items_requested"] = int(items_per_episode)
        return eps

    def _tables(self, assays: list[str], key: str, with_score: bool) -> pd.DataFrame:
        parts = []
        for i, a in enumerate(assays):
            t = self._item(a)[key].copy()
            t.insert(0, "assay_id", a)
            t.insert(0, "item", i)
            parts.append(t)
        df = pd.concat(parts, ignore_index=True)
        cols = ["item", "assay_id", "mutant", "position", "wt_aa", "mut_aa"] + (["DMS_score"] if with_score else [])
        return df[cols].reset_index(drop=True)

    def _episode(self, split: str, base: str, seed: int, j: int, assays: list[str], reused: bool) -> Episode:
        k = len(assays)
        n_query = [len(self._item(a)["query"]) for a in assays]
        n_dev = [len(self._item(a)["dev"]) for a in assays]
        N, ND = int(sum(n_query)), int(sum(n_dev))
        wild = {a: self._item(a)["wild_type"] for a in assays}

        def t_load_train(inputs: dict, config: dict) -> dict:
            return {"table": self._tables(assays, "train", True), "wild_type": dict(wild)}

        def t_load_eval(inputs: dict, config: dict) -> dict:
            return {"table": self._tables(assays, "query", False), "wild_type": dict(wild)}

        def t_load_dev(inputs: dict, config: dict) -> dict:
            return {"table": self._tables(assays, "dev", False), "wild_type": dict(wild)}

        def t_score_dev(inputs: dict, config: dict) -> dict:
            a, msg = self._coerce(inputs.get("predictions"), ND)
            if a is None:
                raise ValueError(f"score_dev: {msg}")
            s = self._score(assays, "dev", a)
            return {"mean_spearman": s["mean"], "per_item_spearman": [float(r) for r in s["rhos"]]}

        def t_features(inputs: dict, config: dict) -> dict:
            tab = inputs.get("table")
            if not isinstance(tab, pd.DataFrame) or not {"wt_aa", "mut_aa"} <= set(tab.columns):
                raise ValueError("substitution_features needs a table with columns wt_aa and mut_aa")
            return {"features": substitution_features(tab)}

        var_cols = ("item (0-based item index), assay_id, mutant (e.g. 'A24G' = wild-type residue, 1-based position, "
                    "mutant residue), position, wt_aa, mut_aa")
        tools = [
            ToolSpec("load_train", "Visible labelled single-substitution variants of every item's assay, plus the "
                                   "wild-type sequence of each assay.", {},
                     {"table": PortSchema("table", ("rows", 7), None, None, f"columns: {var_cols}, DMS_score"),
                      "wild_type": PortSchema("dict", None, None, "str", "assay_id -> wild-type amino-acid sequence")},
                     t_load_train),
            ToolSpec("load_eval_inputs", "Query variants to score (no DMS scores); row order defines the order of y.", {},
                     {"table": PortSchema("table", (N, 6), None, None, f"columns: {var_cols}"),
                      "wild_type": PortSchema("dict", None, None, "str", "assay_id -> wild-type sequence")},
                     t_load_eval),
            ToolSpec("load_dev_inputs", "Visible dev variants of the same assays (scores withheld; see score_dev).", {},
                     {"table": PortSchema("table", (ND, 6), None, None, f"columns: {var_cols}"),
                      "wild_type": PortSchema("dict", None, None, "str", "assay_id -> wild-type sequence")},
                     t_load_dev),
            ToolSpec("score_dev", "Mean per-assay Spearman correlation of dev predictions (row order of load_dev_inputs).",
                     {"predictions": PortSchema("array", (ND,), "1", "float", "one score per dev row")},
                     {"mean_spearman": PortSchema("number", None, "1", "float", ""),
                      "per_item_spearman": PortSchema("list", (k,), "1", "float", "")},
                     t_score_dev),
            ToolSpec("substitution_features", "Physico-chemical descriptors per substitution row: BLOSUM62 score, "
                                              "Kyte-Doolittle hydropathy (wt, mut, delta), side-chain volume delta "
                                              "(A^3), formal charge delta, polarity change, to-proline, from-glycine.",
                     {"table": PortSchema("table", None, None, None, "any table with columns wt_aa and mut_aa")},
                     {"features": PortSchema("table", ("rows", 9), None, "float", "descriptor table, same row order")},
                     t_features),
        ]
        objective = (
            "Prediction of the effects of single amino-acid substitutions on protein function measured by deep "
            "mutational scanning (ProteinGym v1.3 DMS substitution benchmark).\n"
            f"Evaluation items: {k} DMS assays (tool load_eval_inputs; column item = 0..{k - 1}). Each assay measures "
            "one phenotype of one protein's variants (e.g. activity, binding, expression, organismal fitness or "
            "folding stability, depending on the assay); DMS_score is oriented so that higher = fitter / more "
            "functional / more stable. For every item, the query rows are single substitutions of the wild-type "
            "sequence (tool output wild_type). Visible labelled variants of the same assays (disjoint from the query "
            "rows): tool load_train.\n"
            f"Required output y: a one-dimensional float array of length {N}, one predicted score per row of the "
            "load_eval_inputs table, in that row order; higher = predicted fitter. Only the ranking within each item "
            "matters.\n"
            "Score: Spearman rank correlation between y and the measured DMS_score, computed separately for every "
            "item (an item with constant predictions scores 0) and averaged over the items.\n"
            + scilib.describe("proteinfit") + scilib.describe_extra("proteinplm"))
        constraints = [
            ConstraintSpec("output_shape", f"y is a 1-D numeric array of length {N} aligned with the query rows",
                           lambda y, tr: self._check_shape(y, N)),
            ConstraintSpec("finite_values", "all predictions are finite numbers",
                           lambda y, tr: (lambda r: (r[0] is not None, r[1]))(self._coerce(y, N))),
            unit_constraint("1"),
        ]
        ref_memo = Memo()

        def reference() -> dict:
            pred = np.concatenate([self.site_mean_predictions(self._item(a)["train"], self._item(a)["query"])
                                   for a in assays])
            return self._score(assays, "query", pred)

        def evaluate(y: Any, trace: Any) -> EvalResult:
            ref = ref_memo.get("ref", reference)
            a, msg = self._coerce(y, N)
            valid = a is not None
            s = self._score(assays, "query", a) if valid else {"rhos": [0.0] * k, "mean": 0.0}
            payload = {"kind": "spearman_by_assay", "assays": list(assays), "spearman": [float(r) for r in s["rhos"]],
                       "n_query": n_query, "fallback": not valid}
            metrics = {"mean_spearman": s["mean"], "reference_mean_spearman": ref["mean"],
                       "items_beating_reference": float(sum(r > q for r, q in zip(s["rhos"], ref["rhos"])))}
            return make_result(s["mean"], ref["mean"], "max", ACCEPT_MARGIN, metrics,
                               {"pooled_payload": payload, "parse": msg, "per_item_spearman": s["rhos"],
                                "reference_per_item": ref["rhos"], "reference_desc": "site-mean of training scores"},
                               valid)

        idx = self._index()
        lineage = {"dataset": "ProteinGym DMS substitutions", "version": "v1.3",
                   "source_url": "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip",
                   "license": "https://github.com/OATML-Markslab/ProteinGym/blob/main/LICENSE (MIT code; per-assay data licenses of the original studies)",
                   "item_kind": "DMS assay (held-out single-substitution query set)", "item_ids": list(assays),
                   "proteins": sorted({idx[a]["protein"] for a in assays}), "n_query_per_item": n_query,
                   "item_pool": "ood" if base == "ood" else "iid",
                   "ood_kind": "cross_source_within_benchmark" if base == "ood" else None,
                   "ood_rule": "Tsuboyama 2023 cDNA-display proteolysis stability assays (proteins disjoint from IID)"
                   if base == "ood" else None,
                   "pool_seed": self.pool_seed, "episode_seed": int(seed), "episode_index": j, "reused_items": bool(reused),
                   "variant_split": f"per assay: query<= {N_QUERY}, dev<= {N_DEV}, train<= {N_TRAIN}, disjoint, fixed by pool_seed",
                   "rebuilt_split": True, "historical_ids_recovered": False}
        return Episode(
            id=f"{CODE}-{split}-s{seed}-e{j:02d}", discipline=CODE, family=self.family, split=split,
            task_type=self.task_type, objective=objective,
            required_output=PortSchema("array", (N,), "1", "float", "predicted fitness score per query row (row order)"),
            tools=tools, constraints=constraints, budget=default_budget(k, max_node_s=300, max_wall_s=1800),
            lineage=lineage,
            acceptance=f"accepted iff mean Spearman > reference (site-mean on the same items) + {ACCEPT_MARGIN}",
            tolerance={"rtol": 1e-6, "atol": 1e-8}, tags=["protein", "mutation", "dms", "fitness", "biology"],
            metric=self.metric, direction=self.direction, n_items=k, _evaluate=evaluate, _dev_evaluate=None)

    @staticmethod
    def _check_shape(y: Any, n: int) -> tuple[bool, str]:
        try:
            a = np.asarray(y.to_numpy() if isinstance(y, (pd.Series, pd.DataFrame)) else y)
        except (TypeError, ValueError) as ex:
            return False, f"cannot convert y: {ex}"
        a = a.reshape(-1) if a.ndim == 2 and 1 in a.shape else a
        if a.ndim != 1 or a.shape[0] != n:
            return False, f"shape {a.shape}, expected ({n},)"
        return True, "ok"

    def describe_pools(self) -> dict:
        idx = self._index()
        return {k: {"n_assays": len(v), "n_proteins": len({idx[a]["protein"] for a in v})}
                for k, v in self._pools().items()}


if __name__ == "__main__":      # pragma: no cover
    a = Adapter()
    print(a.available())
    print(json.dumps(a.describe_pools(), indent=1))
