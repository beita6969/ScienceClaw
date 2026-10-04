"""FoR42 Health sciences — PhysioNet/CinC Challenge 2019 early sepsis prediction (normalized clinical utility).

Item
    One complete ICU stay (all hourly rows of one patient PSV file). The model outputs one binary label per hour.
IID / OOD
    IID items: hospital A of the challenge's public training data (``training_setA``).
    OOD items: hospital B (``training_setB``) — a different hospital system of the same challenge, which is the
    challenge's own cross-site shift (``lineage["ood_kind"] = "cross_hospital_within_dataset"``). Visible
    training data always come from hospital A only.
Splits (fixed by ``pool_seed``; stratified by stay outcome)
    From the locally available hospital-A stays: id pool 64 stays (8 septic), val pool 32 (4 septic), src pool
    128 (16 septic), dev slice 80 (8 septic, labels never returned by a tool), visible training = all remaining
    A stays. OOD pool: 64 hospital-B stays containing all 8 locally available septic B stays. Every episode of
    16 stays contains exactly 2 septic stays (12.5 %; challenge prevalence 7.3 %), so the utility normalization
    is always defined.
Metric (D_V)
    Normalized clinical utility exactly as in the official ``evaluate_sepsis_score.py`` of the 2019 challenge
    (Reyna et al., Crit Care Med 2020): dt_early=-12, dt_optimal=-6, dt_late=3, max_u_tp=1, min_u_fn=-2,
    u_fp=-0.05, u_tn=0; t_sepsis = argmax(SepsisLabel) - dt_optimal; normalized =
    (U_obs - U_inaction) / (U_best - U_inaction) summed over the stays of the episode.
Reference baseline
    Logistic regression on trivial causal features (last-observation-carried-forward HR, O2Sat, Temp, SBP, MAP,
    Resp, plus Age, Gender, ICULOS, HospAdmTime; training-median imputation, z-scoring), class-balanced, fitted
    on the visible training stays; decision threshold chosen on the visible training stays to maximise normalized
    utility. Acceptance: primary > max(reference, ``ACCEPT_FLOOR``) + ``ACCEPT_MARGIN`` (an episode whose
    reference utility is negative must not be passed by the all-zero output).
Causality
    The stays are given whole, so "the prediction for hour t may use rows 0..t only" is a protocol rule that the
    adapter checks in two ways. (1) Visible constraint ``not_positional_only``: y must not be a function of the
    position in the stay alone (hour index or distance to the end of the record) - this rejects the length-only
    rules that exploit the dataset's record ends (septic records end 8-10 h after the first positive label; "flag the
    last 12 hours" scores NU 0.66-0.72 on full stays). (2) Hidden constraint ``causal_prefix``: a truncation-consistency
    probe (``Episode.run_probes``): the caller re-runs the *same graph* on stays cut at hidden hours and the
    predictions for the hours kept must not change (at most ``CAUSAL_TOL`` of the prefix hours). The probe needs the
    graph, which only the solver has, so the verdict is read from ``trace.probes``; without a probe run the constraint
    reports "not probed" and passes. ``n_hours`` and the row counts stay visible (they are needed to shape y and are
    derivable from the table anyway); using them for the prediction violates the rule and is what the probe detects.
Visible dev signal
    Tool ``score_dev`` scores binary predictions for the dev stays returned by ``load_dev_inputs`` (dev labels are
    never returned). ``Episode._dev_evaluate`` is None.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import scilib
from ...core.schema import PortSchema
from ..registry import DATA_ROOT
from ..task import ConstraintSpec, Episode, EvalResult, Probe, ToolSpec
from ._life_health_common import (PROTOCOL_SEED, Memo, as_list_of_arrays, compose_episodes, default_budget,
                                  default_cache_dir, effective_items, extract_payloads, ids_hash, make_result,
                                  stratified_take, tmp_path_for, unit_constraint)

CODE = "FoR42"
DATASET_DIR = "for42-physionet2019-sepsis"
FEATURES = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "HCO3", "FiO2", "pH",
            "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
            "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct",
            "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2", "HospAdmTime",
            "ICULOS"]
FEATURE_UNITS = {"HR": "beats/min", "O2Sat": "%", "Temp": "deg C", "SBP": "mm Hg", "MAP": "mm Hg", "DBP": "mm Hg",
                 "Resp": "breaths/min", "EtCO2": "mm Hg", "BaseExcess": "mmol/L", "HCO3": "mmol/L", "FiO2": "fraction",
                 "pH": "1", "PaCO2": "mm Hg", "SaO2": "%", "AST": "IU/L", "BUN": "mg/dL", "Alkalinephos": "IU/L",
                 "Calcium": "mg/dL", "Chloride": "mmol/L", "Creatinine": "mg/dL", "Bilirubin_direct": "mg/dL",
                 "Glucose": "mg/dL", "Lactate": "mg/dL", "Magnesium": "mmol/dL", "Phosphate": "mg/dL",
                 "Potassium": "mmol/L", "Bilirubin_total": "mg/dL", "TroponinI": "ng/mL", "Hct": "%", "Hgb": "g/dL",
                 "PTT": "s", "WBC": "10^3/uL", "Fibrinogen": "mg/dL", "Platelets": "10^3/uL", "Age": "years",
                 "Gender": "0=female,1=male", "Unit1": "MICU flag", "Unit2": "SICU flag",
                 "HospAdmTime": "h (hospital admit minus ICU admit)", "ICULOS": "h since ICU admit"}
LABEL = "SepsisLabel"

# official utility parameters (evaluate_sepsis_score.py, PhysioNet/CinC 2019)
DT_EARLY, DT_OPTIMAL, DT_LATE = -12, -6, 3
MAX_U_TP, MIN_U_FN, U_FP, U_TN = 1.0, -2.0, -0.05, 0.0

POOL_SIZES = {"id": (8, 56), "val": (4, 28), "src": (16, 112), "dev": (8, 72)}   # (septic, non-septic) stays
OOD_POOL = (8, 56)
ACCEPT_MARGIN = 0.05          # absolute normalized-utility points above the reference
ACCEPT_FLOOR = 0.0            # the reference used for acceptance is max(reference, ACCEPT_FLOOR)
CAUSAL_TOL = 0.01             # truncation probe: max fraction of prefix hours whose prediction may change
PROBE_KEEP = (0.4, 0.8)       # truncation probe: each stay keeps a hidden fraction in this range of its hours
PROBE_MIN_ROWS = 4            # ... and at least this many rows (stays are never cut to nothing)
POSITION_MIN_STAYS = 4        # not_positional_only: a position must be shared by this many stays to count as evidence
REF_VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "Resp"]
REF_STATIC = ["Age", "Gender", "ICULOS", "HospAdmTime"]


# ----------------------------------------------------------------------------------------------------------------
# official utility
# ----------------------------------------------------------------------------------------------------------------
def compute_prediction_utility_official(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Line-by-line port of ``compute_prediction_utility`` from the official 2019 scorer (reference for tests)."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if len(predictions) != len(labels):
        raise ValueError("Numbers of predictions and labels must be the same.")
    if np.any(labels):
        is_septic = True
        t_sepsis = int(np.argmax(labels)) - DT_OPTIMAL
    else:
        is_septic = False
        t_sepsis = float("inf")
    n = len(labels)
    m_1 = float(MAX_U_TP) / float(DT_OPTIMAL - DT_EARLY)
    b_1 = -m_1 * DT_EARLY
    m_2 = float(-MAX_U_TP) / float(DT_LATE - DT_OPTIMAL)
    b_2 = -m_2 * DT_LATE
    m_3 = float(MIN_U_FN) / float(DT_LATE - DT_OPTIMAL)
    b_3 = -m_3 * DT_OPTIMAL
    u = np.zeros(n)
    for t in range(n):
        if t <= t_sepsis + DT_LATE:
            if is_septic and predictions[t]:
                if t <= t_sepsis + DT_OPTIMAL:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, U_FP)
                elif t <= t_sepsis + DT_LATE:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            elif not is_septic and predictions[t]:
                u[t] = U_FP
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + DT_OPTIMAL:
                    u[t] = 0
                elif t <= t_sepsis + DT_LATE:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            elif not is_septic and not predictions[t]:
                u[t] = U_TN
    return float(np.sum(u))


def prediction_utility(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Vectorized, numerically identical form of the official per-stay utility."""
    labels = np.asarray(labels).astype(int)
    pred = np.asarray(predictions).astype(bool)
    n = len(labels)
    t = np.arange(n, dtype=float)
    if not labels.any():
        return float(np.sum(np.where(pred, U_FP, U_TN)))
    ts = float(np.argmax(labels) - DT_OPTIMAL)
    m_1 = MAX_U_TP / (DT_OPTIMAL - DT_EARLY); b_1 = -m_1 * DT_EARLY
    m_2 = -MAX_U_TP / (DT_LATE - DT_OPTIMAL); b_2 = -m_2 * DT_LATE
    m_3 = MIN_U_FN / (DT_LATE - DT_OPTIMAL); b_3 = -m_3 * DT_OPTIMAL
    d = t - ts
    early = t <= ts + DT_OPTIMAL
    late = (~early) & (t <= ts + DT_LATE)
    u_tp = np.where(early, np.maximum(m_1 * d + b_1, U_FP), np.where(late, m_2 * d + b_2, 0.0))
    u_fn = np.where(early, 0.0, np.where(late, m_3 * d + b_3, 0.0))
    return float(np.sum(np.where(pred, u_tp, u_fn)))


def best_predictions(labels: np.ndarray) -> np.ndarray:
    """Official 'best' predictions: 1 on [t_sepsis + dt_early, t_sepsis + dt_late] for septic stays."""
    labels = np.asarray(labels).astype(int)
    n = len(labels)
    best = np.zeros(n, dtype=int)
    if labels.any():
        ts = int(np.argmax(labels)) - DT_OPTIMAL
        best[max(0, ts + DT_EARLY): min(ts + DT_LATE + 1, n)] = 1
    return best


def utility_triplet(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float, float]:
    """(observed, best, inaction) utilities of one stay."""
    return (prediction_utility(labels, predictions), prediction_utility(labels, best_predictions(labels)),
            prediction_utility(labels, np.zeros(len(labels), dtype=int)))


def normalized_utility(obs: float, best: float, inaction: float) -> float | None:
    den = best - inaction
    if den == 0:
        return None
    return float((obs - inaction) / den)


# ----------------------------------------------------------------------------------------------------------------
# causality checks
# ----------------------------------------------------------------------------------------------------------------
def positional_only(preds: list[np.ndarray], min_stays: int = POSITION_MIN_STAYS) -> tuple[bool, str]:
    """True iff ``preds`` is a function of the position within the stay alone, i.e. carries no patient information.

    Two position frames are tested: the hour index from the start and the distance to the end of the record (the
    dataset's septic records end 8-10 h after the first positive label, so "flag the last K hours" is a strong
    shortcut). In a frame the output is positional when (a) at least one position shared by ``min_stays`` or more
    stays is flagged by *all* of them and (b) no position shared by two or more stays is split (some 0, some 1).
    A classifier that reads the vital signs splits positions between septic and non-septic stays; constant outputs
    (all zeros) have no flagged position and are not positional.
    """
    arrs = [np.asarray(a).reshape(-1).astype(int) for a in preds]
    if len(arrs) < min_stays:
        return False, "too few stays to test"
    width = max((len(a) for a in arrs), default=0)
    for frame, seqs in (("hour index from the start", arrs), ("distance to the end of the record", [a[::-1] for a in arrs])):
        mat = np.full((len(seqs), width), -1, dtype=int)
        for i, a in enumerate(seqs):
            mat[i, :len(a)] = a
        present = (mat >= 0).sum(0)
        ones = (mat == 1).sum(0)
        split = (present >= 2) & (ones > 0) & (ones < present)
        shared = (present >= min_stays) & (ones == present)
        if shared.any() and not split.any():
            pos = int(np.flatnonzero(shared)[0])
            return True, (f"the output is a function of the {frame} alone (no position is split between stays; "
                          f"{int(shared.sum())} shared position(s) flagged in every stay, first at {pos})")
    return False, "ok"


def probe_keeps(pool_seed: int, pids: list[str], n_hours: list[int]) -> list[int]:
    """Hidden number of rows each stay keeps in the truncation probe (deterministic in the pool seed and stay id)."""
    keeps = []
    for pid, n in zip(pids, n_hours):
        if n < 2:
            keeps.append(int(n))                                        # cannot be shortened
            continue
        u = int(hashlib.sha256(f"{CODE}|{pool_seed}|probe|{pid}".encode()).hexdigest()[:8], 16) / 2 ** 32
        frac = PROBE_KEEP[0] + (PROBE_KEEP[1] - PROBE_KEEP[0]) * u
        keeps.append(int(min(n - 1, max(min(PROBE_MIN_ROWS, n - 1), round(frac * n)))))
    return keeps


def prefix_mismatch(full: list[np.ndarray], prefix: list[np.ndarray]) -> tuple[int, int]:
    """(changed hours, compared hours) between the full-stay predictions and the truncated-stay predictions."""
    changed = total = 0
    for f, p in zip(full, prefix):
        n = len(p)
        changed += int(np.sum(np.asarray(f)[:n] != np.asarray(p)))
        total += n
    return changed, total


# ----------------------------------------------------------------------------------------------------------------
# adapter
# ----------------------------------------------------------------------------------------------------------------
class Adapter:
    discipline = CODE
    name = "PhysioNet/CinC 2019 sepsis"
    family = "Life & health"
    metric = "normalized clinical utility"
    direction = "max"
    task_type = "clinical_time_series_early_warning"

    def __init__(self, data_root: str | None = None, cache_dir: str | None = None, pool_seed: int = PROTOCOL_SEED,
                 **_: Any) -> None:
        self.root = Path(data_root or os.environ.get("SCIENCECLAW_DATA_ROOT", DATA_ROOT)) / DATASET_DIR
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir(CODE)
        self.pool_seed = int(pool_seed)
        self._memo = Memo()

    # ---------------------------------------------------------------- data
    def _source_dirs(self) -> dict[str, list[Path]]:
        rec = self.root / "reconstructed_v1" / "patients"
        raw = self.root / "raw"
        return {"A": [rec / "training_setA", raw / "training_setA"], "B": [rec / "training_setB", raw / "training_setB"]}

    def _file_index(self) -> dict[str, dict[str, Path]]:
        """hospital -> patient_id -> file (reconstructed copy preferred when both copies exist)."""
        out: dict[str, dict[str, Path]] = {}
        for hosp, dirs in self._source_dirs().items():
            d: dict[str, Path] = {}
            for dd in dirs:
                if not dd.is_dir():
                    continue
                for f in sorted(dd.glob("p*.psv")):
                    if f.name.startswith("._"):
                        continue
                    d.setdefault(f.stem, f)
            out[hosp] = d
        return out

    def available(self) -> tuple[bool, str]:
        if not self.root.is_dir():
            return False, f"dataset directory missing: {self.root}"
        idx = self._file_index()
        if len(idx.get("A", {})) < 400 or len(idx.get("B", {})) < 64:
            return False, (f"need >= 400 hospital-A and >= 64 hospital-B stays locally; found "
                           f"{len(idx.get('A', {}))} / {len(idx.get('B', {}))}")
        try:
            pools = self._pools()
        except (ValueError, OSError, KeyError) as ex:
            return False, f"cannot build pools: {type(ex).__name__}: {ex}"
        return True, (f"A={len(idx['A'])} stays, B={len(idx['B'])} stays, visible train={len(pools['train'])}")

    def _table(self) -> pd.DataFrame:
        """All local stays as one long table (patient_id, hospital, hour, 40 features, SepsisLabel); disk-cached."""
        def load() -> pd.DataFrame:
            idx = self._file_index()
            files = [(h, pid, f) for h in sorted(idx) for pid, f in sorted(idx[h].items())]
            key = ids_hash([f"{h}/{pid}/{f.stat().st_size}" for h, pid, f in files])
            cache = self.cache_dir / f"stays_{key}.parquet"
            if cache.exists():
                return pd.read_parquet(cache)
            frames = []
            for h, pid, f in files:
                df = pd.read_csv(f, sep="|")
                missing = [c for c in FEATURES + [LABEL] if c not in df.columns]
                if missing:
                    raise ValueError(f"{f}: missing columns {missing}")
                df = df[FEATURES + [LABEL]].astype(float)
                df.insert(0, "hour", np.arange(len(df), dtype=int))
                df.insert(0, "hospital", h)
                df.insert(0, "patient_id", pid)
                frames.append(df)
            tab = pd.concat(frames, ignore_index=True)
            tab[LABEL] = tab[LABEL].astype(int)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = tmp_path_for(cache)
            tab.to_parquet(tmp, index=False)
            os.replace(tmp, cache)
            return tab
        return self._memo.get("table", load)

    def _stays(self) -> dict[str, dict]:
        """patient_id -> {"hospital", "septic", "n_hours", "rows": slice}."""
        def build() -> dict[str, dict]:
            tab = self._table()
            out = {}
            pid = tab["patient_id"].to_numpy()
            starts = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
            ends = np.r_[starts[1:], len(pid)]
            lab = tab[LABEL].to_numpy()
            hosp = tab["hospital"].to_numpy()
            for s, e in zip(starts, ends):
                out[str(pid[s])] = {"hospital": str(hosp[s]), "septic": bool(lab[s:e].any()), "n_hours": int(e - s),
                                    "rows": (int(s), int(e))}
            return out
        return self._memo.get("stays", build)

    def _pools(self) -> dict[str, list[str]]:
        def build() -> dict[str, list[str]]:
            stays = self._stays()
            a_pos = sorted(p for p, s in stays.items() if s["hospital"] == "A" and s["septic"])
            a_neg = sorted(p for p, s in stays.items() if s["hospital"] == "A" and not s["septic"])
            b_pos = sorted(p for p, s in stays.items() if s["hospital"] == "B" and s["septic"])
            b_neg = sorted(p for p, s in stays.items() if s["hospital"] == "B" and not s["septic"])
            pools: dict[str, list[str]] = {}
            rest_p, rest_n = a_pos, a_neg
            for name in ("id", "val", "src", "dev"):
                npos, nneg = POOL_SIZES[name]
                tp, tn, rest_p, rest_n = stratified_take(rest_p, rest_n, npos, nneg, [CODE, self.pool_seed, name])
                pools[name + "_pos"], pools[name + "_neg"] = tp, tn
                pools[name] = tp + tn
            pools["train"] = sorted(rest_p + rest_n)
            tp, tn, _, _ = stratified_take(b_pos, b_neg, OOD_POOL[0], OOD_POOL[1], [CODE, self.pool_seed, "ood"])
            pools["ood_pos"], pools["ood_neg"], pools["ood"] = tp, tn, tp + tn
            if sum(stays[p]["septic"] for p in pools["train"]) < 10:
                raise ValueError("fewer than 10 septic stays left for visible training")
            return pools
        return self._memo.get(f"pools:{self.pool_seed}", build)

    def _rows(self, pids: list[str], with_label: bool) -> pd.DataFrame:
        tab, stays = self._table(), self._stays()
        parts = [tab.iloc[stays[p]["rows"][0]:stays[p]["rows"][1]] for p in pids]
        df = pd.concat(parts, ignore_index=True) if parts else tab.iloc[0:0]
        cols = ["patient_id", "hour"] + FEATURES + ([LABEL] if with_label else [])
        return df[cols].reset_index(drop=True).copy()

    def _measured_share_text(self) -> str:
        """Share of rows with a measured (non-NaN) value per variable in the visible training stays (hospital A);
        computed from the visible labelled rows only."""
        def build() -> str:
            tr = self._rows(self._pools()["train"], with_label=False)
            share = tr[FEATURES].notna().mean()
            parts = []
            for c in FEATURES:
                v = float(share[c])
                parts.append(f"{c} 0% (always NaN)" if v == 0.0 else f"{c} {100 * v:.1f}%" if v < 0.01
                             else f"{c} {100 * v:.0f}%")
            return "Measured (non-NaN) share of the rows of the visible training stays: " + ", ".join(parts) + "."
        return self._memo.get(f"measured_share:{self.pool_seed}", build)

    def _labels(self, pid: str) -> np.ndarray:
        tab, s = self._table(), self._stays()[pid]
        return tab[LABEL].to_numpy()[s["rows"][0]:s["rows"][1]].astype(int).copy()

    # ---------------------------------------------------------------- reference model
    @staticmethod
    def _ref_features(df: pd.DataFrame, med: pd.Series) -> np.ndarray:
        g = df.groupby("patient_id", sort=False)[REF_VITALS].ffill()          # causal LOCF within each stay
        x = pd.concat([g, df[REF_STATIC]], axis=1).fillna(med)
        return x.to_numpy(dtype=float)

    def _reference_model(self) -> dict:
        def fit() -> dict:
            from sklearn.linear_model import LogisticRegression

            pools = self._pools()
            tr = self._rows(pools["train"], with_label=True)
            med = tr[REF_VITALS + REF_STATIC].median()
            x = self._ref_features(tr, med)
            mu, sd = x.mean(0), x.std(0) + 1e-9
            clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
            clf.fit((x - mu) / sd, tr[LABEL].to_numpy())
            p = clf.predict_proba((x - mu) / sd)[:, 1]
            pids = tr["patient_id"].to_numpy()
            labs = tr[LABEL].to_numpy()
            starts = np.flatnonzero(np.r_[True, pids[1:] != pids[:-1]])
            ends = np.r_[starts[1:], len(pids)]
            best_thr, best_nu = 0.5, -np.inf
            for thr in np.unique(np.quantile(p, np.linspace(0.5, 0.995, 60))):
                o = b = i = 0.0
                for s, e in zip(starts, ends):
                    uo, ub, ui = utility_triplet(labs[s:e], (p[s:e] >= thr).astype(int))
                    o, b, i = o + uo, b + ub, i + ui
                nu = normalized_utility(o, b, i)
                if nu is not None and nu > best_nu:
                    best_thr, best_nu = float(thr), nu
            return {"clf": clf, "mu": mu, "sd": sd, "med": med, "thr": best_thr, "train_nu": float(best_nu)}
        return self._memo.get(f"refmodel:{self.pool_seed}", fit)

    def reference_predictions(self, pids: list[str]) -> list[np.ndarray]:
        m = self._reference_model()
        df = self._rows(pids, with_label=False)
        x = (self._ref_features(df, m["med"]) - m["mu"]) / m["sd"]
        p = m["clf"].predict_proba(x)[:, 1]
        out, pos = [], 0
        for pid in pids:
            n = self._stays()[pid]["n_hours"]
            out.append((p[pos:pos + n] >= m["thr"]).astype(int))
            pos += n
        return out

    # ---------------------------------------------------------------- scoring
    def _score(self, pids: list[str], preds: list[np.ndarray]) -> dict:
        obs, best, inact = [], [], []
        for pid, pr in zip(pids, preds):
            o, b, i = utility_triplet(self._labels(pid), pr)
            obs.append(o); best.append(b); inact.append(i)
        nu = normalized_utility(sum(obs), sum(best), sum(inact))
        return {"nu": nu, "observed": obs, "best": best, "inaction": inact}

    @staticmethod
    def _coerce(y: Any, n_hours: list[int]) -> tuple[list[np.ndarray] | None, str]:
        arrs = as_list_of_arrays(y, len(n_hours))
        if arrs is None:
            return None, f"y must be a list of {len(n_hours)} arrays (one per stay)"
        out = []
        for i, (a, n) in enumerate(zip(arrs, n_hours)):
            a = np.asarray(a).reshape(-1) if np.asarray(a).ndim <= 1 else None
            if a is None or a.shape[0] != n:
                return None, f"stay {i}: expected 1-D array of length {n}"
            try:
                af = a.astype(float)
            except (TypeError, ValueError):
                return None, f"stay {i}: non-numeric values"
            if not np.all(np.isfinite(af)) or not np.all(np.isin(af, (0.0, 1.0))):
                return None, f"stay {i}: values must be 0 or 1"
            out.append(af.astype(int))
        return out, "ok"

    def pooled_metric(self, per_episode: list[dict]) -> float | None:
        """Normalized utility over all stays of the given episodes (sums of per-stay utilities)."""
        pays = extract_payloads(per_episode, "sepsis_utility")
        if not pays:
            return None
        o = sum(sum(p["observed"]) for p in pays)
        b = sum(sum(p["best"]) for p in pays)
        i = sum(sum(p["inaction"]) for p in pays)
        return normalized_utility(o, b, i)

    # ---------------------------------------------------------------- episodes
    def build_episodes(self, split: str, n: int, seed: int, items_per_episode: int = 16) -> list[Episode]:
        if split not in ("src", "val", "id", "ood", "rep"):
            raise ValueError(f"unknown split {split!r}")
        ok, why = self.available()
        if not ok:
            raise RuntimeError(f"{CODE} unavailable: {why}")
        base = "src" if split == "rep" else split
        pools = self._pools()
        k_req = int(items_per_episode)
        pos, neg = pools[base + "_pos"], pools[base + "_neg"]
        k = effective_items(split, len(pos) + len(neg), n, k_req)
        k_pos = max(1, int(round(k * 0.125)))
        if split not in ("src", "rep"):                  # held-out splits never reuse stays
            k_pos = min(k_pos, len(pos) // n)
            if k_pos < 1:
                raise ValueError(f"{split}: {len(pos)} septic stays cannot give every one of {n} episodes a septic stay")
            k = k_pos + min(k - k_pos, len(neg) // n)
        strata = {"pos": (pos, k_pos), "neg": (neg, k - k_pos)}
        groups, reused = compose_episodes([], n, k, [CODE, self.pool_seed, base, seed, k_req], strata=strata)
        eps = [self._episode(split, base, seed, j, pids, reused) for j, pids in enumerate(groups)]
        for e in eps:
            e.lineage["items_requested"] = k_req
        return eps

    def _episode(self, split: str, base: str, seed: int, j: int, pids: list[str], reused: bool) -> Episode:
        pools = self._pools()
        k = len(pids)
        n_hours = [self._stays()[p]["n_hours"] for p in pids]
        hospital = "B" if base == "ood" else "A"
        dev_pids = list(pools["dev"])
        dev_hours = [self._stays()[p]["n_hours"] for p in dev_pids]
        train_pids = list(pools["train"])
        n_septic_train = sum(self._stays()[p]["septic"] for p in train_pids)
        memo = Memo()

        def t_load_train(inputs: dict, config: dict) -> dict:
            df = memo.get("train", lambda: self._rows(train_pids, with_label=True))
            return {"table": df.copy(), "n_stays": len(train_pids)}

        def t_load_eval(inputs: dict, config: dict) -> dict:
            df = memo.get("eval", lambda: self._rows(pids, with_label=False))
            return {"table": df.copy(), "patient_ids": list(pids), "n_hours": list(n_hours)}

        def t_load_dev(inputs: dict, config: dict) -> dict:
            df = memo.get("dev", lambda: self._rows(dev_pids, with_label=False))
            return {"table": df.copy(), "patient_ids": list(dev_pids), "n_hours": list(dev_hours)}

        def t_score_dev(inputs: dict, config: dict) -> dict:
            preds, msg = self._coerce(inputs.get("predictions"), dev_hours)
            if preds is None:
                raise ValueError(f"score_dev: {msg}")
            s = self._score(dev_pids, preds)
            return {"normalized_utility": float(s["nu"]) if s["nu"] is not None else float("nan"),
                    "n_stays": len(dev_pids)}

        table_desc = ("long table: patient_id (str), hour (0-based row index within the stay), 40 clinical variables "
                      "(NaN = not measured)")
        tools = [
            ToolSpec("load_train", "Visible labelled training stays from hospital A (hourly rows incl. SepsisLabel).",
                     {}, {"table": PortSchema("table", ("rows", 43), None, None, table_desc + " + SepsisLabel (0/1)"),
                          "n_stays": PortSchema("number", None, "1", "int", "number of training stays")},
                     t_load_train),
            ToolSpec("load_eval_inputs", "Hourly rows of the evaluation stays of this episode (no labels), in item order; "
                                          "rows of a stay are in hour order and the prediction for hour t may use only rows "
                                          "0..t of that stay.",
                     {}, {"table": PortSchema("table", ("rows", 42), None, None, table_desc),
                          "patient_ids": PortSchema("list", (k,), None, "str", "stay ids in item order"),
                          "n_hours": PortSchema("list", (k,), "h", "int", "number of hourly rows per stay")},
                     t_load_eval),
            ToolSpec("load_dev_inputs", "Hourly rows of the visible dev stays (hospital A, labels withheld); the same "
                                        "rule applies as for the evaluation stays: the prediction for hour t may use only "
                                        "rows 0..t of that stay.",
                     {}, {"table": PortSchema("table", ("rows", 42), None, None, table_desc),
                          "patient_ids": PortSchema("list", (len(dev_pids),), None, "str", "dev stay ids"),
                          "n_hours": PortSchema("list", (len(dev_pids),), "h", "int", "rows per dev stay")},
                     t_load_dev),
            ToolSpec("score_dev", "Normalized clinical utility of binary hourly predictions for the dev stays "
                                  "(same order as load_dev_inputs); dev predictions are produced under the same causal "
                                  "rule as the evaluation predictions.",
                     {"predictions": PortSchema("list", (len(dev_pids),), "1", "int",
                                                "one 0/1 array per dev stay, length n_hours[i]")},
                     {"normalized_utility": PortSchema("number", None, "1", "float", "dev normalized utility"),
                      "n_stays": PortSchema("number", None, "1", "int", "")},
                     t_score_dev),
        ]
        var_lines = ", ".join(f"{c} [{FEATURE_UNITS[c]}]" for c in FEATURES)
        objective = (
            f"Early prediction of sepsis (Sepsis-3) in the ICU, PhysioNet/CinC Challenge 2019 data.\n"
            f"Evaluation items: {k} complete ICU stays from hospital {hospital} of the challenge training data "
            f"(tool load_eval_inputs, item order = patient_ids). Each stay is a sequence of hourly rows with 40 "
            f"variables (units as listed in the challenge documentation): {var_lines}. Missing measurements are NaN. "
            f"{self._measured_share_text()} "
            f"Visible labelled training stays (tool load_train) "
            f"come from hospital A; their SepsisLabel is 1 from 6 h before the clinical sepsis onset time t_sepsis "
            f"onward and 0 otherwise (always 0 for non-septic stays). Composition: the {len(train_pids)} visible "
            f"training stays contain {n_septic_train} septic stays.\n"
            f"Required output y: a list of {k} one-dimensional integer arrays in the order of patient_ids; array i "
            f"has exactly n_hours[i] entries, each 0 or 1 (1 = sepsis predicted at that hour).\n"
            f"Interface rule: the prediction for hour t of a stay may use only rows 0..t of that stay (plus the "
            f"training data), as in the challenge's sequential evaluation. Later rows, statistics over the whole "
            f"stay and the length of the stay (n_hours[i] only sets the length of output array i) must not enter "
            f"the prediction for hour t.\n"
            f"Score: normalized clinical utility of the 2019 challenge summed over the {k} stays: per hour of a "
            f"septic stay a positive prediction earns up to +1 (maximal 6 h before t_sepsis, 0 at 12 h before and at "
            f"3 h after; -0.05 when earlier than 12 h before), a missing positive between 6 h before and 3 h after "
            f"t_sepsis costs up to -2, hours later than 3 h after t_sepsis are not scored, every positive hour of a "
            f"non-septic stay costs -0.05; normalized so that 1 = optimal predictions and 0 = never predicting "
            f"sepsis.\n"
            + scilib.describe("sepsis"))
        constraints = [
            ConstraintSpec("output_structure", f"y is a list of {k} arrays with lengths n_hours in item order",
                           lambda y, tr: (lambda r: (r[0] is not None, r[1]))(self._coerce_structure(y, n_hours))),
            ConstraintSpec("binary_values", "every entry is 0 or 1 (finite)",
                           lambda y, tr: (lambda r: (r[0] is not None, r[1]))(self._coerce(y, n_hours))),
            ConstraintSpec("not_positional_only",
                           "y is not a function of the position in the stay alone (hour index or distance to the "
                           "end of the record)",
                           lambda y, tr: (lambda r: (not r[0], r[1]))(self._positional_check(y, n_hours))),
            ConstraintSpec("causal_prefix",
                           "predictions for hours 0..t do not change when the stay is cut after hour t",
                           lambda y, tr: self._probe_verdict(tr, "causal_prefix"), visible=False),
            unit_constraint("1"),
        ]
        reference_cache = Memo()
        keeps = probe_keeps(self.pool_seed, pids, n_hours)

        def probe_eval_inputs(inputs: dict, config: dict) -> dict:
            full = memo.get("eval", lambda: self._rows(pids, with_label=False))
            parts, pos = [], 0
            for n, kp in zip(n_hours, keeps):
                parts.append(full.iloc[pos:pos + kp])
                pos += n
            return {"table": pd.concat(parts, ignore_index=True), "patient_ids": list(pids), "n_hours": list(keeps)}

        def probes(y: Any) -> list[Probe]:
            full_preds, _ = self._coerce(y, n_hours)
            if full_preds is None:
                return []
            base = next(t for t in tools if t.name == "load_eval_inputs")
            cut_tool = ToolSpec(base.name, base.description, dict(base.inputs), dict(base.outputs), probe_eval_inputs,
                                base.config_doc)

            def check(y_p: Any) -> tuple[bool, str]:
                got, msg = self._coerce(y_p, keeps)
                if got is None:
                    return False, f"prefix run: {msg}"
                changed, total = prefix_mismatch(full_preds, got)
                frac = changed / max(total, 1)
                return frac <= CAUSAL_TOL, (f"{changed}/{total} prefix hours changed ({100 * frac:.1f} %; tolerance "
                                            f"{100 * CAUSAL_TOL:.1f} %) when the stays were cut at hidden hours")

            swapped = [cut_tool if t.name == "load_eval_inputs" else t for t in tools]
            structure = ConstraintSpec("output_structure", "y has one array per stay of the cut length",
                                       lambda yy, tr: (lambda r: (r[0] is not None, r[1]))(self._coerce_structure(yy, keeps)))
            return [Probe("causal_prefix", {"tools": swapped, "constraints": [structure]}, check)]

        def evaluate(y: Any, trace: Any) -> EvalResult:
            ref_preds = reference_cache.get("ref", lambda: self.reference_predictions(pids))
            ref = self._score(pids, ref_preds)
            preds, msg = self._coerce(y, n_hours)
            valid = preds is not None
            if not valid:
                preds = [np.zeros(h, dtype=int) for h in n_hours]      # no-skill fallback for pooled metric
            s = self._score(pids, preds)
            payload = {"kind": "sepsis_utility", "patient_ids": list(pids), "observed": s["observed"],
                       "best": s["best"], "inaction": s["inaction"], "fallback": not valid}
            n_pos_pred = int(sum(int(p.sum()) for p in preds))
            metrics = {"normalized_utility": s["nu"], "reference_normalized_utility": ref["nu"],
                       "positive_hours_predicted": n_pos_pred}
            return make_result(s["nu"], ref["nu"], "max", ACCEPT_MARGIN, metrics,
                               {"pooled_payload": payload, "parse": msg,
                                "reference_desc": "logistic regression on LOCF vitals + static, utility-tuned threshold"},
                               valid, floor=ACCEPT_FLOOR)

        n_septic = sum(self._stays()[p]["septic"] for p in pids)
        lineage = {"dataset": "PhysioNet/CinC Challenge 2019 (public training sets A and B)", "version": "1.0.0",
                   "source_url": "https://physionet.org/content/challenge-2019/1.0.0/",
                   "license": "ODC Open Database License (ODbL) v1.0 (raw/LICENSE.txt)",
                   "item_kind": "ICU stay (patient PSV)", "item_ids": list(pids),
                   "item_pool": "ood" if base == "ood" else "iid", "hospital": hospital,
                   "ood_kind": "cross_hospital_within_dataset" if base == "ood" else None,
                   "n_septic_items": int(n_septic), "pool_seed": self.pool_seed, "episode_seed": int(seed),
                   "episode_index": j, "reused_items": bool(reused),
                   "visible_train_ids_hash": ids_hash(train_pids), "n_visible_train": len(train_pids),
                   "dev_ids_hash": ids_hash(dev_pids), "rebuilt_split": True, "historical_ids_recovered": False}
        return Episode(
            id=f"{CODE}-{split}-s{seed}-e{j:02d}", discipline=CODE, family=self.family, split=split,
            task_type=self.task_type, objective=objective,
            required_output=PortSchema("list", (k,), "1", "int",
                                       "one 0/1 integer array per stay (length n_hours[i]), item order"),
            tools=tools, constraints=constraints,
            budget=default_budget(k, max_node_s=300, max_wall_s=1800),
            lineage=lineage,
            acceptance=(f"accepted iff normalized utility > max(reference, {ACCEPT_FLOOR:g}) + {ACCEPT_MARGIN} "
                        f"(reference = logistic-regression baseline on the same stays)"),
            tolerance={"rtol": 0.0, "atol": 0.0}, tags=["health", "sepsis", "icu", "time-series", "clinical-utility"],
            metric=self.metric, direction=self.direction, n_items=k, _evaluate=evaluate, _dev_evaluate=None,
            _probes=probes)

    @staticmethod
    def _positional_check(y: Any, n_hours: list[int]) -> tuple[bool, str]:
        """(is positional-only, message); malformed outputs are left to the structure constraints."""
        arrs = as_list_of_arrays(y, len(n_hours))
        if arrs is None or any(a.ndim != 1 or a.shape[0] != n for a, n in zip(arrs, n_hours)):
            return False, "n/a (malformed output)"
        try:
            return positional_only([np.nan_to_num(np.asarray(a, dtype=float)).astype(int) for a in arrs])
        except (TypeError, ValueError):
            return False, "n/a (non-numeric output)"

    @staticmethod
    def _probe_verdict(trace: Any, name: str) -> tuple[bool, str]:
        """Hidden-constraint verdict from ``trace.probes`` (set by ``Episode.run_probes``); passes when not probed."""
        rec = (getattr(trace, "probes", None) or {}).get(name)
        if not rec:
            return True, "not probed"
        return bool(rec.get("ok")), str(rec.get("msg", ""))

    @staticmethod
    def _coerce_structure(y: Any, n_hours: list[int]) -> tuple[Any, str]:
        arrs = as_list_of_arrays(y, len(n_hours))
        if arrs is None:
            return None, f"y must be a list of {len(n_hours)} arrays"
        for i, (a, n) in enumerate(zip(arrs, n_hours)):
            if a.ndim != 1 or a.shape[0] != n:
                return None, f"stay {i}: shape {a.shape}, expected ({n},)"
        return arrs, "ok"

    # ---------------------------------------------------------------- introspection (tests / task card)
    def describe_pools(self) -> dict:
        pools, stays = self._pools(), self._stays()
        return {k: {"n": len(v), "septic": int(sum(stays[p]["septic"] for p in v))}
                for k, v in pools.items() if not k.endswith(("_pos", "_neg"))}


if __name__ == "__main__":      # pragma: no cover - manual inspection helper
    a = Adapter()
    print(a.available())
    print(json.dumps(a.describe_pools(), indent=1))
