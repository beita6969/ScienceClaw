"""Tools for hourly ICU sepsis early-warning on PhysioNet/CinC-2019-style stays (normalized clinical utility).

A ``table`` is a long pandas DataFrame with one row per patient-hour: ``patient_id``, ``hour`` (0-based row index within
the stay) and the 40 challenge variables (``FEATURES``: 34 time-varying vitals/labs, then Age, Gender, Unit1, Unit2,
HospAdmTime, ICULOS); a training table also has ``SepsisLabel``. Rows of one stay are contiguous; stays are numbered in
order of first appearance and every per-stay list below follows that order. Every feature and prediction for hour t
is computed from rows 0..t of the same stay (and, for the model, the training rows) only.

utility_of_stay(labels, pred) -> float             official 2019 utility of one stay (dt_early -12, dt_optimal -6, dt_late 3,
                                                   max_u_tp 1, min_u_fn -2, u_fp -0.05, u_tn 0)
normalized_utility(labels, preds) -> float         (U_obs - U_inaction) / (U_best - U_inaction) summed over stays;
                                                   ``labels``/``preds``: lists with one array per stay (NaN if U_best == U_inaction)
row_gains(labels) -> np.ndarray                    per hour: utility gained by predicting 1 instead of 0 in this stay;
                                                   utility(pred) = utility(all zeros) + sum(pred * gains)
stay_targets(table, mode) -> np.ndarray            per-row training target: "label" = SepsisLabel, "window" = 1 on hours
                                                   t_sepsis-12 .. t_sepsis+3 of septic stays, "gain" = row_gains
build_features(table, windows=(6,12,24), measurement=True) -> DataFrame
                                                   causal per-hour features aligned with the rows of ``table``: last observation
                                                   carried forward, rolling mean/min/max/slope over 6/12/24 h and expanding
                                                   statistics of the vitals, 12 h changes and running extremes of common labs,
                                                   clinical composites (shock index, SIRS/qSOFA-type counts), static variables;
                                                   with ``measurement=True`` also hours since each variable was last measured and
                                                   measurement counts per window (``X_since``, ``n_meas{w}``)
causal_smooth(scores, patient_ids, window) -> np.ndarray   trailing mean of the scores over the last ``window`` hours of a stay
hold_alarms(pred, patient_ids, hours) -> np.ndarray        keeps a positive prediction on for ``hours`` further hours of the same stay
stay_folds(patient_ids, septic, n_folds, seed) -> dict     patient_id -> fold; septic and non-septic stays are spread evenly
SepsisModel(targets=("label","gain"), n_folds=5, smooth=3, hold=0, threshold=None, prevalence=0.073, seed=0, params=None,
            drop=("Unit1","Unit2"), measurement=False, linear_weight=0.25, linear_c=0.1).fit(train)
                                                   shallow LightGBM models (one fold ensemble per training target) and a
                                                   class-balanced L2 logistic regression (share ``linear_weight`` of the score) on
                                                   ``build_features`` without Unit1/Unit2; ``.oof_scores_`` are the out-of-fold
                                                   scores (0..1), ``.threshold_`` the cut-off maximising the normalized utility
                                                   of the out-of-fold alarms, with the septic stays weighted so that they make up
                                                   the share ``prevalence`` of the stays (0.073 = septic share of the public
                                                   challenge data; None: no weighting, i.e. the septic share of the training
                                                   table; the best cut-off falls as the share rises),
                                                   ``.oof_utility_`` the unweighted normalized utility of those alarms on the
                                                   training stays and ``.oof_alarm_rate_`` their share of the hours;
                                                   ``.score(table)`` -> score per row; ``.predict(table)`` -> list of 0/1 int
                                                   arrays, one per stay
fit_predict(train, tables, **model_kwargs) -> (preds, info)
                                                   ``SepsisModel(**model_kwargs).fit(train)``, then ``.predict`` for every table in
                                                   the list ``tables``; ``preds[i]`` = list of 0/1 arrays for ``tables[i]``; ``info``
                                                   has threshold, oof_utility, oof_alarm_rate, n_positive_hours. Example:
                                                   ``(dev_pred, eval_pred), info = fit_predict(train, [dev_table, eval_table])``

All functions are deterministic for fixed arguments; ``fit_predict`` with the defaults trains 10 small LightGBM models and 5
logistic regressions on CPU (one thread), about 15 s on a training table of 500 stays.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["FEATURES", "utility_of_stay", "normalized_utility", "row_gains", "stay_targets", "build_features",
           "causal_smooth", "hold_alarms", "stay_folds", "SepsisModel", "fit_predict"]

FEATURES = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "HCO3", "FiO2", "pH",
            "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
            "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct",
            "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2", "HospAdmTime",
            "ICULOS"]
VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
STATIC = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
DYNAMIC = [c for c in FEATURES if c not in STATIC]
LABS = [c for c in DYNAMIC if c not in VITALS]
WINDOWS = (6, 12, 24)

# official utility (evaluate_sepsis_score.py, PhysioNet/CinC 2019)
DT_EARLY, DT_OPTIMAL, DT_LATE = -12, -6, 3
MAX_U_TP, MIN_U_FN, U_FP, U_TN = 1.0, -2.0, -0.05, 0.0


# ----------------------------------------------------------------------------------------------------------------
# official utility
# ----------------------------------------------------------------------------------------------------------------
def _utility_rows(labels) -> tuple[np.ndarray, np.ndarray]:
    """(u_if_predicted_1, u_if_predicted_0) per hour of one stay, from the official piecewise-linear utility."""
    labels = np.asarray(labels).astype(int).reshape(-1)
    n = len(labels)
    if not labels.any():
        return np.full(n, U_FP), np.full(n, U_TN)
    ts = float(np.argmax(labels) - DT_OPTIMAL)
    m1 = MAX_U_TP / (DT_OPTIMAL - DT_EARLY)
    b1 = -m1 * DT_EARLY
    m2 = -MAX_U_TP / (DT_LATE - DT_OPTIMAL)
    b2 = -m2 * DT_LATE
    m3 = MIN_U_FN / (DT_LATE - DT_OPTIMAL)
    b3 = -m3 * DT_OPTIMAL
    d = np.arange(n, dtype=float) - ts
    early = d <= DT_OPTIMAL
    late = (~early) & (d <= DT_LATE)
    u1 = np.where(early, np.maximum(m1 * d + b1, U_FP), np.where(late, m2 * d + b2, 0.0))
    u0 = np.where(early, 0.0, np.where(late, m3 * d + b3, 0.0))
    return u1, u0


def _best_predictions(labels) -> np.ndarray:
    labels = np.asarray(labels).astype(int).reshape(-1)
    best = np.zeros(len(labels), dtype=int)
    if labels.any():
        ts = int(np.argmax(labels)) - DT_OPTIMAL
        best[max(0, ts + DT_EARLY): min(ts + DT_LATE + 1, len(labels))] = 1
    return best


def utility_of_stay(labels, pred) -> float:
    """Official 2019 challenge utility of the 0/1 hourly predictions ``pred`` for one stay with hourly ``SepsisLabel``
    ``labels`` (t_sepsis = first label hour + 6; a positive earns up to +1 from 12 h before to 3 h after t_sepsis,
    -0.05 when earlier or in a non-septic stay; a missing positive costs up to -2 from 6 h before to 3 h after)."""
    u1, u0 = _utility_rows(labels)
    pred = np.asarray(pred).astype(bool).reshape(-1)
    if len(pred) != len(u1):
        raise ValueError("Numbers of predictions and labels must be the same.")
    return float(np.sum(np.where(pred, u1, u0)))


def row_gains(labels) -> np.ndarray:
    """Per hour of a stay, the utility gained by predicting 1 instead of 0 (negative: a false alarm costs 0.05; zero
    from 3 h after t_sepsis on). ``utility_of_stay(labels, pred) == utility_of_stay(labels, zeros) + sum(pred * gains)``."""
    u1, u0 = _utility_rows(labels)
    return u1 - u0


def normalized_utility(labels, preds) -> float:
    """Normalized utility of the challenge: (U_observed - U_inaction) / (U_best - U_inaction) with the utilities
    summed over stays. ``labels``/``preds``: sequences with one 1-D array per stay. NaN when no stay is septic."""
    if len(labels) != len(preds):
        raise ValueError("labels and preds must have one array per stay")
    obs = best = inact = 0.0
    for lab, pr in zip(labels, preds):
        u1, u0 = _utility_rows(lab)
        pr = np.asarray(pr).astype(bool).reshape(-1)
        if len(pr) != len(u1):
            raise ValueError("Numbers of predictions and labels must be the same.")
        bp = _best_predictions(lab).astype(bool)
        obs += float(np.sum(np.where(pr, u1, u0)))
        best += float(np.sum(np.where(bp, u1, u0)))
        inact += float(np.sum(u0))
    den = best - inact
    return float((obs - inact) / den) if den != 0 else float("nan")


# ----------------------------------------------------------------------------------------------------------------
# table plumbing
# ----------------------------------------------------------------------------------------------------------------
def _layout(table: pd.DataFrame):
    """(codes in first-appearance order, order that makes stays contiguous and hour-sorted, group start of each sorted
    row, uniques). Rows already contiguous and sorted give the identity order."""
    if "patient_id" not in table.columns:
        raise ValueError("table needs a 'patient_id' column")
    codes, uniques = pd.factorize(table["patient_id"].to_numpy(), sort=False)
    if "hour" in table.columns:
        order = np.lexsort((table["hour"].to_numpy(dtype=float), codes))
    else:
        order = np.argsort(codes, kind="stable")
    sc = codes[order]
    starts = np.flatnonzero(np.r_[True, sc[1:] != sc[:-1]])
    lens = np.diff(np.r_[starts, len(sc)])
    gstart = np.repeat(starts, lens)
    return codes, order, sc, gstart, uniques


def _check_columns(table: pd.DataFrame, need) -> None:
    missing = [c for c in need if c not in table.columns]
    if missing:
        raise ValueError(f"table lacks columns {missing}")


def _split(values: np.ndarray, patient_ids) -> list[np.ndarray]:
    """Split a per-row array into one array per stay (order of first appearance)."""
    codes, _ = pd.factorize(np.asarray(patient_ids), sort=False)
    values = np.asarray(values)
    n = int(codes.max()) + 1 if len(codes) else 0
    idx = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[idx], np.arange(n + 1))
    return [values[idx[bounds[i]:bounds[i + 1]]] for i in range(n)]


def stay_targets(table: pd.DataFrame, mode: str = "label") -> np.ndarray:
    """Per-row training target of a table with ``SepsisLabel``: ``"label"`` = SepsisLabel (1 from 6 h before
    t_sepsis), ``"window"`` = 1 on the hours t_sepsis-12 .. t_sepsis+3 of septic stays (the hours the utility rewards),
    ``"gain"`` = ``row_gains`` of the stay (utility gained by predicting 1 at that hour)."""
    _check_columns(table, ["SepsisLabel"])
    if mode not in ("label", "window", "gain"):
        raise ValueError("mode must be 'label', 'window' or 'gain'")
    lab = table["SepsisLabel"].to_numpy(dtype=float)
    if mode == "label":
        return lab.astype(int)
    out = np.zeros(len(table), dtype=float if mode == "gain" else int)
    for rows, y in zip(_split(np.arange(len(table)), table["patient_id"].to_numpy()),
                       _split(lab, table["patient_id"].to_numpy())):
        if mode == "gain":
            out[rows] = row_gains(y)
        elif y.any():
            a = int(np.argmax(y))                      # first label hour = t_sepsis - 6
            out[rows[max(0, a - 6): a + 10]] = 1        # t_sepsis-12 .. t_sepsis+3
    return out


# ----------------------------------------------------------------------------------------------------------------
# causal features
# ----------------------------------------------------------------------------------------------------------------
def _locf(x: np.ndarray, gstart: np.ndarray):
    """Forward fill within stays. x: (N, k) in stay-sorted order. Returns (filled, hours since the last observation
    (NaN before the first one))."""
    n = x.shape[0]
    pos = np.arange(n)[:, None]
    seen = np.where(np.isfinite(x), pos, -1)
    last = np.maximum.accumulate(seen, axis=0)
    ok = last >= gstart[:, None]
    filled = np.where(ok, np.take_along_axis(x, np.clip(last, 0, None), axis=0), np.nan)
    since = np.where(ok, pos - last, np.nan)
    return filled, since


def _window_stats(v: np.ndarray, sc: np.ndarray, w: int, chunk: int = 60000):
    """Trailing-window (last w rows of the same stay, current row included) mean/min/max/slope of a filled 1-D series."""
    n = len(v)
    vp = np.r_[np.full(w - 1, np.nan), v]
    cp = np.r_[np.full(w - 1, -1), sc]
    out = np.full((4, n), np.nan)
    offs = np.arange(-(w - 1), 1, dtype=float)
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        vw = np.lib.stride_tricks.sliding_window_view(vp, w)[i0:i1]
        cw = np.lib.stride_tricks.sliding_window_view(cp, w)[i0:i1]
        ok = (cw == sc[i0:i1, None]) & np.isfinite(vw)
        cnt = ok.sum(1)
        vz = np.where(ok, vw, 0.0)
        s1 = vz.sum(1)
        has = cnt > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(has, s1 / np.maximum(cnt, 1), np.nan)
            mn = np.where(has, np.where(ok, vw, np.inf).min(1), np.nan)
            mx = np.where(has, np.where(ok, vw, -np.inf).max(1), np.nan)
            sx = (ok * offs).sum(1)
            sxx = (ok * offs ** 2).sum(1)
            sxy = (vz * offs).sum(1)
            den = cnt * sxx - sx ** 2
            slope = np.where((cnt >= 2) & (den > 0), (cnt * sxy - sx * s1) / np.where(den > 0, den, 1.0), np.nan)
        out[:, i0:i1] = np.vstack([mean, mn, mx, slope])
    return out


def _expanding(v: np.ndarray, sc: np.ndarray):
    """Expanding (since the first row of the stay) mean/min/max of a filled series."""
    s = pd.Series(v)
    g = s.groupby(sc)
    return g.expanding().mean().to_numpy(), g.cummin().to_numpy(), g.cummax().to_numpy()


def build_features(table: pd.DataFrame, windows=WINDOWS, measurement: bool = True) -> pd.DataFrame:
    """Causal per-hour features, one row per row of ``table`` (same index). For every time-varying variable ``X``:
    ``X`` (last observation carried forward; NaN before the first measurement), ``X_since`` (hours since it was last
    measured). For the seven vitals additionally ``X_{mean,min,max,slope}{w}``
    over the last w hours (w in ``windows``) and expanding ``X_emean/X_emin/X_emax``; for a few labs the change over
    the last 12 h and the running maximum. Also ``n_meas{w}`` (measurements in the last w hours), ``shock_index``
    (HR/SBP), ``pulse_pressure``, ``sirs`` and ``qsofa`` style counts, an organ-dysfunction count, ``bun_cr``, the static
    variables and ``ICULOS``. ``measurement=False`` leaves out the columns that describe when things were measured
    (``X_since`` and ``n_meas{w}``). Rows of a stay only see the same stay's rows 0..t."""
    _check_columns(table, ["patient_id"] + [c for c in FEATURES if c != "ICULOS"])
    codes, order, sc, gstart, _ = _layout(table)
    n = len(table)
    raw = table[DYNAMIC].to_numpy(dtype=float)[order]
    filled, since = _locf(raw, gstart)
    obs = np.isfinite(raw)
    cols: dict[str, np.ndarray] = {}
    for j, c in enumerate(DYNAMIC):
        cols[c] = filled[:, j]
        if measurement:
            cols[c + "_since"] = since[:, j]
    for c in VITALS:
        v = filled[:, DYNAMIC.index(c)]
        for w in windows:
            mean, mn, mx, slope = _window_stats(v, sc, w)
            cols[f"{c}_mean{w}"], cols[f"{c}_min{w}"], cols[f"{c}_max{w}"], cols[f"{c}_slope{w}"] = mean, mn, mx, slope
        cols[c + "_emean"], cols[c + "_emin"], cols[c + "_emax"] = _expanding(v, sc)
    for c in ("Lactate", "Creatinine", "Bilirubin_total", "Platelets", "WBC", "BUN", "Glucose", "pH", "HCO3",
              "BaseExcess", "Hct", "Hgb", "Potassium"):
        v = filled[:, DYNAMIC.index(c)]
        pos = np.arange(n)
        lag = np.where(pos - 12 >= gstart, np.take(v, np.clip(pos - 12, 0, None)), np.nan)
        cols[c + "_d12"] = v - lag
        cols[c + "_emax"] = _expanding(v, sc)[2]
        cols[c + "_emin"] = _expanding(v, sc)[1]
    for w in (windows if measurement else ()):          # measurement intensity in the last w hours (raw observations)
        cs = np.cumsum(obs.sum(1))
        prev = np.where(np.arange(n) - w >= gstart, cs[np.clip(np.arange(n) - w, 0, None)], np.r_[0, cs][gstart])
        cols[f"n_meas{w}"] = (cs - prev).astype(float)
    f = {c: filled[:, DYNAMIC.index(c)] for c in DYNAMIC}
    with np.errstate(invalid="ignore", divide="ignore"):
        cols["shock_index"] = f["HR"] / f["SBP"]
        cols["pulse_pressure"] = f["SBP"] - f["DBP"]
        cols["bun_cr"] = f["BUN"] / f["Creatinine"]
        cols["spo2_fio2"] = f["O2Sat"] / f["FiO2"]
        b = lambda cond: np.where(np.isfinite(cond[1]), cond[0], 0)                       # NaN -> criterion not met
        sirs = (b(((f["Temp"] > 38) | (f["Temp"] < 36), f["Temp"])) + b((f["HR"] > 90, f["HR"])) +
                b((f["Resp"] > 20, f["Resp"])) + b(((f["WBC"] > 12) | (f["WBC"] < 4), f["WBC"])))
        qsofa = b((f["SBP"] <= 100, f["SBP"])) + b((f["Resp"] >= 22, f["Resp"]))
        organ = (b((f["Platelets"] < 150, f["Platelets"])) + b((f["Bilirubin_total"] >= 1.2, f["Bilirubin_total"])) +
                 b((f["Creatinine"] >= 1.2, f["Creatinine"])) + b((f["MAP"] < 70, f["MAP"])) +
                 b((f["Lactate"] >= 2, f["Lactate"])) + b((f["O2Sat"] < 92, f["O2Sat"])))
    cols["sirs"], cols["qsofa"], cols["organ_count"] = sirs.astype(float), qsofa.astype(float), organ.astype(float)
    for c in STATIC:
        cols[c] = table[c].to_numpy(dtype=float)[order] if c in table.columns else np.arange(n) - gstart + 1.0
    feats = pd.DataFrame(cols)
    inv = np.empty(n, dtype=int)
    inv[order] = np.arange(n)
    out = feats.iloc[inv].reset_index(drop=True)
    out.index = table.index
    return out


# ----------------------------------------------------------------------------------------------------------------
# post-processing
# ----------------------------------------------------------------------------------------------------------------
def causal_smooth(scores, patient_ids, window: int = 3) -> np.ndarray:
    """Trailing mean of per-hour scores over the last ``window`` rows of the same stay (current row included)."""
    scores = np.asarray(scores, dtype=float)
    if window <= 1:
        return scores.copy()
    _, order, sc, _, _ = _layout(pd.DataFrame({"patient_id": np.asarray(patient_ids)}))
    v = scores[order]
    m = _window_stats(v, sc, int(window))[0]
    out = np.empty_like(scores)
    out[order] = m
    return out


def hold_alarms(pred, patient_ids, hours: int = 6) -> np.ndarray:
    """A positive prediction stays on for ``hours`` further rows of the same stay (0 = unchanged)."""
    pred = np.asarray(pred).astype(int)
    if hours <= 0:
        return pred.copy()
    _, order, sc, _, _ = _layout(pd.DataFrame({"patient_id": np.asarray(patient_ids)}))
    v = pred[order].astype(float)
    m = _window_stats(v, sc, int(hours) + 1)[2]
    out = np.empty_like(pred)
    out[order] = (m > 0).astype(int)
    return out


# ----------------------------------------------------------------------------------------------------------------
# validation folds, model
# ----------------------------------------------------------------------------------------------------------------
def stay_folds(patient_ids, septic, n_folds: int = 5, seed: int = 0) -> dict:
    """{patient_id: fold} assigning whole stays to folds; septic and non-septic stays are shuffled separately and dealt
    round-robin so every fold gets about the same number of each. ``patient_ids``/``septic``: one entry per stay."""
    ids = np.asarray(list(patient_ids))
    sep = np.asarray(list(septic), dtype=bool)
    if len(ids) != len(sep):
        raise ValueError("patient_ids and septic must have the same length")
    rng = np.random.default_rng(seed)
    fold = {}
    for flag in (True, False):
        sel = ids[sep == flag]
        sel = sel[rng.permutation(len(sel))]
        for i, p in enumerate(sel):
            fold[p] = i % n_folds
    return fold


def _lgbm_params(seed: int, target: str) -> dict:
    return dict(objective="regression" if target == "gain" else "binary", n_estimators=150, learning_rate=0.03,
                num_leaves=6, max_depth=4, min_child_samples=60, subsample=0.7, subsample_freq=1,
                colsample_bytree=0.3, reg_lambda=20.0, random_state=seed, n_jobs=1, max_bin=63, verbose=-1,
                deterministic=True, force_row_wise=True)


class SepsisModel:
    """Shallow LightGBM models plus an L2 logistic regression on ``build_features``, trained with patient-grouped folds.

    ``targets``: training targets of ``stay_targets`` ("label", "window", "gain"), one fold ensemble of LightGBM models
    each; ``n_folds``: stay-level folds; ``smooth`` / ``hold``: ``causal_smooth`` window and ``hold_alarms`` hours applied
    to the scores / alarms; ``threshold``: fixed cut-off on the combined score, or None to take the cut-off that
    maximises the normalized utility of the out-of-fold alarms. In that utility the septic stays are weighted so that
    they make up the share ``prevalence`` of the stays (default 0.073, the share of septic stays in the public
    2019 challenge data; None: no weighting, i.e. the septic share of the training table). A non-septic stay costs
    -0.05 per alarmed hour whatever the share of septic stays, so the best cut-off falls when that share rises.
    ``measurement``: whether the "when was it measured" columns (``X_since``, ``n_meas{w}``) are inputs; Unit1/Unit2
    (``drop``) are left out of the inputs. ``linear_weight`` w: share of the combined score that comes from a
    class-balanced L2 logistic regression (inverse regularisation ``linear_c``) on the same inputs, median-imputed,
    winsorised at the 0.5/99.5 % training quantiles and standardised; 0 = LightGBM only. The combined score of a row is
    (1 - w) * (mean over targets of the fold-averaged LightGBM output) + w * (fold-averaged logistic output), each
    output first mapped to its quantile among the out-of-fold outputs (0..1)."""

    def __init__(self, targets=("label", "gain"), n_folds: int = 5, smooth: int = 3, hold: int = 0, threshold=None,
                 prevalence: float | None = 0.073, seed: int = 0, params: dict | None = None,
                 drop=("Unit1", "Unit2"), measurement: bool = False, linear_weight: float = 0.25,
                 linear_c: float = 0.1):
        self.measurement = bool(measurement)
        if not 0.0 <= float(linear_weight) < 1.0:
            raise ValueError("linear_weight must be in [0, 1)")
        self.linear_weight, self.linear_c = float(linear_weight), float(linear_c)
        self.targets = (targets,) if isinstance(targets, str) else tuple(targets)
        self.n_folds, self.smooth, self.hold = int(n_folds), int(smooth), int(hold)
        self.fixed_threshold, self.seed, self.params, self.drop = threshold, int(seed), dict(params or {}), tuple(drop)
        if prevalence is not None and not 0.0 < float(prevalence) < 1.0:
            raise ValueError("prevalence must be in (0, 1) or None")
        self.prevalence = None if prevalence is None else float(prevalence)
        for t in self.targets:
            if t not in ("label", "window", "gain"):
                raise ValueError(f"unknown target {t!r}: use 'label', 'window' or 'gain'")

    def _mapped(self, raw: dict) -> np.ndarray:
        """Mean over targets of the quantile of each raw output among the out-of-fold outputs; with ``linear_weight`` w > 0
        the score is (1 - w) * that + w * the quantile of the logistic-regression output."""
        q = lambda t: np.searchsorted(self._ref[t], raw[t], side="right") / len(self._ref[t])
        s = np.mean([q(t) for t in self.targets], axis=0)
        return (1.0 - self.linear_weight) * s + self.linear_weight * q("linear") if self.linear_weight > 0 else s

    def _raw(self, feats: pd.DataFrame) -> dict:
        x = feats[self.columns_].to_numpy(dtype=np.float32)
        out = {}
        for t in self.targets:
            out[t] = np.mean([m.predict(x) if t == "gain" else m.predict_proba(x)[:, 1] for m in self.models_[t]], axis=0)
        if self.linear_weight > 0:
            z = self._linear_prep(x)
            out["linear"] = np.mean([m.predict_proba(z)[:, 1] for m in self.linear_models_], axis=0)
        return out

    def _linear_prep(self, x: np.ndarray) -> np.ndarray:
        """Median-imputed, winsorised, standardised inputs of the logistic-regression component (statistics of the
        training table)."""
        lo, hi, med, mu, sd = self._lin_stats
        z = np.where(np.isfinite(x), x, med)
        z = np.clip(z, lo, hi)
        return ((z - mu) / sd).astype(np.float64)

    def fit(self, train: pd.DataFrame) -> "SepsisModel":
        """Fit on a labelled table (``SepsisLabel``); sets ``oof_scores_`` (combined out-of-fold score per row),
        ``threshold_``, ``oof_utility_`` (normalized utility of the out-of-fold alarms), ``oof_alarm_rate_`` (their share of
        the training hours) and ``models_``."""
        import lightgbm as lgb
        _check_columns(train, ["SepsisLabel"])
        if train["SepsisLabel"].max() < 1:
            raise ValueError("the training table has no septic stay")
        feats = build_features(train, measurement=self.measurement).drop(columns=[c for c in self.drop if c in FEATURES], errors="ignore")
        self.columns_ = list(feats.columns[(feats.nunique() > 1).to_numpy()])
        x = feats[self.columns_].to_numpy(dtype=np.float32)
        pids = train["patient_id"].to_numpy()
        stays = pd.unique(pids)
        septic = train.groupby("patient_id")["SepsisLabel"].max().reindex(stays).to_numpy() > 0
        fold_of = stay_folds(stays, septic, self.n_folds, self.seed)
        fold = np.array([fold_of[p] for p in pids])
        self.models_, self._ref, oof_raw = {}, {}, {}
        for t in self.targets:
            y = stay_targets(train, t)
            ms, oof = [], np.zeros(len(train))
            for k in range(self.n_folds):
                tr, te = fold != k, fold == k
                if not te.any() or not tr.any():
                    continue
                params = _lgbm_params(self.seed + k, t)
                params.update(self.params)
                mdl = lgb.LGBMRegressor(**params) if t == "gain" else lgb.LGBMClassifier(**params)
                mdl.fit(x[tr], y[tr])
                oof[te] = mdl.predict(x[te]) if t == "gain" else mdl.predict_proba(x[te])[:, 1]
                ms.append(mdl)
            self.models_[t], oof_raw[t] = ms, oof
            self._ref[t] = np.sort(oof)
        if self.linear_weight > 0:
            from sklearn.linear_model import LogisticRegression
            import warnings
            with np.errstate(all="ignore"):
                med = np.nanmedian(x, axis=0)
                lo, hi = np.nanquantile(x, 0.005, axis=0), np.nanquantile(x, 0.995, axis=0)
                z0 = np.clip(np.where(np.isfinite(x), x, med), lo, hi)
                mu, sd = z0.mean(0), z0.std(0) + 1e-6
            self._lin_stats = (lo, hi, med, mu, sd)
            z = self._linear_prep(x)
            y = stay_targets(train, "label")
            oof, self.linear_models_ = np.zeros(len(train)), []
            for k in range(self.n_folds):
                tr, te = fold != k, fold == k
                if not te.any() or not tr.any():
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    lr = LogisticRegression(C=self.linear_c, class_weight="balanced", max_iter=300).fit(z[tr], y[tr])
                oof[te] = lr.predict_proba(z[te])[:, 1]
                self.linear_models_.append(lr)
            oof_raw["linear"] = oof
            self._ref["linear"] = np.sort(oof)
        self._train_ids, self._train_labels = pids, train["SepsisLabel"].to_numpy()
        self.oof_scores_ = self._mapped(oof_raw)
        sm = causal_smooth(self.oof_scores_, pids, self.smooth)
        self.threshold_ = float(self.fixed_threshold) if self.fixed_threshold is not None else self._pick(sm)
        pred = hold_alarms((sm >= self.threshold_).astype(int), pids, self.hold)
        self.oof_utility_ = normalized_utility(_split(self._train_labels, pids), _split(pred, pids))
        self.oof_alarm_rate_ = float(pred.mean())
        return self

    def _pick(self, scores: np.ndarray) -> float:
        """Cut-off with the largest weighted utility of the out-of-fold alarms (utility is linear in the alarms: sum of
        the row gains of the alarmed hours; septic stays get the weight that gives them the share ``prevalence``); the
        utility curve is averaged over 5 neighbouring cut-offs."""
        pids, lab = self._train_ids, self._train_labels
        gains, septic = np.zeros(len(lab)), np.zeros(len(lab), dtype=bool)
        for rows, y in zip(_split(np.arange(len(lab)), pids), _split(lab, pids)):
            gains[rows], septic[rows] = row_gains(y), bool(np.max(y) > 0)
        weight = 1.0
        if self.prevalence is not None:
            n_pos = len(np.unique(pids[septic]))
            n_neg = len(np.unique(pids)) - n_pos
            if n_pos and n_neg:
                weight = (self.prevalence / (1.0 - self.prevalence)) / (n_pos / n_neg)
        gains = np.where(septic, weight * gains, gains)
        cand = np.unique(np.quantile(scores, np.linspace(0.5, 0.999, 120)))
        util = np.array([float(np.sum(gains * hold_alarms((scores >= t).astype(int), pids, self.hold)))
                         for t in cand])
        k = 5
        sm = np.convolve(np.pad(util, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
        return float(cand[int(np.argmax(sm))])

    def score(self, table: pd.DataFrame) -> np.ndarray:
        """Combined score per row of ``table`` (causally smoothed when ``smooth`` > 1)."""
        self._need_fit()
        s = self._mapped(self._raw(build_features(table, measurement=self.measurement)))
        return causal_smooth(s, table["patient_id"].to_numpy(), self.smooth)

    def predict(self, table: pd.DataFrame, patient_ids=None) -> list:
        """0/1 int arrays, one per stay: order of first appearance in ``table``, or that of ``patient_ids`` when given."""
        self._need_fit()
        pids = table["patient_id"].to_numpy()
        p = hold_alarms((self.score(table) >= self.threshold_).astype(int), pids, self.hold)
        parts = _split(p, pids)
        if patient_ids is None:
            return [a.astype(int) for a in parts]
        pos = {q: i for i, q in enumerate(pd.unique(pids))}
        missing = [q for q in patient_ids if q not in pos]
        if missing:
            raise ValueError(f"patient_ids not in the table: {missing[:3]}")
        return [parts[pos[q]].astype(int) for q in patient_ids]

    def _need_fit(self) -> None:
        if not hasattr(self, "models_"):
            raise RuntimeError("call fit(train) first")


def fit_predict(train: pd.DataFrame, tables, **model_kwargs):
    """``SepsisModel(**model_kwargs).fit(train)`` and its 0/1 alarms for every table in the list ``tables``.

    Returns ``(preds, info)``: ``preds[i]`` is a list of 0/1 int arrays, one per stay of ``tables[i]`` in order of first
    appearance, each as long as the stay; ``info`` = {"threshold", "oof_utility" (normalized utility of the out-of-fold
    alarms on ``train``), "oof_alarm_rate" (share of out-of-fold training hours alarmed at that threshold),
    "n_positive_hours" (per table)}. Typical call: ``(dev_pred, y), info = fit_predict(train, [dev_table, eval_table])``."""
    if isinstance(tables, pd.DataFrame):
        raise ValueError("tables must be a list of tables, e.g. [dev_table, eval_table]")
    model = SepsisModel(**model_kwargs).fit(train)
    preds = [model.predict(t) for t in tables]
    info = {"threshold": model.threshold_, "oof_utility": model.oof_utility_, "oof_alarm_rate": model.oof_alarm_rate_,
            "n_positive_hours": [int(sum(int(a.sum()) for a in p)) for p in preds]}
    return preds, info
