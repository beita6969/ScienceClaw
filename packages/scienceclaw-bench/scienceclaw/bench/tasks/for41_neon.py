"""FoR41 Environmental sciences — NEON Ecological Forecasting Challenge, aquatics theme (mean CRPS).

Data (read-only): ``<data_root>/for41-neon-aquatics/aquatics-targets.csv.gz`` — the public neon4cast aquatics
target snapshot (``project_id=neon4cast/duration=P1D``, last modified 2026-08-15): daily means of dissolved oxygen
(``oxygen``, mg/L), surface water temperature (``temperature``, °C) and chlorophyll-a (``chla``, µg/L) at 34 NEON
aquatic sites, 2016-03-04 .. 2026-07-31 (``NA`` = not observed).

Item = one (site, reference date t0) forecast: input the site's daily history of the three variables for the
1461 days ending at t0; target the observed daily oxygen and temperature on t0+1 .. t0+30 (the challenge's 30-day
horizon). Reference dates lie on a 37-day grid from 2019-06-01 (37 > 30 + 7, so an item's target window never
overlaps the last 7 history days of another item at the same site). Eligible: for both oxygen and temperature,
>= 10 observed target days, >= 60 observed days in the 365 days up to t0, and an observation within the last 7
days up to t0.

Pools / splits (fixed by ``partition_seed``, independent of the episode seed):

* IID pool = the 24 NEON **wadeable-stream** sites; temporal windows for t0: ``src`` 2019-06-01..2022-12-31 (all
  eligible items), ``val`` 2023 (32 items), ``id`` 2024-01-01..2026-06-30 (64 items), val/id picked round-robin
  over sites;
* OOD pool = the 7 **lake** and 3 **non-wadeable river** sites, t0 in 2024-01-01..2026-06-30 (64 items)
  -> ``ood_kind = "proxy_within_dataset"`` (ecosystem-type shift; no second forecasting dataset is available).
* The 30-day target windows of all val / id / ood items are masked (NaN, all variables) in every visible history,
  so no evaluation label is visible in any episode; in addition every episode withholds the target windows of its
  own items from all of its inputs (relevant when a site occurs twice in one episode). Episodes are drawn with
  distinct sites where the sub-pool allows it.

Deliverable: a normal predictive distribution per item, variable and lead day, the neon4cast ``family = "normal"``
format: ``y = {"oxygen_mu", "oxygen_sigma", "temperature_mu", "temperature_sigma"}``, each (n, 30).

Metric (lower is better): CRPS of N(mu, sigma²) at the observation (closed form, Gneiting & Raftery 2007; the
challenge's ``scoringRules::crps_norm``), averaged over all observed (item, day) pairs **per variable**; the
primary score is the prespecified equal-weight mean of the oxygen CRPS (mg/L) and the temperature CRPS (°C), as in
the paper. The two per-variable CRPS values have different units and are always reported separately in
``metrics``. Reference: smoothed day-of-year climatology from the visible history (mean / sd of observations
within ±7 days of the target day-of-year, sd floored at 0.1; fallback last-365-day mean / sd), the neon4cast
climatology null model. Acceptance: score <= 0.95 x reference.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import scilib
from ...core.schema import PortSchema
from ...core.trace import Trace
from ..task import ConstraintSpec, EvalResult, Episode, ToolSpec
from ._forecast_common import (ForecastAdapterBase, PoolItem, _round_robin, common_lineage, crps_normal,
                               default_budget, episode_id, finish_eval, invalid_eval, make_rng, to_float_array,
                               unwrap_payloads)

CODE = "FoR41"
VARIABLES = ("oxygen", "temperature", "chla")
UNITS = {"oxygen": "mg/L", "temperature": "degC", "chla": "ug/L"}
TARGET_VARS = ("oxygen", "temperature")
KEYS = ("oxygen_mu", "oxygen_sigma", "temperature_mu", "temperature_sigma")
H = 30
L = 1461
GRID_START = np.datetime64("2019-06-01", "D")
GRID_STEP = 37
WINDOWS = {"src": ("2019-06-01", "2022-12-31"), "val": ("2023-01-01", "2023-12-31"),
           "id": ("2024-01-01", "2026-06-30"), "ood": ("2024-01-01", "2026-06-30")}
POOL_SIZES = {"val": 32, "id": 64, "ood": 64}
MIN_TARGET_OBS, MIN_HIST_OBS, MAX_STALENESS = 10, 60, 7
CLIM_HALF_WINDOW, CLIM_MIN_N, SIGMA_FLOOR = 7, 5, 0.1
ACCEPT_MARGIN = 0.05
RANGES = {"oxygen_mu": (0.0, 30.0), "temperature_mu": (-5.0, 45.0), "oxygen_sigma": (0.0, 50.0),
          "temperature_sigma": (0.0, 50.0)}

# NEON aquatic site types (public NEON field-site metadata; neon4cast aquatics site list)
LAKES = ("BARC", "CRAM", "LIRO", "PRLA", "PRPO", "SUGG", "TOOK")
RIVERS = ("BLWA", "FLNT", "TOMB")
STREAMS = ("ARIK", "BIGC", "BLDE", "BLUE", "CARI", "COMO", "CUPE", "GUIL", "HOPB", "KING", "LECO", "LEWI", "MART",
           "MAYF", "MCDI", "MCRA", "OKSR", "POSE", "PRIN", "REDB", "SYCA", "TECR", "WALK", "WLOU")
SITE_TYPE = {**{s: "lake" for s in LAKES}, **{s: "non-wadeable river" for s in RIVERS},
             **{s: "wadeable stream" for s in STREAMS}}

DATASET = "NEON Ecological Forecasting Challenge (neon4cast) aquatics targets, daily (P1D)"
VERSION = "public target snapshot, last-modified 2026-08-15 (sha256 0851479375ef...)"
SOURCE = "https://projects.ecoforecast.org/neon4cast-ci/targets.html"
LICENSE = "NEON data policy (CC0 / public domain for NEON data products); neon4cast targets are derived NEON data"


@dataclass
class NeonData:
    days: np.ndarray                       # (D,) datetime64[D]
    raw: dict[str, np.ndarray]             # site -> (D, 3) observations (NaN missing)
    visible: dict[str, np.ndarray]         # site -> (D, 3) with val/id/ood target windows masked
    items: dict[str, tuple[str, int]]      # item id -> (site, t0 index)
    pools: dict[str, list[str]]
    excluded_sites: list[str]


def _item_id(site: str, t0: np.datetime64) -> str:
    return f"neon4cast-aquatics:{site}:ref={t0}:h{H}"


def load_neon(root: Path, partition_seed: int) -> NeonData:
    df = pd.read_csv(root / "for41-neon-aquatics" / "aquatics-targets.csv.gz",
                     usecols=["site_id", "datetime", "variable", "observation"], dtype={"observation": str})
    df["observation"] = pd.to_numeric(df["observation"], errors="coerce")
    df["date"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None).dt.normalize()
    df = df[df["variable"].isin(VARIABLES)]
    d0, d1 = df["date"].min().to_datetime64().astype("datetime64[D]"), df["date"].max().to_datetime64().astype("datetime64[D]")
    days = np.arange(d0, d1 + np.timedelta64(1, "D"))
    raw: dict[str, np.ndarray] = {}
    sites = sorted(df["site_id"].unique())
    excluded = [s for s in sites if s not in SITE_TYPE]
    obs = df.dropna(subset=["observation"])
    obs = obs[obs["site_id"].isin(SITE_TYPE)]
    g = obs.groupby(["site_id", "date", "variable"], as_index=False)["observation"].mean()
    day_idx = ((g["date"].to_numpy().astype("datetime64[D]") - d0).astype(int))
    var_idx = g["variable"].map({v: i for i, v in enumerate(VARIABLES)}).to_numpy()
    site_col = g["site_id"].to_numpy()
    vals = g["observation"].to_numpy(dtype=float)
    for s in sites:
        if s in excluded:
            continue
        a = np.full((days.size, len(VARIABLES)), np.nan)
        m = site_col == s
        a[day_idx[m], var_idx[m]] = vals[m]
        raw[s] = a
    # candidate items
    cands: dict[str, list[tuple[str, int]]] = {k: [] for k in WINDOWS}
    j = 0
    while True:
        t0 = GRID_START + np.timedelta64(GRID_STEP * j, "D")
        j += 1
        if t0 > np.datetime64(WINDOWS["id"][1], "D") or t0 + np.timedelta64(H, "D") > d1:
            break
        ti = int((t0 - d0).astype(int))
        for s, a in raw.items():
            ok = True
            for vi in (0, 1):
                col = a[:, vi]
                tgt = col[ti + 1:ti + 1 + H]
                hist365 = col[max(0, ti - 364):ti + 1]
                recent = col[max(0, ti - MAX_STALENESS + 1):ti + 1]
                if (np.isfinite(tgt).sum() < MIN_TARGET_OBS or np.isfinite(hist365).sum() < MIN_HIST_OBS
                        or not np.isfinite(recent).any()):
                    ok = False
                    break
            if not ok:
                continue
            for split, (lo, hi) in WINDOWS.items():
                iid_site = SITE_TYPE[s] == "wadeable stream"
                if (split == "ood") == iid_site:
                    continue
                if np.datetime64(lo, "D") <= t0 <= np.datetime64(hi, "D"):
                    cands[split].append((s, ti))
    items: dict[str, tuple[str, int]] = {}
    pools: dict[str, list[str]] = {}
    for split, lst in cands.items():
        pitems = [PoolItem(_item_id(s, days[ti]), s) for s, ti in lst]
        for (s, ti), it in zip(lst, pitems):
            items[it.id] = (s, ti)
        if split in POOL_SIZES:
            order = _round_robin(pitems, make_rng(CODE, "pool", split, partition_seed))
            if len(order) < POOL_SIZES[split]:
                raise ValueError(f"only {len(order)} eligible {split} items, need {POOL_SIZES[split]}")
            pools[split] = [it.id for it in order[:POOL_SIZES[split]]]
        else:
            pools[split] = [it.id for it in pitems]
    visible = {s: a.copy() for s, a in raw.items()}
    for split in POOL_SIZES:
        for iid in pools[split]:
            s, ti = items[iid]
            visible[s][ti + 1:ti + 1 + H, :] = np.nan
    return NeonData(days=days, raw=raw, visible=visible, items=items, pools=pools, excluded_sites=excluded)


# --------------------------------------------------------------------------------------------- reference
def climatology(hist: np.ndarray, hist_days: np.ndarray, target_days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed day-of-year climatology (mean, sd) of one variable for each target day (neon4cast null model)."""
    fin = np.isfinite(hist)
    hv, hd = hist[fin], hist_days[fin]
    if hv.size == 0:
        raise ValueError("climatology: the visible history holds no observation")
    doy_h = (hd - hd.astype("datetime64[Y]")).astype(int)
    last = hist[-365:]
    last = last[np.isfinite(last)]
    fb_mu = float(np.mean(last)) if last.size >= 2 else float(np.mean(hv))
    fb_sd = max(float(np.std(last, ddof=1)) if last.size >= 2 else SIGMA_FLOOR, SIGMA_FLOOR)
    mu, sd = np.empty(len(target_days)), np.empty(len(target_days))
    for k, td in enumerate(target_days):
        doy = int((td - td.astype("datetime64[Y]")).astype(int))
        dist = np.abs(doy_h - doy)
        dist = np.minimum(dist, 365 - dist)
        sel = hv[dist <= CLIM_HALF_WINDOW]
        if sel.size >= CLIM_MIN_N:
            mu[k], sd[k] = float(sel.mean()), max(float(sel.std(ddof=1)), SIGMA_FLOOR)
        else:
            mu[k], sd[k] = fb_mu, fb_sd
    return mu, sd


def crps_sums(pred: dict[str, np.ndarray], obs: np.ndarray) -> dict[str, np.ndarray]:
    """Per-item sum and count of CRPS over observed days for oxygen and temperature. obs: (n, H, 2)."""
    out = {}
    for vi, v in enumerate(TARGET_VARS):
        o = obs[:, :, vi]
        m = np.isfinite(o)
        c = np.zeros_like(o)
        c[m] = crps_normal(pred[f"{v}_mu"][m], pred[f"{v}_sigma"][m], o[m])
        out[f"{v}_sum"] = c.sum(axis=1)
        out[f"{v}_n"] = m.sum(axis=1)
    return out


def score_from_sums(s: dict[str, np.ndarray]) -> tuple[float, dict[str, float]]:
    per = {v: float(np.sum(s[f"{v}_sum"]) / max(int(np.sum(s[f"{v}_n"])), 1)) for v in TARGET_VARS}
    return float(np.mean([per[v] for v in TARGET_VARS])), per


def coerce_prediction(y: Any, n: int) -> dict[str, np.ndarray]:
    """Validate the deliverable dict; raises ValueError with a precise message."""
    if not isinstance(y, dict):
        raise ValueError(f"y must be a dict with keys {list(KEYS)}, got {type(y).__name__}")
    missing = [k for k in KEYS if k not in y]
    if missing:
        raise ValueError(f"y is missing keys {missing}")
    out = {}
    for k in KEYS:
        a = to_float_array(y[k])
        if a.shape != (n, H):
            raise ValueError(f"y[{k!r}] has shape {tuple(a.shape)}, required ({n}, {H})")
        out[k] = a
    return out


# --------------------------------------------------------------------------------------------- adapter
class Adapter(ForecastAdapterBase):
    """ScienceClaw-Eval FoR41 adapter (neon4cast aquatics: oxygen + temperature, 30-day normal forecasts)."""

    code = CODE
    name = "NEON aquatics forecasting"
    task_type = "probabilistic_forecasting"

    def _check_files(self) -> list[str]:
        p = self.data_root / "for41-neon-aquatics" / "aquatics-targets.csv.gz"
        return [] if p.is_file() else [f"missing {p}"]

    def data(self) -> NeonData:
        return self._memo.get("data", lambda: load_neon(self.data_root, self.partition_seed))

    def _split_pools(self) -> dict[str, list[PoolItem]]:
        d = self.data()
        return {k: [PoolItem(i, d.items[i][0]) for i in ids] for k, ids in d.pools.items()}

    def _draw_options(self, split: str) -> dict:
        return {"distinct_groups": True}

    def _window(self, a: np.ndarray, ti: int) -> tuple[np.ndarray, np.ndarray]:
        """(L, 3) history of the site array ``a`` ending at day index ti (NaN-padded) and its (L,) dates."""
        d = self.data()
        lo = ti - L + 1
        out = np.full((L, len(VARIABLES)), np.nan)
        src_lo = max(lo, 0)
        out[src_lo - lo:] = a[src_lo:ti + 1]
        dates = d.days[0] + np.arange(lo, ti + 1).astype("timedelta64[D]")
        return out, dates

    # ----------------------------------------------------------------------------------------------
    def _make_episode(self, split: str, k: int, items: list[PoolItem], seed: int, reused: bool) -> Episode:
        d = self.data()
        ids = [it.id for it in items]
        n = len(ids)
        sites = [d.items[i][0] for i in ids]
        tis = [d.items[i][1] for i in ids]
        t0s = [d.days[ti] for ti in tis]
        # per-episode view: the globally masked series with every item's own target window also withheld, so
        # that an item never sees another item's target at the same site (all inputs below come from ep_vis)
        ep_vis: dict[str, np.ndarray] = {}
        for s, ti in zip(sites, tis):
            ep_vis.setdefault(s, d.visible[s].copy())[ti + 1:ti + 1 + H] = np.nan
        hist, hdates = zip(*[self._window(ep_vis[s], ti) for s, ti in zip(sites, tis)])
        hist = np.stack(hist)
        obs = np.stack([d.raw[s][ti + 1:ti + 1 + H, :2] for s, ti in zip(sites, tis)])
        tdays = [t0 + np.arange(1, H + 1).astype("timedelta64[D]") for t0 in t0s]

        def reference(hist_: np.ndarray, hdates_: list[np.ndarray], tdays_: list[np.ndarray]) -> dict[str, np.ndarray]:
            ref = {key: np.empty((len(hist_), H)) for key in KEYS}
            for j in range(len(hist_)):
                for vi, v in enumerate(TARGET_VARS):
                    mu, sd = climatology(hist_[j][:, vi], hdates_[j], tdays_[j])
                    ref[f"{v}_mu"][j], ref[f"{v}_sigma"][j] = mu, sd
            return ref

        ref_pred = reference(hist, list(hdates), tdays)
        ref_sums = crps_sums(ref_pred, obs)
        ref_score, ref_per = score_from_sums(ref_sums)

        def payload_of(s: dict[str, np.ndarray]) -> dict:
            return {"items": [{"id": ids[j], **{kk: float(s[kk][j]) for kk in s}} for j in range(n)]}

        ref_payload = payload_of(ref_sums)
        pool = "ood" if split == "ood" else "iid"

        # ------------------------------------------------------------------ dev backtest (visible data only)
        dev_windows = [self._window(ep_vis[s], ti - H) for s, ti in zip(sites, tis)]
        dev_hist = np.stack([w[0] for w in dev_windows])
        dev_obs = np.stack([ep_vis[s][ti - H + 1:ti + 1, :2] for s, ti in zip(sites, tis)])
        dev_tdays = [t0 - np.timedelta64(H, "D") + np.arange(1, H + 1).astype("timedelta64[D]") for t0 in t0s]
        dev_ref_score, _ = score_from_sums(crps_sums(reference(dev_hist, [w[1] for w in dev_windows], dev_tdays),
                                                     dev_obs))
        meta = {"site_id": list(sites), "site_type": [SITE_TYPE[s] for s in sites],
                "reference_date": [str(t) for t in t0s]}

        def load_eval_inputs(inputs: dict, config: dict) -> dict:
            return {"history": hist.copy(), "variables": list(VARIABLES), "units": [UNITS[v] for v in VARIABLES],
                    "horizon_days": H, **{kk: list(v) for kk, v in meta.items()}}

        def load_dev(inputs: dict, config: dict) -> dict:
            return {"history": dev_hist.copy(), "variables": list(VARIABLES), "horizon_days": H,
                    "site_id": list(sites), "site_type": list(meta["site_type"]),
                    "reference_date": [str(t - np.timedelta64(H, "D")) for t in t0s]}

        def score_dev(inputs: dict, config: dict) -> dict:
            pred = coerce_prediction(inputs.get("pred"), n)
            for key, a in pred.items():
                if not np.all(np.isfinite(a)):
                    raise ValueError(f"pred[{key!r}] contains non-finite values")
                if key.endswith("sigma") and np.any(a <= 0):
                    raise ValueError(f"pred[{key!r}] must be > 0")
            s, per = score_from_sums(crps_sums(pred, dev_obs))
            return {"score": s, "report": {"metric": "mean CRPS on the dev backtest = (CRPS oxygen [mg/L] + CRPS "
                                                     "temperature [degC]) / 2, lower is better", "score": s,
                                           "per_variable": per, "reference_score": dev_ref_score, "n_items": n}}

        pred_schema = PortSchema("dict", None, None, None,
                                 f"keys {list(KEYS)}; each a float array ({n}, {H}); oxygen in mg/L, temperature in degC")
        hist_schema = PortSchema("array", (n, L, len(VARIABLES)), None, "float",
                                 "daily observations; channel order oxygen [mg/L], temperature [degC], chla [ug/L]; "
                                 "history[i, -1] is reference_date[i]; NaN = not observed or withheld")
        lst = PortSchema("list", (n,), None, "str")
        tools = [
            ToolSpec("load_eval_inputs",
                     f"Evaluation inputs (no targets) for the {n} items in item order: history[i, t, c] = daily mean "
                     f"of variable c at site_id[i] on day reference_date[i] - ({L - 1} - t) days (t = 0..{L - 1}); "
                     "site_type[i] (NEON aquatic site class); variables, units; horizon_days = 30.",
                     {}, {"history": hist_schema, "variables": PortSchema("list", (3,), None, "str"),
                          "units": PortSchema("list", (3,), None, "str"),
                          "horizon_days": PortSchema("number", None, "d", "int"),
                          "site_id": lst, "site_type": lst, "reference_date": lst},
                     load_eval_inputs),
            ToolSpec("load_dev",
                     "Dev backtest inputs: for every item, the same site's history ending 30 days before its "
                     "reference date (same layout as load_eval_inputs; reference_date[i] here is the last day of "
                     "that history); score_dev holds the following 30 observed days, which are part of the "
                     "visible history (days withheld from the history are not scored).",
                     {}, {"history": hist_schema, "variables": PortSchema("list", (3,), None, "str"),
                          "horizon_days": PortSchema("number", None, "d", "int"),
                          "site_id": lst, "site_type": lst, "reference_date": lst},
                     load_dev),
            ToolSpec("score_dev",
                     "Scores a forecast of the dev backtest given in the deliverable format (dict of four (n, 30) "
                     "arrays): returns the mean CRPS (lower is better) and, in report, per-variable CRPS and the "
                     "reference forecast's score.",
                     {"pred": pred_schema},
                     {"score": PortSchema("number", None, None, "float"), "report": PortSchema("dict")},
                     score_dev),
        ]

        # ------------------------------------------------------------------ D_V constraints
        def c_struct(y: Any, trace: Trace | None) -> tuple[bool, str]:
            try:
                coerce_prediction(y, n)
            except ValueError as ex:
                return False, str(ex)
            return True, f"dict with {list(KEYS)} of shape ({n}, {H})"

        def c_fin(y: Any, trace: Trace | None) -> tuple[bool, str]:
            try:
                p = coerce_prediction(y, n)
            except ValueError as ex:
                return False, str(ex)
            bad = {kk: int((~np.isfinite(a)).sum()) for kk, a in p.items() if (~np.isfinite(a)).any()}
            return not bad, ("all finite" if not bad else f"non-finite entries {bad}")

        def c_sigma(y: Any, trace: Trace | None) -> tuple[bool, str]:
            try:
                p = coerce_prediction(y, n)
            except ValueError as ex:
                return False, str(ex)
            bad = {kk: int((p[kk] <= 0).sum()) for kk in ("oxygen_sigma", "temperature_sigma") if (p[kk] <= 0).any()}
            return not bad, ("sigma > 0" if not bad else f"non-positive sigma entries {bad}")

        def c_rng(y: Any, trace: Trace | None) -> tuple[bool, str]:
            try:
                p = coerce_prediction(y, n)
            except ValueError as ex:
                return False, str(ex)
            for kk, (lo, hi) in RANGES.items():
                a = p[kk][np.isfinite(p[kk])]
                if a.size and (a.min() < lo or a.max() > hi):
                    return False, f"{kk} outside [{lo:g}, {hi:g}] (min {a.min():.4g}, max {a.max():.4g})"
            return True, "within physical ranges"

        constraints = [
            ConstraintSpec("output_structure", f"y is a dict with keys {list(KEYS)}, each a numeric array of shape "
                                               f"({n}, {H}) aligned with the items and lead days", c_struct),
            ConstraintSpec("finite", "every mu and sigma is finite", c_fin),
            ConstraintSpec("positive_sigma", "every sigma is > 0 (valid normal distribution)", c_sigma),
            ConstraintSpec("physical_range", "oxygen_mu in [0, 30] mg/L, temperature_mu in [-5, 45] degC, sigmas "
                                             "<= 50 (native units: mg/L, degC)", c_rng),
        ]

        def evaluate(y: Any, trace: Trace | None) -> EvalResult:
            common = {"reference": ref_score, "direction": "min", "margin": ACCEPT_MARGIN,
                      "reference_payload": ref_payload}
            base_m = {"reference_mean_crps": ref_score, "reference_crps_oxygen_mgL": ref_per["oxygen"],
                      "reference_crps_temperature_degC": ref_per["temperature"]}
            try:
                pred = coerce_prediction(y, n)
            except ValueError as ex:
                return invalid_eval(str(ex), metrics=base_m, **common)
            if not all(np.all(np.isfinite(a)) for a in pred.values()):
                return invalid_eval("non-finite predictions", metrics=base_m, **common)
            if np.any(pred["oxygen_sigma"] <= 0) or np.any(pred["temperature_sigma"] <= 0):
                return invalid_eval("non-positive sigma", metrics=base_m, **common)
            sums = crps_sums(pred, obs)
            score, per = score_from_sums(sums)
            metrics = {"mean_crps": score, "crps_oxygen_mgL": per["oxygen"], "crps_temperature_degC": per["temperature"],
                       "n_obs_oxygen": float(np.sum(sums["oxygen_n"])),
                       "n_obs_temperature": float(np.sum(sums["temperature_n"])), **base_m}
            return finish_eval(primary=score, metrics=metrics, payload=payload_of(sums), **common)

        objective = (
            "Environmental sciences / ecological forecasting (NEON Ecological Forecasting Challenge, aquatics "
            f"theme). There are {n} evaluation items; each is one NEON aquatic site and a reference date t0. Input of "
            f"item i (tool load_eval_inputs): history[i], the site's daily means of dissolved oxygen (mg/L), water "
            f"temperature (degC) and chlorophyll-a (ug/L) for the {L} days ending at t0 (NaN = not observed or "
            "withheld), plus site id and site type. Target of item i: the observed daily mean oxygen and temperature "
            f"at the same site on each of the days t0+1 .. t0+{H} (days without an observation are not scored).\n"
            f"Deliverable y: a dict of four float arrays of shape ({n}, {H}): 'oxygen_mu', 'oxygen_sigma' (mg/L), "
            "'temperature_mu', 'temperature_sigma' (degC); [i, h] = mean and standard deviation of a normal "
            "predictive distribution for item i (order of load_eval_inputs) on day t0+h+1. Sigmas must be > 0; all "
            "values finite; oxygen_mu in [0, 30], temperature_mu in [-5, 45], sigmas <= 50.\n"
            "Score (lower is better): for each variable, the mean CRPS of the normal forecasts over all observed "
            "(item, day) pairs; the score is the equal-weight mean of the oxygen CRPS (mg/L) and the temperature "
            "CRPS (degC).\n"
            + scilib.describe("aquatics") + scilib.describe_extra("tsfm")
        )
        lineage = common_lineage(
            dataset=DATASET, version=VERSION, source=SOURCE, license_=LICENSE, pool=pool,
            ood_kind="proxy_within_dataset" if pool == "ood" else None, items=items, seed=seed,
            partition_seed=self.partition_seed, k=k, reused=reused,
            split_rule=("IID = 24 wadeable-stream sites; reference dates on a 37-day grid from 2019-06-01; src "
                        "2019-06-01..2022-12-31 (all eligible), val 2023 (32), id 2024-01-01..2026-06-30 (64), "
                        "round-robin over sites; OOD = 7 lakes + 3 non-wadeable rivers, 2024-01-01..2026-06-30 (64); "
                        "val/id/ood target windows masked in all visible histories"),
            visible=f"per-item {L}-day site history (masked), dev backtest ending 30 days before t0",
            extra={"horizon_days": H, "history_days": L, "reference_dates": meta["reference_date"],
                   "site_types": sorted(set(meta["site_type"])), "excluded_sites": list(d.excluded_sites)})
        return Episode(
            id=episode_id(CODE, split, seed, k), discipline=CODE, family=self.family, split=split,
            task_type=self.task_type, objective=objective, required_output=pred_schema, tools=tools,
            constraints=constraints, budget=default_budget(max_node_s=300.0, max_llm_items=2 * n), lineage=lineage,
            acceptance=(f"accepted iff mean CRPS (oxygen, temperature equal weight) <= {1 - ACCEPT_MARGIN:.2f} x "
                        "the reference CRPS on the same items (reference recipe and value only in "
                        "EvalResult.details / docs, never shown to the policy)"),
            tolerance={"rtol": 1e-5, "atol": 1e-6},
            tags=[CODE, self.family, "time-series", "forecasting", "probabilistic", "CRPS", "ecology", "freshwater",
                  "dissolved-oxygen", "water-temperature", "NEON", "neon4cast"],
            metric=self.metric, direction="min", n_items=n, _evaluate=evaluate, _dev_evaluate=None)

    # ----------------------------------------------------------------------------------------------
    def pooled_metric(self, per_episode: list[dict]) -> float | None:
        """Equal-weight mean of the oxygen and temperature CRPS, each averaged over all observed (item, day) pairs."""
        acc = {f"{v}_{s}": 0.0 for v in TARGET_VARS for s in ("sum", "n")}
        seen = False
        for p in unwrap_payloads(per_episode):
            for it in p.get("items", []):
                seen = True
                for kk in acc:
                    acc[kk] += float(it[kk])
        if not seen or any(acc[f"{v}_n"] == 0 for v in TARGET_VARS):
            return None
        return float(np.mean([acc[f"{v}_sum"] / acc[f"{v}_n"] for v in TARGET_VARS]))

    def pooled_per_variable(self, per_episode: list[dict]) -> dict[str, float]:
        """Pooled CRPS per variable (oxygen in mg/L, temperature in degC) for reporting."""
        out = {}
        for v in TARGET_VARS:
            s = n = 0.0
            for p in unwrap_payloads(per_episode):
                for it in p.get("items", []):
                    s += float(it[f"{v}_sum"])
                    n += float(it[f"{v}_n"])
            if n:
                out[v] = s / n
        return out
