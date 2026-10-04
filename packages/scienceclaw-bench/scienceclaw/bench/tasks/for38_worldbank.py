"""FoR38 Economics — World Bank WDI annual macro forecasting (mean sMAPE, %).

Data (read-only): ``<data_root>/for38-worldbank-wdi/source-api/`` — the recorded World Bank API v2 responses
(WDI, source=2, years 1990-2025, API ``lastupdated`` 2026-07-13) for the four indicators selected by the data team:

* ``NY.GDP.PCAP.KD``  GDP per capita (constant 2015 US$)            — forecast target (positive level)
* ``SL.UEM.TOTL.ZS``  Unemployment, total (% of total labor force)  — forecast target (positive level)
* ``NY.GDP.MKTP.KD.ZG`` GDP growth (annual %)                        — visible covariate only
* ``FP.CPI.TOTL.ZG``  Inflation, consumer prices (annual %)          — visible covariate only

plus ``countries.json`` (region / income-level metadata; aggregates excluded). sMAPE is only meaningful for
positive-valued series, so the growth-rate indicators (which cross zero) are covariates, not targets.

Item = one (economy, target indicator) annual series: input the values 1990-2021 (NaN = missing), target the four
values 2022-2025 (forecast origin 2021, horizon 4). Eligible items: non-aggregate economy, value in 2021 present,
>= 20 non-missing values in 1990-2021, all four target values present, all values > 0.

Pools / splits (economy-disjoint; fixed by ``partition_seed``, independent of the episode seed):

* IID pool: economies of the World Bank regions Europe & Central Asia, East Asia & Pacific, Latin America &
  Caribbean and North America; economies are hash-ordered and assigned whole (both indicators) to ``val`` until it
  holds >= 32 items, then ``id`` until >= 64 items, the rest to ``src``;
* OOD pool: economies of Sub-Saharan Africa; Middle East, North Africa, Afghanistan & Pakistan; South Asia
  -> ``ood_kind = "proxy_within_dataset"`` (regional shift; no second macro dataset is available locally).

Visible data (no value after the 2021 origin is ever shown): ``load_train`` = the 1990-2021 panel (all four
indicators) of all IID-region economies; ``load_eval_inputs`` = each item's history plus its economy's four
indicators 1990-2021; ``load_dev``/``score_dev`` = backtest on the items' own histories (origin 2017, targets
2018-2021, all visible years).

Anonymisation. The targets are public WDI statistics, so a policy that can recognise the
economy, the indicator and the calendar years could recall the 2022-2025 values from pre-training instead of
forecasting. Everything the policy sees is therefore opaque: economies are ids ``E000``.. (a partition-seed hash
permutation of *all* non-aggregate economies), indicators are series kinds ``T1``/``T2`` (targets: a positive
level, a percentage bounded by 100) and ``C1``/``C2`` (covariates: annual percentage changes), time is given as
periods relative to the origin (``t-31`` .. ``t0``; targets ``t+1`` .. ``t+4``), names/codes/WDI vintage/dataset
name are absent from the objective, tool descriptions and tags. Region and income level stay (coarse groups of
tens of economies; the OOD split is defined by region). The hidden payloads, metrics, item ids and split
manifests are unchanged (they carry the real ids; the policy never sees them). Residual risk, documented in
docs/tasks/FoR38.md: numeric fingerprints (levels, the 2009/2020 shocks) can still identify well-known series.

Metric (lower is better): mean over items of sMAPE = 200/4 Σ_h |y_h - ŷ_h| / (|y_h| + |ŷ_h|) (M4 definition, %).
Reference: naive (random walk) ŷ_h = value in 2021. Acceptance: mean sMAPE <= 0.97 x reference.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import scilib
from scilib import macro, tsfm

from ...core.schema import PortSchema
from ...core.trace import Trace
from ..task import EvalResult, Episode, ToolSpec
from ._forecast_common import (ForecastAdapterBase, PoolItem, c_declared_unit, c_finite, c_range, c_shape,
                               check_split, common_lineage, default_budget, episode_id, finish_eval, hash_order, invalid_eval,
                               smape, to_float_array, unwrap_payloads)

CODE = "FoR38"
TARGETS = ("NY.GDP.PCAP.KD", "SL.UEM.TOTL.ZS")
INDICATORS = ("NY.GDP.PCAP.KD", "SL.UEM.TOTL.ZS", "NY.GDP.MKTP.KD.ZG", "FP.CPI.TOTL.ZG")
FIRST_YEAR, ORIGIN, H = 1990, 2021, 4
DEV_ORIGIN = ORIGIN - H
MIN_OBS = 20
IID_REGIONS = ("Europe & Central Asia", "East Asia & Pacific", "Latin America & Caribbean", "North America")
OOD_REGIONS = ("Sub-Saharan Africa", "Middle East, North Africa, Afghanistan & Pakistan", "South Asia")
N_VAL, N_ID = 32, 64
ACCEPT_MARGIN = 0.03
UNIT = "native"

DATASET = "World Bank World Development Indicators (API v2 source=2), 4 indicators, 1990-2025"
VERSION = "WDI API lastupdated 2026-07-13 (recorded responses, sha256 in receipt.json)"
SOURCE = "https://datahelpdesk.worldbank.org/knowledgebase/articles/889392 ; https://api.worldbank.org/v2/"
LICENSE = "CC BY 4.0 (World Bank Open Data terms of use)"

# Explicit specialist route.  Keeping the configuration here makes the
# policy-visible tool reproducible instead of relying on whatever scilib
# defaults happen to be installed on a worker.
MACRO_TOOL_MEMBERS = tuple(macro.DEFAULT_MEMBERS)
MACRO_TOOL_CONFIG = {
    "members": list(MACRO_TOOL_MEMBERS),
    "weights": "equal",
    "n_backtest": 0,
    "seed": 0,
    "n_jobs": 1,
}

# Optional pretrained route.  Chronos-2 weights are staged on Leonardo and are
# called through the existing scilib remote broker when the torch-free runner
# cannot load them locally.  Keeping the model/context fixed makes comparisons
# reproducible and prevents policy-supplied model choices.
CHRONOS_TOOL_MODEL = "chronos_2"
CHRONOS_TOOL_CONTEXT = 32


def _fixed_macro_predict(inputs: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the fixed visible-data macro route without accepting target-bearing inputs."""
    import pandas as pd

    panel = inputs.get("panel")
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame from load_train")
    required = {"economy_id", "region", "income_level", "indicator"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"panel missing required columns: {missing}")
    if any(str(c).startswith("t+") for c in panel.columns):
        raise ValueError("panel contains post-origin columns")
    kinds = inputs.get("indicator_kinds")
    if not isinstance(kinds, dict) or set(kinds) != set(panel["indicator"].astype(str)):
        raise ValueError("indicator_kinds must describe exactly the visible panel kinds")
    history = np.asarray(inputs.get("history"), dtype=float)
    indicator = [str(x) for x in inputs.get("indicator", [])]
    covariates = np.asarray(inputs.get("covariates"), dtype=float)
    covariate_indicators = [str(x) for x in inputs.get("covariate_indicators", [])]
    n = len(indicator)
    if history.ndim != 2 or history.shape[0] != n:
        raise ValueError("history must have one row per evaluation item")
    if covariates.ndim != 3 or covariates.shape[:2] != (n, len(covariate_indicators)):
        raise ValueError("covariates must have shape (n_items, n_kinds, n_periods)")
    target_offsets = list(inputs.get("target_offsets", []))
    if target_offsets != list(range(1, H + 1)):
        raise ValueError("target_offsets must be [1, 2, 3, 4]")
    kwargs = {
        "train_panel": panel,
        "history": history,
        "indicator": indicator,
        "indicator_kinds": dict(kinds),
        "covariates": covariates,
        "covariate_indicators": covariate_indicators,
        "economy_id": [str(x) for x in inputs.get("economy_id", [])],
        "region": [str(x) for x in inputs.get("region", [])],
        "income_level": [str(x) for x in inputs.get("income_level", [])],
        "target_offsets": target_offsets,
        "horizon": H,
        **MACRO_TOOL_CONFIG,
    }
    pred = np.asarray(macro.fit_predict(**kwargs), dtype=float)
    if pred.shape != (n, H) or not np.all(np.isfinite(pred)) or np.any(pred < 0):
        raise ValueError(f"macro route returned invalid predictions with shape {pred.shape}")
    roles = macro.infer_roles(panel, kinds)
    for j, kind in enumerate(indicator):
        if roles.get(kind) == "percent" and np.any(pred[j] > 100.0 + 1e-9):
            raise ValueError("macro route returned percentage forecast above 100")
    info = dict(MACRO_TOOL_CONFIG)
    info.update({"method": "scilib.macro.fit_predict", "label_source": "none",
                 "target_source": "label-free load_eval_inputs only", "n_items": n})
    return pred, info


def _fixed_chronos_predict(inputs: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the fixed Chronos-2 median route from label-free evaluation inputs."""
    history = np.asarray(inputs.get("history"), dtype=float)
    indicator = [str(x) for x in inputs.get("indicator", [])]
    n = len(indicator)
    if history.ndim != 2 or history.shape[0] != n or history.shape[1] < 3:
        raise ValueError("history must have shape (n_items, n_periods >= 3)")
    if np.isinf(history).any():
        raise ValueError("history contains infinite values")
    target_offsets = list(inputs.get("target_offsets", []))
    if target_offsets != list(range(1, H + 1)):
        raise ValueError("target_offsets must be [1, 2, 3, 4]")
    kinds = inputs.get("indicator_kinds")
    if not isinstance(kinds, dict):
        raise ValueError("indicator_kinds must be a visible mapping")
    if any(kind not in kinds for kind in indicator):
        raise ValueError("indicator_kinds must describe every evaluation indicator")
    raw = tsfm.forecast(history, H, quantiles=(0.5,), model=CHRONOS_TOOL_MODEL,
                        context_length=CHRONOS_TOOL_CONTEXT)
    arr = np.asarray(raw, dtype=float)
    if arr.shape == (n, H, 1):
        pred = arr[:, :, 0]
    elif arr.shape == (n, H):
        pred = arr
    else:
        raise ValueError(f"Chronos route returned invalid shape {arr.shape}")
    if not np.all(np.isfinite(pred)):
        raise ValueError("Chronos route returned non-finite predictions")
    pred = np.maximum(pred, 0.0)
    hmax = np.nanmax(history, axis=1)
    if not np.all(np.isfinite(hmax)):
        raise ValueError("history must have at least one finite value per item")
    for j, kind in enumerate(indicator):
        doc = str(kinds[kind]).lower()
        cap = 100.0 if ("percent" in doc or "percentage" in doc) else 10.0 * hmax[j]
        pred[j] = np.minimum(pred[j], cap)
    info = {"method": "scilib.tsfm.forecast", "model": CHRONOS_TOOL_MODEL,
            "context_length": CHRONOS_TOOL_CONTEXT, "quantiles": [0.5],
            "label_source": "none", "target_source": "label-free load_eval_inputs only",
            "n_items": n}
    return pred, info


@dataclass
class WDIData:
    values: dict[str, dict[str, np.ndarray]]   # indicator -> iso3 -> (36,) values 1990..2025 (NaN missing)
    meta: dict[str, dict[str, str]]           # iso3 -> {name, region, income}
    ind_names: dict[str, str]
    items: dict[str, tuple[str, str]]          # item id -> (iso3, indicator)
    pools: dict[str, list[str]]
    iid_economies: list[str]
    econ_ids: dict[str, str] = field(default_factory=dict)      # iso3 -> opaque id shown to the policy
    series_ids: dict[str, str] = field(default_factory=dict)    # indicator code -> opaque kind (T1/T2/C1/C2)


YEARS = np.arange(FIRST_YEAR, 2026)

# What the policy is told about a series kind (no indicator name, no unit of account, no source).
KIND_DOC = {
    "NY.GDP.PCAP.KD": "strictly positive level series (native units)",
    "SL.UEM.TOTL.ZS": "percentage of a total, in (0, 100]",
    "NY.GDP.MKTP.KD.ZG": "annual percentage change (may be negative)",
    "FP.CPI.TOTL.ZG": "annual percentage change (may be negative)",
}


def _yi(year: int) -> int:
    return int(year - FIRST_YEAR)


def rel_label(year: int, origin: int) -> str:
    """Period label relative to ``origin`` (``t-31`` .. ``t0`` .. ``t+4``): the calendar year is never shown."""
    r = int(year - origin)
    return "t0" if r == 0 else f"t{r:+d}"


def _anon_ids(meta: dict[str, dict[str, str]], partition_seed: int) -> tuple[dict[str, str], dict[str, str]]:
    """Opaque, deterministic economy ids (over *all* non-aggregate economies, so eligibility is not leaked by
    numbering) and series-kind ids: target kinds ``T1``/``T2``, covariate kinds ``C1``/``C2``."""
    order = hash_order(sorted(meta), f"{CODE}|anon-econ|{partition_seed}")
    econ_ids = {c: f"E{i:03d}" for i, c in enumerate(order)}
    series_ids: dict[str, str] = {}
    for prefix, group in (("T", TARGETS), ("C", [i for i in INDICATORS if i not in TARGETS])):
        for j, ind in enumerate(hash_order(list(group), f"{CODE}|anon-series|{prefix}|{partition_seed}")):
            series_ids[ind] = f"{prefix}{j + 1}"
    return econ_ids, series_ids


def load_wdi(root: Path, partition_seed: int) -> WDIData:
    api = root / "for38-worldbank-wdi" / "source-api"
    countries = json.loads((api / "countries.json").read_text())[1]
    meta = {c["id"]: {"name": c["name"], "region": c["region"]["value"].strip(), "income": c["incomeLevel"]["value"].strip()}
            for c in countries if c["region"]["value"].strip() != "Aggregates"}
    values: dict[str, dict[str, np.ndarray]] = {}
    names: dict[str, str] = {}
    for ind in INDICATORS:
        recs = json.loads((api / f"{ind}.json").read_text())[1]
        per: dict[str, np.ndarray] = {}
        for r in recs:
            c = r.get("countryiso3code")
            if c not in meta:
                continue
            y = int(r["date"])
            if FIRST_YEAR <= y <= 2025:
                arr = per.setdefault(c, np.full(YEARS.size, np.nan))
                if r["value"] is not None:
                    arr[_yi(y)] = float(r["value"])
            names[ind] = r["indicator"]["value"]
        values[ind] = per
    items: dict[str, tuple[str, str]] = {}
    for ind in TARGETS:
        for c, v in sorted(values[ind].items()):
            hist, tgt = v[:_yi(ORIGIN) + 1], v[_yi(ORIGIN) + 1:_yi(ORIGIN) + 1 + H]
            fin = v[np.isfinite(v)]
            if (np.isfinite(hist[-1]) and np.isfinite(hist).sum() >= MIN_OBS and np.all(np.isfinite(tgt))
                    and fin.size and fin.min() > 0):
                items[f"WDI:{c}:{ind}:origin{ORIGIN}:h{H}"] = (c, ind)
    by_econ: dict[str, list[str]] = {}
    for iid, (c, _) in items.items():
        by_econ.setdefault(c, []).append(iid)
    iid_econ = sorted(c for c in by_econ if meta[c]["region"] in IID_REGIONS)
    ood_econ = sorted(c for c in by_econ if meta[c]["region"] in OOD_REGIONS)
    other = sorted(c for c in by_econ if c not in iid_econ and c not in ood_econ)
    if other:
        raise ValueError(f"economies in unmapped regions: {other}")
    pools: dict[str, list[str]] = {"val": [], "id": [], "src": []}
    for c in hash_order(iid_econ, f"{CODE}|econ|{partition_seed}"):
        key = "val" if len(pools["val"]) < N_VAL else "id" if len(pools["id"]) < N_ID else "src"
        pools[key].extend(sorted(by_econ[c]))
    pools["ood"] = [i for c in ood_econ for i in sorted(by_econ[c])]
    iid_all = sorted(c for c in meta if meta[c]["region"] in IID_REGIONS)
    econ_ids, series_ids = _anon_ids(meta, partition_seed)
    return WDIData(values=values, meta=meta, ind_names=names, items=items, pools=pools, iid_economies=iid_all,
                   econ_ids=econ_ids, series_ids=series_ids)


def naive(history: np.ndarray, horizon: int) -> np.ndarray:
    """Random-walk forecast from the last non-missing value."""
    x = np.asarray(history, float)
    fin = np.where(np.isfinite(x))[0]
    if fin.size == 0:
        raise ValueError("history has no observed value")
    return np.full(horizon, x[fin[-1]])


def damped_trend(history: np.ndarray, horizon: int, *, window: int = 8, damping: float = 0.85) -> np.ndarray:
    """A deterministic, history-only damped linear-trend forecast.

    This is a trusted-side reference diagnostic, not the official reference and not an acceptance shortcut.  It
    uses only the last ``window`` finite history values, so it remains valid for the anonymised held-out inputs and
    cannot inspect the post-origin targets.  Damping prevents a short noisy trend from growing without bound over
    the four-step horizon.
    """
    x = np.asarray(history, float)
    if x.ndim != 1 or horizon < 1:
        raise ValueError("history must be one-dimensional and horizon must be positive")
    fin = x[np.isfinite(x)]
    if fin.size == 0:
        raise ValueError("history has no observed value")
    if fin.size == 1:
        return np.full(horizon, fin[-1], dtype=float)
    y = fin[-max(2, int(window)):]
    t = np.arange(y.size, dtype=float)
    slope = float(np.polyfit(t, y, 1)[0])
    steps = np.arange(1, horizon + 1, dtype=float)
    # Sum of the damped one-step increments: at damping=1 this tends to a linear trend.
    growth = (1.0 - float(damping) ** steps) / (1.0 - float(damping))
    return y[-1] + slope * growth


def history_backtest_forecast(history: np.ndarray, horizon: int, *, window: int = 8) -> np.ndarray:
    """Select random-walk versus damped-trend using only rolling backtests inside ``history``.

    The selection targets the same sMAPE horizon as the task, but every validation target is an earlier value in
    the visible history.  This gives a stronger, still label-free reference without using any post-origin value.
    """
    x = np.asarray(history, float)
    fin = x[np.isfinite(x)]
    if fin.size <= horizon + 2:
        return damped_trend(x, horizon, window=window)
    scores = [[], []]
    first = max(8, horizon + 2)
    for end in range(first, fin.size - horizon + 1):
        train, target = fin[:end], fin[end:end + horizon]
        for j, pred in enumerate((naive(train, horizon), damped_trend(train, horizon, window=window))):
            scores[j].append(smape(target, pred))
    mean = [float(np.mean(s)) for s in scores]
    # Ties prefer the random walk because it has no trend extrapolation risk.
    return damped_trend(x, horizon, window=window) if mean[1] < mean[0] else naive(x, horizon)


class Adapter(ForecastAdapterBase):
    """ScienceClaw-Eval FoR38 adapter (World Bank WDI macro forecasting, mean sMAPE)."""

    code = CODE
    name = "World Bank WDI macro forecasting"
    task_type = "time_series_forecasting"

    def _check_files(self) -> list[str]:
        api = self.data_root / "for38-worldbank-wdi" / "source-api"
        need = [api / "countries.json"] + [api / f"{i}.json" for i in INDICATORS]
        return [f"missing {p}" for p in need if not p.is_file()]

    def data(self) -> WDIData:
        return self._memo.get("data", lambda: load_wdi(self.data_root, self.partition_seed))

    def _split_pools(self) -> dict[str, list[PoolItem]]:
        d = self.data()
        return {k: [PoolItem(i, d.items[i][0]) for i in ids] for k, ids in d.pools.items()}

    def _panel(self) -> pd.DataFrame:
        def build() -> pd.DataFrame:
            d = self.data()
            rows = []
            for c in d.iid_economies:
                for ind in INDICATORS:
                    v = d.values[ind].get(c, np.full(YEARS.size, np.nan))[:_yi(ORIGIN) + 1]
                    rows.append({"economy_id": d.econ_ids[c], "region": d.meta[c]["region"],
                                 "income_level": d.meta[c]["income"], "indicator": d.series_ids[ind],
                                 **{rel_label(y, ORIGIN): float(v[_yi(y)]) for y in range(FIRST_YEAR, ORIGIN + 1)}})
            df = pd.DataFrame(rows)
            # opaque-id order: neither the row order nor the ids reveal the alphabetical (ISO3) order
            return df.sort_values(["economy_id", "indicator"], kind="stable").reset_index(drop=True)
        return self._memo.get("panel", build)

    # ----------------------------------------------------------------------------------------------
    def _make_episode(self, split: str, k: int, items: list[PoolItem], seed: int, reused: bool) -> Episode:
        d = self.data()
        ids = [it.id for it in items]
        n = len(ids)
        econ = [d.items[i][0] for i in ids]
        inds = [d.items[i][1] for i in ids]
        full = np.stack([d.values[inds[j]][econ[j]] for j in range(n)])
        hist = full[:, :_yi(ORIGIN) + 1]
        tgt = full[:, _yi(ORIGIN) + 1:_yi(ORIGIN) + 1 + H]
        cov = np.stack([np.stack([d.values[ind].get(c, np.full(YEARS.size, np.nan))[:_yi(ORIGIN) + 1]
                                  for ind in INDICATORS]) for c in econ])
        ref_pred = np.stack([naive(h, H) for h in hist])
        ref_per = [smape(tgt[j], ref_pred[j]) for j in range(n)]
        ref_score = float(np.mean(ref_per))
        # History-only references are retained as diagnostics.  They never change the official naive reference or
        # acceptance margin, and they are not returned by any visible tool.
        trend_pred = np.stack([damped_trend(h, H) for h in hist])
        adaptive_pred = np.stack([history_backtest_forecast(h, H) for h in hist])
        # Enforce the same physical plausibility bounds used for submitted forecasts before comparing the reference.
        hmax = np.nanmax(hist, axis=1)
        trend_pred = np.maximum(trend_pred, 0.0)
        for j, ind in enumerate(inds):
            trend_pred[j] = np.minimum(trend_pred[j], 100.0 if ind == "SL.UEM.TOTL.ZS" else 10.0 * hmax[j])
            adaptive_pred[j] = np.minimum(np.maximum(adaptive_pred[j], 0.0),
                                          100.0 if ind == "SL.UEM.TOTL.ZS" else 10.0 * hmax[j])
        trend_per = [smape(tgt[j], trend_pred[j]) for j in range(n)]
        adaptive_per = [smape(tgt[j], adaptive_pred[j]) for j in range(n)]
        trend_score = float(np.mean(trend_per))
        adaptive_score = float(np.mean(adaptive_per))
        ref_payload = {"smape": ref_per, "reference_smape": ref_per, "trend_smape": trend_per,
                       "strong_smape": adaptive_per,
                       "ids": ids, "indicator": inds}
        pool = "ood" if split == "ood" else "iid"
        # periods relative to the origin (t0 = last observed period); calendar years are never shown
        offsets = [y - ORIGIN for y in range(FIRST_YEAR, ORIGIN + 1)]
        dev_offsets = [y - DEV_ORIGIN for y in range(FIRST_YEAR, DEV_ORIGIN + 1)]
        target_offsets = list(range(1, H + 1))
        span, dev_span = len(offsets), len(dev_offsets)
        sid = d.series_ids
        kinds_doc = {sid[i]: KIND_DOC[i] for i in INDICATORS}
        t_level, t_rate = sid["NY.GDP.PCAP.KD"], sid["SL.UEM.TOTL.ZS"]
        cov_codes = [sid[i] for i in INDICATORS]
        dev_hist = hist[:, :_yi(DEV_ORIGIN) + 1]
        dev_tgt = hist[:, _yi(DEV_ORIGIN) + 1:]
        dev_ok = np.all(np.isfinite(dev_tgt), axis=1) & np.array([np.isfinite(h).any() for h in dev_hist])
        dev_ref = float(np.mean([smape(dev_tgt[j], naive(dev_hist[j], H)) for j in range(n) if dev_ok[j]])) \
            if dev_ok.any() else float("nan")
        cmeta = {"economy_id": [d.econ_ids[c] for c in econ],
                 "region": [d.meta[c]["region"] for c in econ], "income_level": [d.meta[c]["income"] for c in econ],
                 "indicator": [sid[i] for i in inds]}
        panel = self._panel()

        # ------------------------------------------------------------------ D_E tools
        def load_train(inputs: dict, config: dict) -> dict:
            return {"panel": panel.copy(), "indicator_kinds": dict(kinds_doc)}

        def load_dev(inputs: dict, config: dict) -> dict:
            return {"history": dev_hist.copy(), "time_offsets": list(dev_offsets),
                    "target_offsets": list(target_offsets), **{k_: list(v) for k_, v in cmeta.items()}}

        def score_dev(inputs: dict, config: dict) -> dict:
            pred = to_float_array(inputs.get("pred"))
            if pred.shape != (n, H):
                raise ValueError(f"pred must have shape ({n}, {H}), got {tuple(pred.shape)}")
            if not np.all(np.isfinite(pred)):
                raise ValueError("pred contains non-finite values")
            per = [smape(dev_tgt[j], pred[j]) if dev_ok[j] else None for j in range(n)]
            vals = [v for v in per if v is not None]
            s = float(np.mean(vals)) if vals else float("nan")
            return {"score": s, "report": {"metric": "mean sMAPE (%) on the dev backtest, periods t+1..t+4 after the "
                                                     "dev origin (lower is better); items with a missing target "
                                                     "value are skipped",
                                           "score": s, "reference_score": dev_ref, "per_item": per,
                                           "n_scored": len(vals)}}

        def load_eval_inputs(inputs: dict, config: dict) -> dict:
            return {"history": hist.copy(), "time_offsets": list(offsets), "target_offsets": list(target_offsets),
                    "covariates": cov.copy(), "covariate_indicators": list(cov_codes),
                    **{k_: list(v) for k_, v in cmeta.items()}}

        def macro_fixed_predict(inputs: dict, config: dict) -> dict:
            pred, info = _fixed_macro_predict(inputs)
            return {"y": pred, "provenance": info}

        def chronos_fixed_predict(inputs: dict, config: dict) -> dict:
            pred, info = _fixed_chronos_predict(inputs)
            return {"y": pred, "provenance": info}

        lst = PortSchema("list", (n,), None, "str")
        meta_ports = {"economy_id": lst, "region": lst, "income_level": lst, "indicator": lst}
        first_col, last_col = rel_label(FIRST_YEAR, ORIGIN), rel_label(ORIGIN, ORIGIN)
        tools = [
            ToolSpec("load_train",
                     "Visible panel of annual series: one row per (economy, indicator kind) for every economy of the "
                     "source regions (Europe & Central Asia, East Asia & Pacific, Latin America & Caribbean, North "
                     f"America) and the four indicator kinds; period columns {first_col}..{last_col} (t0 = the last "
                     "observed period) hold annual values (NaN = missing). indicator_kinds maps each kind to a short "
                     "description. Economies are opaque ids; no value after t0 is included.",
                     {}, {"panel": PortSchema("table", (len(panel), len(panel.columns)), UNIT, None,
                                              "columns economy_id, region, income_level, indicator, "
                                              f"'{first_col}'..'{last_col}'"),
                          "indicator_kinds": PortSchema("dict")},
                     load_train),
            ToolSpec("load_dev",
                     f"Dev backtest inputs: each evaluation item's history over the periods {dev_offsets[0]}..0 "
                     f"relative to the dev origin (item order); score_dev holds the next {H} periods.",
                     {}, {"history": PortSchema("array", (n, dev_span), UNIT, "float"),
                          "time_offsets": PortSchema("list", (dev_span,), "period", "int"),
                          "target_offsets": PortSchema("list", (H,), "period", "int"), **meta_ports},
                     load_dev),
            ToolSpec("score_dev",
                     f"Scores a forecast of the dev backtest (row i = item i, columns = periods +1..+{H} after the dev "
                     "origin): returns the mean sMAPE in % (lower is better) and, in report, the same statistic of "
                     "the reference forecast.",
                     {"pred": PortSchema("array", (n, H), UNIT, "float", "dev forecast, same layout as y")},
                     {"score": PortSchema("number", None, "%", "float"), "report": PortSchema("dict")},
                     score_dev),
            ToolSpec("load_eval_inputs",
                     f"Evaluation inputs (no targets) for the {n} items in item order: history[i] = annual values over "
                     f"the periods {offsets[0]}..0 relative to the origin t0 (NaN = missing) of the item's indicator "
                     "kind for its (opaque) economy; covariates[i, c] = the same economy's series of "
                     "covariate_indicators[c] over the same periods; coarse economy metadata (region, income level); "
                     f"target_offsets = 1..{H}.",
                     {}, {"history": PortSchema("array", (n, span), UNIT, "float"),
                          "time_offsets": PortSchema("list", (span,), "period", "int"),
                          "target_offsets": PortSchema("list", (H,), "period", "int"),
                          "covariates": PortSchema("array", (n, len(INDICATORS), span), UNIT, "float"),
                     "covariate_indicators": PortSchema("list", (len(INDICATORS),), None, "str"), **meta_ports},
                     load_eval_inputs),
            ToolSpec("macro_fixed_predict",
                     "Fixed visible-data macro predictor. Wire panel and indicator_kinds from load_train and all "
                     "label-free fields from load_eval_inputs. It runs the frozen scilib.macro route with equal "
                     f"weights over {', '.join(MACRO_TOOL_MEMBERS)}, n_backtest=0, seed=0, n_jobs=1; it reads no "
                     "post-origin values and returns submit-ready y plus provenance.",
                     {"panel": PortSchema("table", (len(panel), len(panel.columns)), UNIT, None),
                      "indicator_kinds": PortSchema("dict"),
                      "history": PortSchema("array", (n, span), UNIT, "float"),
                      "target_offsets": PortSchema("list", (H,), "period", "int"),
                      "covariates": PortSchema("array", (n, len(INDICATORS), span), UNIT, "float"),
                      "covariate_indicators": PortSchema("list", (len(INDICATORS),), None, "str"),
                      **meta_ports},
                     {"y": PortSchema("array", (n, H), UNIT, "float"),
                      "provenance": PortSchema("dict")}, macro_fixed_predict,
                     config_doc=("members=" + ",".join(MACRO_TOOL_MEMBERS) + "; weights=equal; "
                                 "n_backtest=0; seed=0; n_jobs=1")),
            ToolSpec("chronos_fixed_predict",
                     "Fixed pretrained Chronos-2 median forecast. Wire history, indicator, indicator_kinds and "
                     "target_offsets from load_eval_inputs/load_train; it runs the staged Chronos-2 weights with "
                     f"context_length={CHRONOS_TOOL_CONTEXT}, quantile=0.5, and returns submit-ready y plus "
                     "provenance. It reads no target values or scorer output.",
                     {"history": PortSchema("array", (n, span), UNIT, "float"),
                      "indicator": lst, "indicator_kinds": PortSchema("dict"),
                      "target_offsets": PortSchema("list", (H,), "period", "int")},
                     {"y": PortSchema("array", (n, H), UNIT, "float"),
                      "provenance": PortSchema("dict")}, chronos_fixed_predict,
                     config_doc=f"model={CHRONOS_TOOL_MODEL}; context_length={CHRONOS_TOOL_CONTEXT}; quantile=0.5"),
        ]

        # ------------------------------------------------------------------ D_V
        hmax = np.nanmax(hist, axis=1)
        is_unemp = np.array([i == "SL.UEM.TOTL.ZS" for i in inds])

        def upper(arr: np.ndarray) -> np.ndarray:
            if arr.shape != (n, H):
                raise ValueError(f"shape {tuple(arr.shape)} != ({n}, {H})")
            return np.where(is_unemp, 100.0, 10.0 * hmax)[:, None]

        constraints = [c_shape((n, H)), c_finite(),
                       c_range(0.0, None, UNIT, upper_fn=upper,
                               upper_doc=f"rows of kind {t_rate} <= 100, rows of kind {t_level} <= 10 x max(history)",
                               desc=f"all values >= 0; rows of kind {t_rate} (a percentage) <= 100; rows of kind "
                                    f"{t_level} <= 10 x their historical maximum (scale guard)"),
                       c_declared_unit(UNIT)]

        def evaluate(y: Any, trace: Trace | None) -> EvalResult:
            common = {"reference": ref_score, "direction": "min", "margin": ACCEPT_MARGIN,
                      "reference_payload": ref_payload}
            base_m = {"reference_mean_smape_pct": ref_score}
            try:
                arr = to_float_array(y)
            except ValueError as ex:
                return invalid_eval(str(ex), metrics=base_m, **common)
            if arr.shape != (n, H):
                return invalid_eval(f"shape {tuple(arr.shape)} != ({n}, {H})", metrics=base_m, **common)
            if not np.all(np.isfinite(arr)):
                return invalid_eval("non-finite predictions", metrics=base_m, **common)
            per = [smape(tgt[j], arr[j]) for j in range(n)]
            score = float(np.mean(per))
            metrics = {"mean_smape_pct": score, "damped_trend_reference_mean_smape_pct": trend_score,
                       "strong_reference_mean_smape_pct": adaptive_score, **base_m}
            for ind, key in (("NY.GDP.PCAP.KD", "smape_gdp_per_capita_pct"), ("SL.UEM.TOTL.ZS", "smape_unemployment_pct")):
                sel = [p for p, i in zip(per, inds) if i == ind]
                if sel:
                    metrics[key] = float(np.mean(sel))
            return finish_eval(primary=score, metrics=metrics,
                               payload={"smape": per, "reference_smape": ref_per, "trend_smape": trend_per,
                                        "strong_smape": adaptive_per,
                                        "ids": ids, "indicator": inds},
                               **common)

        objective = (
            "Economics / macroeconomic forecasting of annual country-level series (anonymised: economies are opaque "
            "ids, indicators are series kinds, time is given as periods relative to the forecast origin t0). "
            f"There are {n} evaluation items; each is one (economy, indicator) annual series of kind {t_level} = "
            f"{kinds_doc[t_level]} or {t_rate} = {kinds_doc[t_rate]}. Input of item i (tool load_eval_inputs): "
            f"history[i], the values over the periods {offsets[0]}..0 relative to t0 (NaN = not reported), the "
            f"economy's series of the four kinds over the same periods (covariates; {cov_codes[2]} and "
            f"{cov_codes[3]} are annual percentage changes) and coarse economy metadata (region, income level). "
            f"Target of item i: the reported values of the next {H} periods t0+1..t0+{H}. No value after t0 is "
            "available from any tool.\n"
            f"Deliverable y: a float array of shape ({n}, {H}) in each item's own unit; row i is the forecast for "
            f"item i in the order of load_eval_inputs, column h (0-based) is the forecast for period t0+1+h. "
            f"Values must be finite and >= 0 (rows of kind {t_rate} <= 100).\n"
            "Score (lower is better): mean over items of sMAPE (%) = 200/4 * sum_h |y_h - ŷ_h| / (|y_h| + |ŷ_h|).\n"
            "For a fixed no-label route, wire load_train.panel and indicator_kinds plus all label-free fields from "
            "load_eval_inputs to macro_fixed_predict; its y output is submit-ready and provenance fixes the member "
            f"set ({', '.join(MACRO_TOOL_MEMBERS)}), equal weights, seed=0 and n_backtest=0.\n"
            "A pretrained comparison route is also available: chronos_fixed_predict consumes only the history, "
            "indicator kinds and target offsets from load_eval_inputs, runs frozen Chronos-2 median forecasts, and "
            "returns submit-ready y with provenance.\n"
            + scilib.describe("macro")
        )
        lineage = common_lineage(
            dataset=DATASET, version=VERSION, source=SOURCE, license_=LICENSE, pool=pool,
            ood_kind="proxy_within_dataset" if pool == "ood" else None, items=items, seed=seed,
            partition_seed=self.partition_seed, k=k, reused=reused,
            split_rule=("economy-disjoint: IID regions " + ", ".join(IID_REGIONS) + "; economies hash-ordered, "
                        f"whole economies to val (>= {N_VAL} items), id (>= {N_ID}), rest src; OOD regions "
                        + ", ".join(OOD_REGIONS)),
            visible=f"panel {FIRST_YEAR}-{ORIGIN} of IID-region economies (4 indicators); per-item history + covariates "
                    f"{FIRST_YEAR}-{ORIGIN}; dev backtest {DEV_ORIGIN + 1}-{ORIGIN}",
            extra={"origin_year": ORIGIN, "horizon_years": H, "target_indicators": list(TARGETS),
                   "covariate_indicators": list(INDICATORS),
                   "anonymised_view": {"economy_ids": "E000.. (partition-seed hash permutation of all non-aggregate "
                                                      "economies)", "series_kinds": dict(sid),
                                       "time": "periods relative to the origin (t-31..t0, targets t+1..t+4)"}})
        return Episode(
            id=episode_id(CODE, split, seed, k), discipline=CODE, family=self.family, split=split,
            task_type=self.task_type, objective=objective,
            required_output=PortSchema("array", (n, H), UNIT, "float", f"annual forecasts for periods t0+1..t0+{H}"),
            tools=tools, constraints=constraints, budget=default_budget(max_node_s=300.0, max_llm_items=2 * n),
            lineage=lineage,
            acceptance=(f"accepted iff mean sMAPE <= {1 - ACCEPT_MARGIN:.2f} x the reference mean sMAPE on the same "
                        "items (reference recipe and value only in EvalResult.details / docs, never shown to the "
                        "policy)"),
            tolerance={"rtol": 1e-5, "atol": 1e-6},
            tags=[CODE, self.family, "time-series", "forecasting", "macroeconomics", "annual", "panel", "sMAPE"],
            metric=self.metric, direction="min", n_items=n, _evaluate=evaluate, _dev_evaluate=None)

    # ----------------------------------------------------------------------------------------------
    def pooled_metric(self, per_episode: list[dict]) -> float | None:
        """Mean sMAPE (%) over all items of the given episodes."""
        vals = [float(v) for p in unwrap_payloads(per_episode) for v in p.get("smape", [])]
        return float(np.mean(vals)) if vals else None

    def pooled_diagnostics(self, per_episode: list[dict]) -> dict[str, Any]:
        """Aggregate official and stronger reference scores over complete episode payloads.

        ``pooled_metric`` remains the sole pooled primary.  The extra fields make the full-split comparison
        explicit: the official random-walk reference and a deterministic damped-trend reference are both pooled
        over items, rather than comparing a 16-item episode score with an external leaderboard number.
        """
        payloads = unwrap_payloads(per_episode)
        primary = [float(v) for p in payloads for v in p.get("smape", [])]
        reference = [float(v) for p in payloads for v in p.get("reference_smape", p.get("smape", []))]
        strong = [float(v) for p in payloads for v in p.get("strong_smape", [])]
        out: dict[str, Any] = {
            "n_episodes": len(payloads),
            "n_items": len(primary),
            "pooled_primary": float(np.mean(primary)) if primary else None,
            "pooled_reference": float(np.mean(reference)) if reference else None,
            "pooled_strong_reference": float(np.mean(strong)) if strong else None,
        }
        if out["pooled_reference"] is not None and out["pooled_strong_reference"] is not None:
            out["strong_reference_gain_pct"] = float(out["pooled_reference"] - out["pooled_strong_reference"])
        return out

    def full_split_reference(self, split: str) -> dict[str, Any]:
        """Compute history-only reference diagnostics over every item in ``split``.

        Episode diagnostics are useful for catching a weak reference, but their 16-item samples are too noisy for
        comparison with published macro benchmarks.  This helper applies the exact same target, plausibility cap,
        and sMAPE definitions as ``_make_episode`` to the complete split pool.  It is trusted-side evidence only:
        it reads held-out targets for reporting, never changes ``evaluate``/acceptance, and exposes no values to the
        policy.  ``naive`` is the frozen official reference; ``damped_trend`` and ``history_backtest_forecast``
        are history-only comparators.
        """
        check_split(split)
        d = self.data()
        ids = list(d.pools[split])
        if not ids:
            return {"split": split, "n_items": 0, "reference_smape": None,
                    "damped_trend_smape": None, "history_backtest_smape": None,
                    "diagnostic_only": True}
        econ = [d.items[i][0] for i in ids]
        inds = [d.items[i][1] for i in ids]
        full = np.stack([d.values[inds[j]][econ[j]] for j in range(len(ids))])
        hist = full[:, :_yi(ORIGIN) + 1]
        tgt = full[:, _yi(ORIGIN) + 1:_yi(ORIGIN) + 1 + H]
        ref_pred = np.stack([naive(h, H) for h in hist])
        trend_pred = np.stack([damped_trend(h, H) for h in hist])
        adaptive_pred = np.stack([history_backtest_forecast(h, H) for h in hist])
        hmax = np.nanmax(hist, axis=1)
        trend_pred = np.maximum(trend_pred, 0.0)
        adaptive_pred = np.maximum(adaptive_pred, 0.0)
        for j, ind in enumerate(inds):
            cap = 100.0 if ind == "SL.UEM.TOTL.ZS" else 10.0 * hmax[j]
            trend_pred[j] = np.minimum(trend_pred[j], cap)
            adaptive_pred[j] = np.minimum(adaptive_pred[j], cap)
        ref = np.asarray([smape(tgt[j], ref_pred[j]) for j in range(len(ids))], dtype=float)
        trend = np.asarray([smape(tgt[j], trend_pred[j]) for j in range(len(ids))], dtype=float)
        adaptive = np.asarray([smape(tgt[j], adaptive_pred[j]) for j in range(len(ids))], dtype=float)
        ref_mean, trend_mean, adaptive_mean = map(float, (np.mean(ref), np.mean(trend), np.mean(adaptive)))
        return {"split": split, "n_items": len(ids), "reference_smape": ref_mean,
                "damped_trend_smape": trend_mean, "history_backtest_smape": adaptive_mean,
                "damped_trend_gain_pct": float(ref_mean - trend_mean),
                "history_backtest_gain_pct": float(ref_mean - adaptive_mean),
                "diagnostic_only": True}
