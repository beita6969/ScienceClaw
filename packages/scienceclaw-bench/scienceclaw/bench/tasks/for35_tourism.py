"""FoR35 Commerce, management, tourism and services — Monash Tourism Monthly forecasting (mean MASE).

Data (read-only): ``<data_root>/for35-monash-tourism-monthly/data/tourism_monthly_dataset.tsf`` — the Monash
Time Series Forecasting Archive release of the Kaggle/IJF tourism competition monthly data (Athanasopoulos et al.
2011), Zenodo record 4656096 v3: 366 monthly series, forecast horizon 24.

Item = one series: input its history (all but the last 24 observations, the Monash train part), target the last
24 observations (the Monash test part).

Pools / splits (series-disjoint; the partition is fixed by ``partition_seed`` and never depends on the seed):

* IID pool: the 264 series of the dominant start cohort (series starting in 1980). Hash-ordered with the partition
  seed: 32 -> ``val``, 64 -> ``id``, the remaining 168 -> ``src``.
* OOD pool:
  - if a Monash **tourism quarterly** TSF is present locally (``for35-*/**/tourism_quarterly*.tsf`` or
    ``*tourism*quarterly*/**/*.tsf`` under the data root) its 427 series are the OOD pool (horizon 8, period 4;
    ``ood_kind = "cross_dataset"``);
  - otherwise (the case on 2026-09-28) the 102 monthly series of the *other* start cohorts (1979, 1981, 1985,
    1986, 1991, 2000: different providers / shorter histories) -> ``ood_kind = "proxy_within_dataset"``.

Visible data: histories of all 264 IID series (``load_train``; never a target value), the items' own histories
(``load_eval_inputs``) and a dev backtest on the items' histories (``load_dev`` / ``score_dev``: forecast the last
H observations of each history from the part before them).

Metric (lower is better): mean over items of MASE (Hyndman & Koehler 2006, Monash archive convention) with the
seasonal scale computed on the item's history: ``MASE = mean_h |y_h - ŷ_h| / mean_{t>m} |x_t - x_{t-m}|``,
m = 12 (monthly) / 4 (quarterly). Reference: seasonal naive ``ŷ_h = x_{T - m + ((h - 1) mod m) + 1}`` (Monash
SNaive: mean MASE 1.631 on all 366 series, reproduced by this code). Acceptance: mean MASE <= 0.95 x reference.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import scilib
from ...core.schema import PortSchema
from ...core.trace import Trace
from ..task import EvalResult, Episode, ToolSpec
from ._forecast_common import (ForecastAdapterBase, PoolItem, c_declared_unit, c_finite, c_range, c_shape,
                               common_lineage, default_budget, episode_id, finish_eval, hash_order, invalid_eval,
                               mase, smape, to_float_array, unwrap_payloads)

CODE = "FoR35"
ACCEPT_MARGIN = 0.05
IID_START_YEAR = 1980
N_VAL, N_ID = 32, 64
UNIT = "native"          # the series' own unit (tourism counts as recorded by the provider)

DATASET = "Monash Tourism Monthly (tourism_monthly_dataset.tsf)"
VERSION = "Zenodo 4656096 / version 3"
SOURCE = "https://zenodo.org/records/4656096"
LICENSE = "CC BY 4.0 (Monash Time Series Forecasting Archive, Zenodo 4656096)"
QUARTERLY_SOURCE = "https://zenodo.org/records/4656093 (tourism_quarterly_dataset.zip)"


@dataclass(frozen=True)
class Series:
    id: str
    start: str          # ISO date of the first observation
    freq: str           # "monthly" | "quarterly"
    horizon: int
    period: int
    values: np.ndarray  # full series (history + target)

    @property
    def history(self) -> np.ndarray:
        return self.values[:-self.horizon]

    @property
    def target(self) -> np.ndarray:
        return self.values[-self.horizon:]


def read_tsf(path: Path, prefix: str) -> list[Series]:
    """Parse a Monash .tsf file (attributes series_name, start_timestamp; @frequency; @horizon)."""
    freq, horizon = "", 0
    out: list[Series] = []
    in_data = False
    with path.open(encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not in_data:
                low = line.lower()
                if low.startswith("@frequency"):
                    freq = line.split()[1].strip()
                elif low.startswith("@horizon"):
                    horizon = int(line.split()[1])
                elif low.startswith("@data"):
                    in_data = True
                continue
            name, start, vals = line.split(":", 2)
            v = np.array([float("nan") if x.strip() == "?" else float(x) for x in vals.split(",")], dtype=float)
            out.append(Series(f"{prefix}:{name}", start.split(" ")[0], freq, horizon, 12 if freq == "monthly" else 4, v))
    if freq not in ("monthly", "quarterly") or horizon <= 0:
        raise ValueError(f"{path.name}: unsupported @frequency {freq!r} / @horizon {horizon}")
    return out


def snaive(history: np.ndarray, horizon: int, m: int) -> np.ndarray:
    """Seasonal naive: repeat the last observed season."""
    x = np.asarray(history, float)
    return np.array([x[len(x) - m + (h % m)] for h in range(horizon)])


@dataclass
class TourismData:
    series: dict[str, Series]
    pools: dict[str, list[str]]
    iid_ids: list[str]
    ood_kind: str
    ood_source: str


class Adapter(ForecastAdapterBase):
    """ScienceClaw-Eval FoR35 adapter (Monash tourism forecasting, mean MASE)."""

    code = CODE
    name = "Monash Tourism Monthly"
    task_type = "time_series_forecasting"

    def _monthly_path(self) -> Path:
        return self.data_root / "for35-monash-tourism-monthly" / "data" / "tourism_monthly_dataset.tsf"

    def _quarterly_path(self) -> Path | None:
        pats = ["for35-*/**/tourism_quarterly*.tsf", "*tourism*quarterly*/**/*.tsf"]
        for pat in pats:
            hits = sorted(self.data_root.glob(pat))
            if hits:
                return hits[0]
        return None

    def _check_files(self) -> list[str]:
        p = self._monthly_path()
        return [] if p.is_file() else [f"missing {p}"]

    def data(self) -> TourismData:
        return self._memo.get("data", self._load)

    def _load(self) -> TourismData:
        monthly = read_tsf(self._monthly_path(), "tourism_monthly")
        for s in monthly:
            if len(s.history) <= 2 * s.period or not np.all(np.isfinite(s.values)):
                raise ValueError(f"series {s.id} too short or has missing values")
        iid = [s for s in monthly if int(s.start[:4]) == IID_START_YEAR]
        order = hash_order([s.id for s in iid], f"{CODE}|iid|{self.partition_seed}")
        pools = {"val": order[:N_VAL], "id": order[N_VAL:N_VAL + N_ID], "src": order[N_VAL + N_ID:]}
        series = {s.id: s for s in monthly}
        qp = self._quarterly_path()
        if qp is not None:
            quarterly = read_tsf(qp, "tourism_quarterly")
            quarterly = [s for s in quarterly if np.all(np.isfinite(s.values)) and len(s.history) > 2 * s.period]
            series.update({s.id: s for s in quarterly})
            pools["ood"] = [s.id for s in quarterly]
            kind, src = "cross_dataset", f"{qp} ({QUARTERLY_SOURCE})"
        else:
            pools["ood"] = [s.id for s in monthly if int(s.start[:4]) != IID_START_YEAR]
            kind, src = "proxy_within_dataset", ("tourism monthly series of the non-1980 start cohorts (1979, 1981, "
                                                 "1985, 1986, 1991, 2000); tourism quarterly not present locally")
        for sid in (i for ids in pools.values() for i in ids):      # MASE scale must be defined
            s = series[sid]
            h = s.history
            if not np.mean(np.abs(h[s.period:] - h[:-s.period])) > 0:
                raise ValueError(f"{sid}: zero seasonal-difference scale")
        return TourismData(series=series, pools=pools, iid_ids=[s.id for s in iid], ood_kind=kind, ood_source=src)

    def _split_pools(self) -> dict[str, list[PoolItem]]:
        d = self.data()
        return {k: [PoolItem(i, i) for i in ids] for k, ids in d.pools.items()}

    # ----------------------------------------------------------------------------------------------
    def _make_episode(self, split: str, k: int, items: list[PoolItem], seed: int, reused: bool) -> Episode:
        d = self.data()
        ss = [d.series[it.id] for it in items]
        n = len(ss)
        H, m, freq = ss[0].horizon, ss[0].period, ss[0].freq
        if any(s.horizon != H or s.period != m for s in ss):
            raise ValueError("mixed horizons within one episode")
        ids = [s.id for s in ss]
        hist = [s.history.copy() for s in ss]
        tgt = np.stack([s.target for s in ss])
        ref_pred = np.stack([snaive(h, H, m) for h in hist])
        ref_mase = [mase(tgt[i], ref_pred[i], hist[i], m) for i in range(n)]
        ref_score = float(np.mean(ref_mase))
        ref_payload = {"mase": ref_mase, "ids": ids}
        pool = "ood" if split == "ood" else "iid"

        # ------------------------------------------------------------------ D_E tools
        train_ids = list(d.iid_ids)
        train_hist = [d.series[i].history.copy() for i in train_ids]
        train_start = [d.series[i].start for i in train_ids]
        dev_hist = [h[:-H].copy() for h in hist]
        dev_tgt = np.stack([h[-H:] for h in hist])
        dev_ref = float(np.mean([mase(dev_tgt[i], snaive(dev_hist[i], H, m), dev_hist[i], m) for i in range(n)]))
        starts = [s.start for s in ss]

        def load_train(inputs: dict, config: dict) -> dict:
            return {"series": [h.copy() for h in train_hist], "series_id": list(train_ids), "start": list(train_start)}

        def load_dev(inputs: dict, config: dict) -> dict:
            return {"history": [h.copy() for h in dev_hist], "series_id": list(ids), "start": list(starts)}

        def score_dev(inputs: dict, config: dict) -> dict:
            pred = to_float_array(inputs.get("pred"))
            if pred.shape != (n, H):
                raise ValueError(f"pred must have shape ({n}, {H}), got {tuple(pred.shape)}")
            if not np.all(np.isfinite(pred)):
                raise ValueError("pred contains non-finite values")
            per = [mase(dev_tgt[i], pred[i], dev_hist[i], m) for i in range(n)]
            s = float(np.mean(per))
            return {"score": s, "report": {"metric": "mean MASE on the dev backtest (lower is better)", "score": s,
                                           "reference_score": dev_ref, "per_item": per, "n_items": n}}

        def load_eval_inputs(inputs: dict, config: dict) -> dict:
            return {"history": [h.copy() for h in hist], "series_id": list(ids), "start": list(starts),
                    "horizon": H, "period": m}

        list_n = PortSchema("list", (n,), None, "str")
        tools = [
            ToolSpec("load_eval_inputs",
                     f"Evaluation inputs (no targets) for the {n} items in item order; the deliverable y is a forecast of "
                     f"these histories. history[i] = all observed {freq} values of series i up to its forecast origin "
                     f"(oldest first), series_id, start (ISO date of the first observation), horizon = {H}, period = {m} "
                     "(observations per year).",
                     {}, {"history": PortSchema("list", (n,), UNIT, "float", "list of 1-D float arrays"),
                          "series_id": list_n, "start": list_n,
                          "horizon": PortSchema("number", None, None, "int"),
                          "period": PortSchema("number", None, None, "int")},
                     load_eval_inputs),
            ToolSpec("load_train",
                     f"Visible training data: the histories (all observations before each series' forecast origin) "
                     f"of the {len(train_ids)} monthly tourism series of the source collection, with series ids and "
                     "ISO start dates (first observation). No forecast-period values are included.",
                     {}, {"series": PortSchema("list", (len(train_ids),), UNIT, "float",
                                               "list of 1-D float arrays (monthly observations, oldest first)"),
                          "series_id": PortSchema("list", (len(train_ids),), None, "str"),
                          "start": PortSchema("list", (len(train_ids),), None, "str")},
                     load_train),
            ToolSpec("load_dev",
                     f"Dev backtest inputs (not the inputs of the deliverable): for every evaluation item, its history "
                     f"without the last {H} observations (same item order); score_dev holds those last {H} observations. "
                     f"Series of the source collection appear in load_train with all their observations, including these "
                     f"last {H}.",
                     {}, {"history": PortSchema("list", (n,), UNIT, "float", "list of 1-D float arrays"),
                          "series_id": list_n, "start": list_n},
                     load_dev),
            ToolSpec("score_dev",
                     "Scores a forecast of the dev backtest (a forecast for the load_dev histories, not for the "
                     "load_eval_inputs histories): returns the mean MASE (lower is better) and, in report, "
                     "the same statistic of the reference forecast.",
                     {"pred": PortSchema("array", (n, H), UNIT, "float", "dev forecast, same layout as y")},
                     {"score": PortSchema("number", None, "1", "float"), "report": PortSchema("dict")},
                     score_dev),
        ]

        # ------------------------------------------------------------------ D_V
        hist_max = np.array([np.max(h) for h in hist])

        def upper(arr: np.ndarray) -> np.ndarray:
            if arr.shape != (n, H):
                raise ValueError(f"shape {tuple(arr.shape)} != ({n}, {H})")
            return np.maximum(10.0 * hist_max, 1.0)[:, None]

        constraints = [c_shape((n, H)), c_finite(),
                       c_range(0.0, None, UNIT, upper_fn=upper,
                               upper_doc="y[i, h] <= 10 * max(history[i]) (scale guard)",
                               desc="tourism quantities are non-negative (y >= 0) and y[i, h] <= 10 * max(history[i])"),
                       c_declared_unit(UNIT)]

        def evaluate(y: Any, trace: Trace | None) -> EvalResult:
            common = {"reference": ref_score, "direction": "min", "margin": ACCEPT_MARGIN,
                      "reference_payload": ref_payload}
            base_m = {"reference_mean_mase": ref_score}
            try:
                arr = to_float_array(y)
            except ValueError as ex:
                return invalid_eval(str(ex), metrics=base_m, **common)
            if arr.shape != (n, H):
                return invalid_eval(f"shape {tuple(arr.shape)} != ({n}, {H})", metrics=base_m, **common)
            if not np.all(np.isfinite(arr)):
                return invalid_eval("non-finite predictions", metrics=base_m, **common)
            per = [mase(tgt[i], arr[i], hist[i], m) for i in range(n)]
            score = float(np.mean(per))
            metrics = {"mean_mase": score, "median_mase": float(np.median(per)),
                       "mean_smape_pct": float(np.mean([smape(tgt[i], arr[i]) for i in range(n)])), **base_m}
            return finish_eval(primary=score, metrics=metrics, payload={"mase": per, "ids": ids},
                               extra={"per_item_mase": dict(zip(ids, per))}, **common)

        objective = (
            f"Tourism demand forecasting ({'Monash tourism quarterly' if freq == 'quarterly' else 'Monash Tourism Monthly'}"
            f", tourism forecasting competition data). There are {n} evaluation items; each is one {freq} tourism "
            f"time series. Input of item i (tool load_eval_inputs): history[i], all observed values of the series up "
            f"to its forecast origin (oldest first, no missing values), with its start date. Target of item i: the "
            f"next {H} {freq} values of the same series.\n"
            f"Deliverable y: a float array of shape ({n}, {H}) in the series' own units; row i is the forecast for "
            f"item i in the order of load_eval_inputs, column h (0-based) is the forecast {h_text(freq)} h+1 steps "
            "after the last history value. Values must be finite and >= 0.\n"
            f"Score (lower is better): mean over items of MASE = mean_h |y_h - ŷ_h| / (mean over t > {m} of "
            f"|x_t - x_(t-{m})| computed on history[i]).\n"
            + scilib.describe("forecast") + scilib.describe_extra("tsfm")
        )
        lineage = common_lineage(
            dataset=DATASET if pool == "iid" or d.ood_kind != "cross_dataset" else "Monash Tourism Quarterly",
            version=VERSION, source=SOURCE if pool == "iid" or d.ood_kind != "cross_dataset" else QUARTERLY_SOURCE,
            license_=LICENSE, pool=pool, ood_kind=d.ood_kind if pool == "ood" else None, items=items, seed=seed,
            partition_seed=self.partition_seed, k=k, reused=reused,
            split_rule=(f"series-disjoint: IID = 264 series starting in {IID_START_YEAR}, hash-ordered with the "
                        f"partition seed: {N_VAL} val, {N_ID} id, rest src; OOD = {d.ood_source}"),
            visible="load_train: histories (train parts) of the 264 IID series; per-item history; dev backtest",
            extra={"horizon": H, "period": m, "frequency": freq})
        return Episode(
            id=episode_id(CODE, split, seed, k), discipline=CODE, family=self.family, split=split,
            task_type=self.task_type, objective=objective,
            required_output=PortSchema("array", (n, H), UNIT, "float", f"{H}-step {freq} forecast per series"),
            tools=tools, constraints=constraints, budget=default_budget(max_node_s=300.0, max_llm_items=2 * n),
            lineage=lineage,
            acceptance=(f"accepted iff mean MASE <= {1 - ACCEPT_MARGIN:.2f} x the reference mean MASE on the same "
                        "items (reference recipe and value only in EvalResult.details / docs, never shown to the "
                        "policy)"),
            tolerance={"rtol": 1e-5, "atol": 1e-6},
            tags=[CODE, self.family, "time-series", "forecasting", "tourism", freq, "seasonal", "MASE", "Monash"],
            metric=self.metric, direction="min", n_items=n, _evaluate=evaluate, _dev_evaluate=None)

    # ----------------------------------------------------------------------------------------------
    def pooled_metric(self, per_episode: list[dict]) -> float | None:
        """Mean MASE over all items of the given episodes."""
        vals = [float(v) for p in unwrap_payloads(per_episode) for v in p.get("mase", [])]
        return float(np.mean(vals)) if vals else None


def h_text(freq: str) -> str:
    return "for the month" if freq == "monthly" else "for the quarter"
