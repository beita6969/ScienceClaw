# FoR35 Commerce, management, tourism and services — Monash Tourism Monthly (mean MASE, lower is better)

Adapter: `scienceclaw/bench/tasks/for35_tourism.py` (`Adapter`), shared helpers in `_forecast_common.py`.
Status: **available** (tests: `tests/test_task_for35.py`).

## Dataset
| field | value |
|---|---|
| name / version | Monash Time Series Forecasting Archive, *Tourism Monthly* (`tourism_monthly_dataset.tsf`), Zenodo record 4656096 **version 3** |
| URL | https://zenodo.org/records/4656096 (zip sha256 `6e434ad8a0ef6acb55ebee0e1d1ffe22707ea4e62c05e091326267f3a5b4a93c`, md5 verified by the data team) |
| license | CC BY 4.0 (Monash archive on Zenodo); original data: Athanasopoulos et al. (2011) tourism forecasting competition |
| local path | `<DATA_ROOT>/for35-monash-tourism-monthly/data/tourism_monthly_dataset.tsf` |
| size | 366 monthly series (91–333 observations), `@horizon 24`, no missing values |

## Items, pools, splits
* **Item** = one series: input = all observations except the last 24 (the Monash train part), target = the last 24 (the Monash test part). Id `tourism_monthly:T<k>`; lineage group = the series.
* **IID pool** = the 264 series that start in **1980** (the dominant start cohort). Hash-ordered with `sha256("FoR35|iid|20260928|id")`: 32 → val, 64 → id, 168 → src.
* **OOD pool** (automatic):
  * if a Monash **tourism quarterly** TSF is present (`<DATA_ROOT>/for35-*/**/tourism_quarterly*.tsf` or `<DATA_ROOT>/*tourism*quarterly*/**/*.tsf`), its series form the OOD pool (horizon 8, period 4) → `ood_kind = "cross_dataset"` (code path covered by a test with a synthetic fixture);
  * otherwise — **the case on 2026-09-28** — the 102 monthly series of the other start cohorts (1979, 1981, 1985, 1986, 1991, 2000; different providers, shorter histories) → `ood_kind = "proxy_within_dataset"`.
* Series-disjoint; partition fixed by `partition_seed`, independent of episode seeds. Episodes: seeded prefix-stable random draws. Capacity: src 10, val 2, id 4, ood 6 episodes.

## Visible data and tools (D_E)
| tool | returns |
|---|---|
| `load_eval_inputs` | `history` (list of 16 arrays) = the inputs of the deliverable `y`, `series_id`, `start`, `horizon` = 24, `period` = 12 |
| `load_train` | histories (train parts only) of all 264 IID series, series ids, start dates — never a forecast-period value (the 16 items of an episode are among these series, with all their observations before the forecast origin) |
| `load_dev` | dev backtest inputs (not the inputs of the deliverable): each item's history without its last 24 observations |
| `score_dev(pred)` | mean MASE of `pred` (16 × 24, a forecast for the `load_dev` histories) on the backtest targets (= the last 24 history values) + the reference's dev score |

The tools are listed in this order (deliverable inputs first) and `load_dev` / `score_dev` state that they concern the dev histories: in the frozen full run (A_0, 2026-09-29) the id episode `FoR35-id-s5a69a3a8-e01` forecast the `load_dev` histories for its deliverable (primary 1.864 vs reference 1.618) because both loaders name their output port `history`.

## Metric, reference, acceptance (D_V)
* **MASE** per item (Hyndman & Koehler 2006; Monash convention): `mean_h |y_h − ŷ_h| / mean_{t>m}|x_t − x_{t−m}|`, m = 12, scale from the item's history; primary = mean over items. Median MASE and mean sMAPE are also reported. The implementation reproduces the Monash published SNaive mean MASE on tourism monthly (1.631; asserted in the tests).
* **Pooled metric**: mean MASE over all items of the given episodes.
* **Reference**: seasonal naive (repeat the last observed 12 months).
* **Acceptance**: mean MASE ≤ 0.95 × reference.
* **Hard constraints**: `output_shape` (16, 24), `finite`, `physical_range` (y ≥ 0; y[i,h] ≤ 10·max(history[i]) scale guard), `declared_unit` (`native`, the series' own unit).
* **Calibration** (not a research result): reference mean src/val/id/ood = 1.734/1.743/1.670/1.400; a damped Holt-Winters ETS (statsmodels) scores 1.555/1.633/1.531/1.238 and is accepted on 5/7, 1/2, 3/4, 3/4 episodes.

## Domain library (`scilib.forecast`)
The objective ends with `scilib.describe("forecast")` (repo root on the worker `PYTHONPATH`; numpy / scipy / statsmodels / scikit-learn). It offers MASE / sMAPE / RMSSE, local methods (`repeat_last`, `repeat_season`, ETS variants, Theta, STL+ETS, airline SARIMA), `combine`, a rolling-origin `backtest_panel` (local methods plus the global ridge / ExtraTrees window models fitted on a pool cut at the same origin, so one call backtests a whole pipeline) with `combination_mase`, `global_window`, and the one-call `fit_predict(histories, horizon, period, train=..., train_cut=...)` = element-wise median of four local methods (damped ETS, Theta, STL-ETS, airline SARIMA) and the two global models, clipped at 0. Because `load_train` contains the full histories of the eval and dev items, global models fitted on it would see the dev targets: `fit_predict` / `backtest_panel` therefore truncate every train series that extends a given history to that history's length (`train_cut="auto"`, default; an int drops that many trailing observations of every train series, 0 uses the train series as given), which makes the dev score honest and leaves the eval call unchanged (there the train series equal the histories). Evidence: in the first end-to-end probe after the pass (id `FoR35-id-s5a69a3a8-e01`, deliverable correctly wired) the agent fitted `extra_trees` on the uncut train series, saw dev MASE 1.145 (vs 1.496 for the default ensemble), submitted it alone and scored 1.539 on the eval inputs (norm 1.0516, 0.001 below acceptance). End-to-end probes with the local 27B policy (`scripts/probe.py`, workers 1): src 3/3 passed (primary 1.328 / 1.691 / 1.533, 16 / 5 / 10 steps); id (same seed as the failing frozen episode) with the leaky default 1/2 (1.401 pass, 1.539 miss), after the automatic truncation 2/2 (1.310 in 10 steps, 1.444 in 12 steps vs 1.864 in the frozen run; dev scores 1.27 / 1.54 vs references 1.85 / 1.70). ood (same seed as the frozen `wall_budget` episode, machine load ~6): 2/2 passed (1.108 and 1.058 in 17 steps, 207-209 s wall, of which ~138 s policy time). Functions raise messages that name the offending shape / empty series (the frozen full run lost steps to `combine` shape mismatches and to a hand-rolled global-model backtest). CPU cost: `fit_predict` for 16 series with the 264-series pool ~4-8 s; `backtest_panel` on the 264-series pool with 32 targets and 2 origins ~15-30 s.
Improver pass 2026-09-29 (frozen full run, load 40-75): id `FoR35-id-s5a69a3a8-e01` ended worse than the reference because the deliverable was built from the `load_dev` histories (adapter/ergonomics cause, not the library); ood `FoR35-ood-s1b207873-e01` hit `wall_budget` mostly through policy latency (~1550 s of 2027 s outside code nodes) after several hand-rolled global-backtest alignment errors. Library-level numbers (`scripts/dev/run_node_on_episodes.py`, 16 items/episode, `fit_predict(..., train=load_train series)` for both dev and eval): src 4/4 accepted (norm 1.17 / 1.11 / 1.20 / 1.23), id 3/4 (1.25 / 1.04 / 1.16 / 1.20), ood 3/4 (1.16 / 1.16 / 1.15 / 1.05); the four-method local median alone (previous default) gave 3/4 on each split (norm 1.18 / 1.04 / 1.22 / 1.21, 1.21 / 1.00 / 1.14 / 1.15, 1.26 / 1.19 / 1.14 / 1.02). Whole-pool ratio to the reference (history minus its last 24, real last-24 window): src 0.821 vs 0.839, val 0.856 vs 0.878, id 0.859 vs 0.887, ood (train pool = IID cohort) 0.861 vs 0.853. Acceptance needs a ratio <= 0.95, and the episode-level standard error of the ratio is about 0.05-0.06 (16 items), so a 0.02-0.03 library gain moves single-episode outcomes only within noise.

## Pretrained time-series models (`scilib/tsfm.py`)
`scilib.describe_extra("tsfm")` is appended to the objective only when `tsfm.available()` (local torch + chronos + weights, or the remote GPU bridge). `tsfm.forecast(histories, horizon, quantiles, model, context_length, lengths)` returns float32 quantile forecasts (n, horizon, Q) from `chronos_2` (`amazon/chronos-2`) or `chronos_bolt` (`amazon/chronos-bolt-base`); the series are forecast independently, neither model is adapted on them. The docstring is factual (inputs, outputs, quantile levels, training corpora, measured cost; `tests/test_scilib_tsfm.py`, `tests/test_adapter_visible_text.py`). Weights on the GPU host: `models/tsfm/{chronos_2,chronos_bolt_base}`; worker whitelist entry `("tsfm", "forecast")`. Pretraining overlap with Monash / Chronos Datasets / GIFT-Eval-type corpora cannot be ruled out and is stated in the docstring.
* Offline result with the pre-declared pipeline (0.5 x `fit_predict` + 0.5 x Chronos-2 median, context 120 months, log1p / expm1; chosen on src/val among 12 settings, id/ood evaluated once): MASE id 1.3908 vs 1.4347 (-3.1%, 95% CI -8.0% / +1.4%, 64 series), ood 1.1873 vs 1.2296 (-3.4%, CI -7.2% / +0.5%, 102 series). Not significant. Tourism series are likely part of the pretraining corpora. Details: `reports/below_peers_rootcause/tsfm.md`.
* **Option in the main library (added 2026-10-01).** In the 2026-10-01 tool-ON runs with a self-hosted 27B model (id + ood + light batch) the agent never imported `scilib.tsfm` (0 of 12 FoR35 episodes; its reasoning never mentioned it), while an option on a function it already calls (FoR48 `plm=`) was used in 8 of 8 episodes. The pre-declared pipeline is therefore also exposed as default-off arguments of the library the agent uses: `forecast.fit_predict(..., pretrained=w)` returns (1 - w) x its usual result + w x `forecast.pretrained_forecast(histories, horizon, model='chronos_2', context=120)` (Chronos-2 median from the last 120 observations, log1p / expm1 for non-negative series). `pretrained=0` (default) is the previous behaviour bit for bit and makes no GPU call. The text in the module docstring is factual (signature and definition only; no target, threshold or recipe). Unit tests: `tests/test_pretrained_options.py`.

## Budget
`max_steps 12, max_policy_tokens 200k, max_wall_s 1800, max_node_s 300, max_llm_items 32`.

## Deviations from the historical protocol
* Exact historical sample ids / split rules were **not recovered**; these are **rebuilt splits**. Historical protocol 64 IID + 64 OOD: here id = 64, ood = 64 items.
* No second tourism dataset is available locally, so OOD is currently a within-dataset cohort shift. **Data team:** fetching `tourism_quarterly_dataset.zip` (Zenodo 4656093, Monash archive) into `for35-monash-tourism-quarterly/data/` (or `for35-monash-tourism-monthly/data/`) turns OOD into the recommended cross-dataset shift with no code change (the OOD horizon then becomes 8 quarters).
