# FoR33 Built environment and design — BuildingsBench day-ahead building load (balanced CVRMSE %, lower is better)

Adapter: `scienceclaw/bench/tasks/for33_buildingsbench.py` (`Adapter`), shared helpers in `_forecast_common.py`, delivery
resolution in `_delivery_roles.py`.
Status: **available** on the data team's `reconstructed_v2` delivery (tests: `tests/test_task_for33.py`).

## Dataset
| field | value |
|---|---|
| name / version | BuildingsBench **v1.0.0** (official OEDI S3 release), real-building constituents LCL, IDEAL, Borealis, BDG-2 (residential + commercial) and SMART, Electricity (OOD). Data team delivery: `reconstructed_v2` (seed 20260928). |
| URL | https://github.com/NatLabRockies/BuildingsBench ; data prefix `https://oedi-data-lake.s3.amazonaws.com/buildings-bench/v1.0.0/BuildingsBench/` |
| license | https://data.openei.org/submissions/5859 (BuildingsBench CC-BY-4.0); LCL: CC BY 4.0; SMART: UMass Trace Repository terms |
| delivery | catalog `configs/data-delivery-v1/catalog.json` entry `FoR33` -> `data_root = .../datasets/for33-buildingsbench/reconstructed_v2` (`resolve_delivery`; a newer `reconstructed_v*` directory that the catalog does not bind is only listed in `lineage.newer_reconstructed_dirs_on_disk`) |
| role files | `configs/episode-designs/episodes/FoR33/{source,val,id,ood}.jsonl` (sha256 `4a1573ce…`, `b411d6bc…`, `fb86804f…`, `b124f41c…`, checked against the catalog on every load): one row per physical building (`unit_id` = building, `category`, `dataset_group`, 8 `windows` with `window_id`, `context_start`, `target_start`, and the `adaptation` history file with its sha256) |
| arrays | `reconstructed_v2/{source,validation,evaluation}.npz` (168-h `context`, 24-h `target`, ids, timestamps; cross-checked with the role rows: ids, building, category, group, timestamps, finite / non-negative / <= 1300 kWh; the npz files themselves are not hash-pinned by the catalog) and `reconstructed_v2/adaptation/<role>/<building>.npz` (hourly `load`, `timestamp`, `internal_train_mask`; sha256-checked against the role row) |

## Items, pools, splits
* **Item** = one building-specific window: input 168 hourly loads (kWh), target the next 24 hourly loads. Item id = the
  row's `window_id` (e.g. `residential/MAC002290/2012-11-06T16:00:00`); **lineage group = physical building**.
* Roles (27 buildings, both BuildingsBench categories in every role, 8 windows per building):

| role | buildings | residential / commercial | dataset groups | windows |
|---|---|---|---|---|
| src | 8 | 4 / 4 | LCL 3, IDEAL 1; BDG-2 bear 1, rat 2, fox 1 | 64 |
| val | 3 | 2 / 1 | Borealis 1, IDEAL 1; BDG-2 bear 1 | 24 |
| id | 11 | 7 / 4 | LCL 7; BDG-2 rat 3, bear 1 | 88 |
| ood | 5 | 1 / 4 | SMART HomeB 1; Electricity (MT_*) 4 | 40 |

* OOD is a held-out BuildingsBench *sub-dataset group* (different constituent datasets and, for the commercial part,
  building type), not a separately acquired benchmark: `lineage.ood_kind = "proxy_within_dataset"`. `MT_070` (reserved
  validation target-group building) is in `validation.npz` but in no role and is never used (`reserved_buildings_excluded`).
* Window ids **and** buildings are disjoint across roles (`verify_disjoint`, asserted at load and in the tests).
* **Episodes** hold every building of their role with `m = max(1, min(M_TARGET[split], items_per_episode // n_buildings))`
  windows each (M_TARGET src 1 / val 2 / id 2 / ood 2): at 16 items **src 8, val 6, id 11, ood 10 items** per episode. The
  windows of one building are drawn conflict-free (no target inside another item's context; consecutive official windows
  can be only 7 days apart) **and at least `MIN_DELAY_H = 144` h apart** (LEAK-4: the target of one window ends >= 144 h
  before the other window's context starts, both directions, i.e. targets >= 14 days apart), and successive episodes take
  disjoint windows (seeded exact packing). Capacity at 16 items: src 8, val 4, id 8, ood 4 episodes (the same as without
  the delay rule; 168 h would already cut val to 3, 240 h to 2, 336 h to 1); the default plan (7 / 2 / 4 / 4) needs no
  reuse. `src` recycles windows beyond its capacity (flagged in the lineage); the held-out splits raise `PoolExhausted`.
  `lineage.min_window_delay_h` / `min_window_delay_observed_h` record the rule and the smallest delay in the episode
  (None when the episode has one window per building: src and id at 16 items).
* **Residual same-building leak (LEAK-4, documented).** Every context of an episode is visible, so a later window of a
  building tells something about an earlier window's target. Measured on the permitted history of all 27 buildings
  (one-off script, not in the repo; synthetic pairs, target of A vs a context starting `D` h after A's target ended; pooled
  least squares fitted on the scored pairs, i.e. an upper bound; extra RMSE gain from the later context over the best
  own-context predictor "previous day + same weekday last week"): D = 0 h 7.0 %, 24 h 5.8 %, 72 h 5.1 %, 120 h 4.5 %,
  **144 h 3.7 %**, then a plateau of 3.5-4.9 % up to D = 1500 h (median over buildings of the per-building CVRMSE ratio
  0.89 at D = 0, 0.92-0.95 beyond). A naive "average previous day and the nearest same-hour value of the later context"
  beats persistence by 12 % at D = 0 and 6-9 % beyond. The delay rule therefore removes the near-neighbour effect
  (about half of the D = 0 gain, and the "tomorrow" values) but not the plateau, which is the building's weekly profile /
  level read from another week: spacing cannot remove it, and it is of the size of half the 10 % acceptance margin.
  On the 8 real windows per building (673 conflict-free ordered pairs, all roles) the incremental oracle gain per delay
  bucket is 2-5 % without a monotone trend (too few pairs per bucket for a curve). Episodes with one window per building
  (src, id at 16 items) have no such pairs.

## Visible data and tools (D_E)
The data team permits a building's own history for adaptation ("evaluation-building history is permitted for that
building's adaptation; its future targets are evaluator-only"). No other building's data and no target is exposed.
| tool | returns |
|---|---|
| `load_history` | `load` (n_buildings x 3601 h, kWh), `building_id`, `category`, `history_start` — the first 3601 hours (`internal_train_mask` block) of each episode building; it ends before every evaluation context (asserted) |
| `load_dev` | 4 dev windows per building (168-h contexts + `building_id`, `category`, `context_start`, `target_start`); the targets lie in the 720 hours after the history block, 8 days apart (no dev target inside another dev context) |
| `score_dev(pred)` | balanced CVRMSE of `pred` (n_dev x 24) on the dev windows + the reference's dev score (visible dev signal; `_dev_evaluate = None`) |
| `load_eval_inputs` | `context` (n x 168 kWh), `building_id`, `category`, `context_start`, `target_start` (ISO) |

## Metric, reference, acceptance (D_V)
* **Balanced CVRMSE (%)** — the data team's normative convention: per building `NRMSE_b = 100 * sqrt(mean_{all target
  hours of b}(y - ŷ)^2) / mean_{all target hours of b}(y)` (BuildingsBench `Metric('cvrmse')`, kWh); the **median over the
  buildings of each category**; primary = `0.5 * (residential median + commercial median)`. In `metrics`: the category
  medians, pooled NRMSE, RMSE / MAE (kWh) and the reference score.
* **Pooled metric** (`pooled_metric`): the same formula over all items of several episodes (per-building SSE, Σy and counts
  are accumulated from `pooled_payload.items`).
* **Reference**: previous-day persistence `ŷ[i,h] = context[i,144+h]` (BuildingsBench `CopyLastDayPersistence`).
* **Acceptance**: `balanced CVRMSE <= 0.90 x reference` (`ACCEPT_MARGIN = 0.10`, v1: 0.15). The persistence forecast is
  never accepted, the oracle is (both asserted in the tests).
* `details`: `reference`, `norm_score` = reference/primary clipped to [0,10], `pooled_payload`, `reference_payload`,
  per-building values.
* **Hard constraints**: `output_shape` (n, 24); `finite`; `physical_range` 0 <= y <= 1300 kWh and
  `y[i,h] <= max(20 * max(context[i]), 1)` (scale/unit guard, e.g. Wh submitted as kWh); `declared_unit` = kWh.
* **Calibration** (adapter check at the default plan of bench seed 20260928; not a research result): persistence reference
  mean per split src 46.6 / val 71.6 / id 53.0 / ood 53.0 % (per episode src 38-60, val 65-78, id 38-69, ood 39-65); a generic
  7-day mean hourly profile scores src 34.6-48.9, val 53-58, id 35-62, ood 33-60 % and is accepted on 3/7, 2/2, 2/4, 2/4 episodes.
* **Effect-size / noise analysis of `scilib.loadforecast`** (improver pass, 2026-09-29; developer-side, hidden targets used
  only for this analysis). Frozen-run `id` e00 (57.60 vs 53.35, ratio 1.08) is reproduced exactly offline with the library
  defaults; the agent had used `backtest_history` -> `pick_lowest` -> `ens` (no override, no `score_dev`), so it is neither an
  agent misuse nor an adapter/evaluator bug: residential median 79.0 vs 72.0 with the commercial median equal, while the
  in-history backtest had put `ens` at 0.80 x persistence (the evaluation windows lie months after the history and residential
  day levels move by +-30-60 %). Planned default episodes (src 8 / val 4 / id 8 / ood 4, bench seeds of the frozen run, real
  evaluator): `ens` accepted 20/24 (8/8, 4/4, 5/8, 3/4; mean ratio 0.791), `ml` 18/24, `core` 15/24, `median7` 15/24,
  persistence 0/24. Resampled episodes (2000 draws per role from the 8 windows of every building, common random numbers):
  `ens` mean ratio 0.786, per-episode acceptance 0.85 / 0.94 / 0.71 / 0.66 (src / val / id / ood), i.e. about 0.21 misses
  overall; a perfectly known daily load level would give 0.71 (needs weather), the iid floor without it is about 0.75, the
  best pretrained + fine-tuned model of the BuildingsBench paper is 0.78. Variants tried without a gain beyond +-0.005 in the
  mean ratio / +-0.02 in acceptance: ahead-hour, day-level, neighbour-hour and day-type features; in-context backtest-error and
  profile-agreement features; absolute level / volatility features; models without the history profile; recency-weighted and
  per-building-reweighted fits; LightGBM 80x7 and 300x31; `w_core` 0.3-0.6; three-seed bagging (identical); in-sample NNLS
  stacking by category (0.794); a level calibration factor. The remaining misses are therefore noise of a median over 4-6
  single-day residential CVRMSEs (episode ratio sd about 0.1), not a library defect; a higher pass rate needs weather or larger
  episodes, which are acceptance/episode-design decisions and were not changed. End-to-end with the 27B agent (default
  library, response cache replays the frozen run for the first episodes): `id` e00 57.60 (not accepted, replayed), `id` e01
  38.40 (accepted); `src` e00-e03 40.68 / 31.81 / 37.12 / 42.07 (all accepted, norm 1.14-1.33; e02/e03 fresh calls).

## Pretrained time-series models (`scilib/tsfm.py`)
`scilib.describe_extra("tsfm")` is appended to the objective only when `tsfm.available()` (local torch + chronos + weights, or the remote GPU bridge). `tsfm.forecast(histories, horizon, quantiles, model, context_length, lengths)` returns float32 quantile forecasts (n, horizon, Q) from `chronos_2` (`amazon/chronos-2`) or `chronos_bolt` (`amazon/chronos-bolt-base`); the series are forecast independently, neither model is adapted on them. The docstring is factual (inputs, outputs, quantile levels, training corpora, measured cost; `tests/test_scilib_tsfm.py`, `tests/test_adapter_visible_text.py`). Weights on the GPU host: `models/tsfm/{chronos_2,chronos_bolt_base}`; worker whitelist entry `("tsfm", "forecast")`. Pretraining overlap with Monash / Chronos Datasets / GIFT-Eval-type corpora cannot be ruled out and is stated in the docstring.
* Offline result with the pre-declared pipeline (0.5 x `ens` + 0.5 x Chronos-2 median over the full 168 h context, clipped at 0; id/ood evaluated once): balanced CVRMSE id 51.62 vs 54.82 % (-5.8%, building-cluster bootstrap CI -13.3% / +0.1%, 11 buildings), ood 46.06 vs 45.21 % (+1.9%, CI -16.3% / +2.4%, 5 buildings). Not significant and low power. Details: `reports/below_peers_rootcause/tsfm.md`.
* **Option in the main library (added 2026-10-01).** Same reason as FoR35 (0 of 12 FoR33 episodes imported `scilib.tsfm`): `LoadForecaster.candidates(..., pretrained=True)`, `forecast_candidates(..., pretrained=True)` and `backtest_history(..., pretrained=True)` add the keys `PRETRAINED_CANDIDATES = ('chronos2', 'ens_chronos2')` (`chronos2` = Chronos-2 median over the whole 168-h context clipped at 0; `ens_chronos2` = 0.5 x `ens` + 0.5 x `chronos2`, the pre-declared pipeline). Default `pretrained=False` returns exactly the previous dict and makes no GPU call. Tests: `tests/test_pretrained_options.py`.

## Budget
`max_steps 12, max_policy_tokens 200k, max_wall_s 1800, max_node_s 300, max_llm_items 2 x items`. Data load and plan
building < 0.5 s; one evaluation takes milliseconds.

## Changes from v1 (`reconstructed_v1`)
* v1 was LCL-only residential (702 LCL-2013 buildings, seeded first-20 rule, 192-h grid windows, SMART OOD read from CSVs) with a
  uniform mean of per-building NRMSE and "unselected LCL buildings" as training data; v2 adds commercial buildings (BDG-2,
  Electricity), reports the category medians / balanced score and gives per-building adaptation history instead.
* Pools, building membership and all 8 windows per building now come from the frozen role files (src 8, val 3, id 11, ood 5 buildings)
  rather than from the adapter's own reconstruction; window arrays come from the three npz files.
* OOD changed from "cross_dataset" (SMART) to "proxy_within_dataset" (SMART HomeB + Electricity commercial group).
* Episodes are per role (all role buildings, few windows each) instead of round-robin draws from a large window grid.
* Metric semantics: the per-building CVRMSE formula is unchanged (all target hours of the building), the aggregation is now
  the median within each category and the mean of the two category medians (v1: unweighted mean over buildings);
  persistence reference unchanged; acceptance margin 15 % -> 10 % (calibrated on the v2 baselines above).

## Deviations from the historical protocol and open issues
* Exact historical sample ids / split rules were **not recovered**; these are the data team's rebuilt splits
  (`rebuilt_split = true`, `historical_sample_ids_recovered = false`).
* Small pools: 8 windows per building, so an id episode is 11 buildings x 1-2 windows and the number of disjoint
  held-out episodes is limited (id 8, ood 4, val 4 at 16 items).
* The delivery README still quotes stale building counts; the role files (and this document) are authoritative.
* The full BuildingsBench real-building suite (Sceaux, Buildings-900K, ...) is not used; only the constituents of the
  delivery are.
