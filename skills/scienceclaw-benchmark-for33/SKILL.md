---
name: scienceclaw-benchmark-for33
description: "Use when forecasting day-ahead hourly electricity load of individual buildings (residential and commercial smart-meter data, BuildingsBench-style) from the previous 168 hours and optionally the building's own earlier history, delivering a 24-hour forecast in kWh per window scored by balanced CVRMSE. Corresponds to FoR33 of the companion ScienceClaw-Eval benchmark."
metadata: { "openclaw": { "emoji": "📊" } }
---

# Day-ahead building load forecasting (BuildingsBench)

**Task.** Input per window: 168 hourly loads in kWh (oldest first), building id, category (`residential` / `commercial`), ISO `target_start`; optionally each building's earlier hourly history (n_buildings, T) and latitude/longitude. Deliverable: float array (n, 24), the next 24 hourly loads in kWh, finite, >= 0, in window order.

**Quality.** Balanced CVRMSE in percent, lower is better: per building `100 * RMSE / mean(y)` over all its target hours, median within each category, mean of the two medians. Locally: `scilib.loadforecast.balanced_cvrmse(y_true, y_pred, building_id, category)`. Reference: yesterday's load repeated.

**Library** (check `scienceclaw_tools(operation=show|weights)`):
- `scilib.loadforecast`: context-only `yesterday`, `day_median`, `core_forecast(context, category)`; learned `fit(load, history_start, building_id, category)` then `LoadForecaster.candidates(...)` (keys yesterday, mean7, median7, core, ml, ens); `forecast_candidates`, `backtest_history` (hold-out inside the histories), `pick_lowest`, `history_windows`.
- Zero-shot pretrained, guard with `available()`: `scilib.tsfm.forecast` (Chronos-2, assets `chronos_2` / `chronos_bolt_base`; `candidates(..., pretrained=True)` adds chronos2 and ens_chronos2); `scilib.buildingsbench.forecast` (Transformer-Gaussian-L trained on simulated Buildings-900K, asset `buildingsbench_gaussian_l`; needs Box-Cox-normalised loads, calendar, latitude/longitude and building type as wired in the operator below).
- Operators: `building_load_day_ahead_context_forecast`, `building_load_day_ahead_learned_ensemble`, `building_load_day_ahead_pretrained_transformer`, `building_load_balanced_cvrmse`.

**Routes.** Baselines: persistence, per-hour median of the last 7 days (strong for residential), `core_forecast`. Learned: the `ens` ensemble fitted on the buildings' own history. Choose with `backtest_history`; keep the simple candidate unless another wins clearly.

**Rules.**
- Respect causality: use only data observed before each target window and only that building's history. With several windows per building, never use a later window's context to forecast an earlier one.
- Medians over a few buildings are noisy and loads drift months after the history ends; a small backtest edge is weak evidence.
- Keep kWh units and (n, 24) shape; clip at 0; a forecast far above the recent maximum signals a unit error.
- Chronos corpora include public electricity-load data, and BuildingsBench models come from the same project as the data; disclose possible overlap.
