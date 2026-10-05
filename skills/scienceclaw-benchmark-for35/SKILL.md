---
name: scienceclaw-benchmark-for35
description: "Use when forecasting monthly tourism, visitor or other seasonal business and service series (Monash Tourism Monthly, 24-month horizon, period 12) from each series' history, delivering non-negative forecasts per series scored by mean MASE. Corresponds to FoR35 of the companion ScienceClaw-Eval benchmark."
metadata: { "openclaw": { "emoji": "📊" } }
---

# Monthly seasonal series forecasting (Monash Tourism Monthly)

**Task.** Input: histories of monthly series (about 90-330 observations, own units), horizon 24, seasonal period 12, possibly a pool of other series of the same collection. Deliverable: float array (n_series, 24), the 24 months after each history, finite, >= 0, in the series' units.

**Quality.** Mean MASE over series, lower is better: mean absolute error over the horizon divided by the in-sample seasonal-naive error (lag 12). Reference: seasonal naive (repeat the last 12 months); its published mean MASE on the full Tourism Monthly set is about 1.63.

**Library** (check `scienceclaw_tools(operation=show, target=forecast)`):
- `scilib.forecast.fit_predict(histories, horizon, period, train=..., phase=...)`: median of damped ETS, Theta, STL+ETS, airline SARIMA and global ridge / extra-trees window models, clipped at 0. Also `forecast_panel`, `combine`, `global_window`, `global_lgbm`, rolling-origin `backtest_panel` + `combination_mase`, metrics `mase`, `panel_mase`, `smape`, `rmsse`.
- Pretrained, zero-shot, GPU or remote worker (guard with `scilib.tsfm.available()`; assets `chronos_2`, `chronos_bolt_base`): `forecast.pretrained_forecast(histories, 24)` (Chronos-2 median, last 120 values, log1p for non-negative series); `fit_predict(..., pretrained=w)` blends it with weight `w`; `tsfm.forecast` gives quantiles.
- Operators: `seasonal_panel_forecast_ensemble`, `statistical_forecast_single_method`, `forecast_backtest_method_ranking`, `chronos_median_forecast_nonnegative_panel`.

**Routes.** Baselines: seasonal naive, per-season mean, damped ETS, Theta. Default: the `fit_predict` ensemble, optionally blended with a pretrained forecast. Choose by `backtest_panel` (hold out the last 24 months of each history).

**Rules.**
- Respect temporal causality: nothing after a series' forecast origin may inform its forecast. When backtesting, cut every pool series at the same origin (`train_cut="auto"` does this when the pool contains the histories).
- Build the deliverable from the series to be forecast, not from truncated backtest copies of them.
- Pass `phase` / `train_phase` (month - 1 of the first observation) when series start in different months.
- Mean MASE over few series is noisy; a 2-3 % backtest gain is within noise.
- Chronos pretraining corpora include Monash-archive-type data, so tourism series may have been seen. Disclose this overlap.
