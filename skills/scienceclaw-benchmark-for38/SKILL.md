---
name: scienceclaw-benchmark-for38
description: "Use when a task asks to forecast annual country-level economic series several years ahead (for example GDP per capita levels or unemployment rates, horizon 4) for many economies, from a panel of past series and covariate indicators such as growth and inflation, delivered as level forecasts and judged by mean sMAPE (World Bank WDI-style macro panel forecasting; FoR38 of the companion ScienceClaw-Eval benchmark)."
metadata: { "openclaw": { "emoji": "📊" } }
---

# Annual macro-panel forecasting (sMAPE)

## Task
- Input: a panel table (`economy_id, region, income_level, indicator`, then one column per period `t-31 .. t0`, NaN = missing); per item `history (n, L)` oldest first, its `indicator` kind, `covariates (n, k, L)` (the same economy over all kinds) with `covariate_indicators`, and the kind descriptions (`indicator_kinds`).
- Deliverable: float `(n, H)` level forecasts for the H periods after `t0` (H = 4 in the companion benchmark), in each item's own unit, finite and non-negative; percentage kinds at most 100.
- Quality: mean sMAPE in percent, 200/H * sum_h |y - yhat| / (|y| + |yhat|), lower is better (`scilib.forecast.smape`). The no-change (random-walk) forecast is hard to beat on macro series; always report it as the reference. Series that cross zero (growth, inflation) serve as covariates only, since sMAPE is ill-defined there.

## Routes
- `scilib.macro.fit_predict(train_panel, history, indicator, indicator_kinds=..., covariates=..., covariate_indicators=..., economy_id=..., region=..., income_level=..., horizon=H)` (operators `annual_panel_forecast_pooled`, `annual_panel_forecast_with_covariates`) forecasts the log-ratio ln(y[t+h]/y[t]) with members from `macro.MEMBERS`. The default `DEFAULT_MEMBERS = (ridge, huber, lgbm_core, robdrift)` is equally weighted; `weights="auto"` fits weights on a rolling-origin backtest and mixes them 50:50 with equal weights; `return_info=True` adds backtest sMAPE per member, kind and origin.
- Blocks: `rolling_backtest`, `fit_weights`, `period_profile` (cross-economy median change per period, to spot common-shock periods), `local_forecast(x, "flat"|"drift"|"revert"|"theta"|"holt", horizon)` (operator `annual_series_local_forecast`).
- Pretrained comparison: `scilib.tsfm.forecast(histories, H, quantiles=(0.5,), model="chronos_2")` (assets `chronos_2`, `chronos_bolt_base`; operator `chronos_quantile_forecast`); clip to each kind's plausible range. Check `tsfm.available()` and `scienceclaw_tools(operation=weights)`.

## Pitfalls
- No value after the forecast origin may be used. `fit_predict` restricts the panel to its first `n_periods` columns (default: the history length), so a backtest on a shortened history never sees later columns; do not append later values to any history or panel.
- A single-origin validation split is a noisy guide: a few dozen values whose targets hold whatever shock hit the last observed periods. Compare members on a multi-origin `rolling_backtest` (`info["backtest_by_origin"]`) and check `period_profile` before replacing the equal-weight default.
- Percentage kinds (unemployment) are the hard part: features from other kinds can be out of distribution after a common shock, so prefer own-series members there.
- Memorisation risk: these are public statistics, and pretrained forecasters or an LLM may have seen the realised values. Never recall or look up realised values; report gains as potentially contaminated unless they hold on a held-out region or economy group.
