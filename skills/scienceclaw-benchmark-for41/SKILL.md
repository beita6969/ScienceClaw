---
name: scienceclaw-benchmark-for41
description: "Use when a task asks for 30-day-ahead probabilistic daily forecasts of dissolved oxygen (mg/L) and water temperature (degC) at freshwater monitoring sites from multi-year daily histories (NEON aquatics, neon4cast ecological forecasting), delivered as a normal predictive distribution (mu and sigma per variable and lead day) and judged by CRPS (FoR41 of the companion ScienceClaw-Eval benchmark)."
metadata: { "openclaw": { "emoji": "📊" } }
---

# Probabilistic aquatic forecasts (oxygen and temperature, CRPS)

## Task
- Input: `history (n, L, 3)` daily means (channel 0 oxygen mg/L, 1 temperature degC, 2 chlorophyll-a ug/L, not forecast), the last column being each item's `reference_date`; NaN = not observed or withheld; site id and type may be given.
- Deliverable: a dict of four float arrays `(n, 30)`: `oxygen_mu`, `oxygen_sigma`, `temperature_mu`, `temperature_sigma` for the days reference_date + 1 .. + 30. All finite, sigma > 0, oxygen mu in [0, 30] mg/L, temperature mu in [-5, 45] degC (this catches kelvin), sigma at most 50.
- Quality: closed-form CRPS of N(mu, sigma^2) at the observed (item, day) pairs (unobserved days are not scored), averaged per variable; primary = equal-weight mean of the oxygen and temperature CRPS, lower is better. The two have different units, so report both (`scilib.aquatics.score(pred, obs)`, `crps_normal`). Reference: day-of-year climatology (mean and sd within +-7 days of each target day of year, `aquatics.doy_window_forecast`).

## Routes
- `scilib.aquatics.fit_predict(history, reference_date, lam=0.75, smult=1.0)` (operator `aquatic_oxygen_temperature_forecast_30d`): smooth day-of-year profile, then per lead day a ridge regression of the anomaly on the recent anomaly (mean of the last 3 and last 30 days), the site fit blended with a pooled fit by `lam`, sigma from the residual sd times `smult`. `lam` and `smult` moved the mean CRPS by only a few percent in backtests; do not over-tune them. Items with fewer than 60 observed days of a variable fall back to the climatology.
- Checks: `aquatics.backtest(history, reference_date, n_origins=3, offset=30)` is a rolling-origin test that needs only `history`; `data_report` shows coverage and age of the last observation; `seasonal_profile`, `history_dates` are building blocks. Operators `aquatic_climatology_forecast_30d`, `normal_forecast_crps_score`.
- Pretrained blend: `aquatics.fit_predict(..., pretrained=w)` mixes every mu and sigma with Chronos-2 (median as mu, (q75 - q25)/1.349 as sigma; `scilib.tsfm`, asset `chronos_2`), w in [0, 1]; w = 0 makes no GPU call. Check `tsfm.available()` and `scienceclaw_tools(operation=weights)`. Pretraining overlap with public time-series corpora cannot be ruled out; state it.

## Pitfalls
- CRPS punishes both overconfident and needlessly wide sigma; check calibration in a backtest rather than only the mean error. The remaining error at long leads is mostly weather-driven temperature shocks.
- Respect temporal causality: forecast each item from history that ends at its own reference date. If one site appears with several reference dates, a later history can contain an earlier item's forecast window; never use it for the earlier date, and keep withheld (NaN) windows as NaN, not zero.
- A backtest window at the end of a history can be partly withheld, so its score is a noisy guide.
- Keep units: temperature in degC (not K), oxygen in mg/L; clip mu and sigma to the ranges above.
