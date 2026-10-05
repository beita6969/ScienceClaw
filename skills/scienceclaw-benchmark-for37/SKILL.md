---
name: scienceclaw-benchmark-for37
description: "Use when a task asks for a 24-hour-ahead forecast of a global gridded surface field (2 m temperature on a 64x32 equiangular ERA5 / WeatherBench 2 grid, 6-hourly) from a training record and the four preceding fields of each initialisation, delivered in kelvin and judged by latitude-weighted RMSE (FoR37 of the companion ScienceClaw-Eval benchmark)."
metadata: { "openclaw": { "emoji": "📊" } }
---

# Global gridded 2 m temperature forecast, 24 h lead

## Task
- Input: a training record `fields (T, lon, lat)` in K with ISO UTC `times` (6-hourly, consecutive steps); per initialisation `context (n, 4, lon, lat)` in K at init-18 h, -12 h, -6 h, 0 h and `init_time` (n ISO strings).
- Deliverable: float forecast `(n, lon, lat)` in K for init + 24 h, axes in the store's (longitude, latitude) order, finite and within a physical range (about 150-350 K, which catches degrees Celsius).
- Quality: WeatherBench 2 RMSE, lower is better: per initialisation sqrt of the area-weighted mean squared error (weights proportional to sin(lat + d/2) - sin(lat - d/2), mean 1), then the mean over initialisations (`scilib.weather.lat_weighted_rmse`). Natural references: persistence (the field at init) and seasonal cycle plus damped anomaly persistence.

## Routes
- `scilib.weather.fit_predict(fields, times, context, init_time)` (operator `gridded_field_patch_ridge_forecast`): per grid cell and UTC-hour bin a seasonal cycle (constant plus 3 annual harmonics), then one ridge regression per latitude row of the anomaly at +24 h on the (2r+1)x(2r+1) anomaly patches of the four context fields (defaults radius 2, lam 30; longitude wraps). Building blocks: `fit_seasonal_cycle`, `seasonal_cycle_at`, `anomalies` (operator `gridded_field_seasonal_anomalies`), `fit_patch_ridge`, `predict_patch_ridge`, `lat_weights`.
- Validation: `weather.blocked_cv(fields, times, gap_days=6)` runs temporal block cross-validation inside the record (training steps farther than `gap_days` from the held-out block) and returns per-initialisation RMSE. Operator `gridded_field_lat_weighted_rmse` scores any forecast.
- The tool library has no pretrained global weather model asset (`scienceclaw_tools(operation=weights)` lists what is staged).

## Pitfalls
- Respect temporal causality: fit only on record steps that precede the forecasts you issue, and compute row i of the forecast from `context[i]` and `init_time[i]` alone. Another initialisation's context can contain the future of item i, so never pool contexts across items to forecast a target.
- A seasonal cycle fitted on a short record (for instance one year) is extrapolated to other seasons or years; validate on a later or different period, not on random time steps (neighbouring steps are strongly autocorrelated, hence the block gap).
- Use one forecast function for validation initialisations and final initialisations, and score validation forecasts only against validation targets built from validation contexts.
- Keep units in K and the (lon, lat) axis order; longitude is periodic, latitude is not. Do not convert to degrees Celsius or transpose the output.
