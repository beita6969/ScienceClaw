# FoR38 fixed macro tool engineering batch (2026-10-03)

## Route

`macro_fixed_predict` is a policy-visible wrapper around `scilib.macro.fit_predict`.
It fixes members to `ridge,huber,lgbm_core,robdrift`, equal weights, `n_backtest=0`,
`seed=0`, and `n_jobs=1`. Its inputs are the visible `load_train` panel and the
label-free `load_eval_inputs` fields; it returns forecast `y` and provenance. The
wrapper rejects post-origin columns, non-finite forecasts, and invalid percentage
ranges. It does not read target values or the scorer.

## Leonardo run

- Existing one-card Qwen service: job `59277346`, node `lrdn2979`, endpoint `37346`.
- Runner: `H38`, `run_existing_singlecard_probe.sh`, no GPU broker.
- ID: 4/4 accepted and reproducible; sMAPE values `17.5165, 11.6449, 11.2290,
  14.2470`, mean **13.6593**.
- OOD: 4/4 accepted and reproducible; sMAPE values `9.5319, 11.3382, 10.6144,
  10.3970`, mean **10.4704**.
- All eight final graphs contain the direct `macro_fixed_predict -> submit.y` edge.

This is engineering evidence on a new batch. It does not change the formal manifest
or claim an external same-metric SOTA comparison; the public WDI panel has no
verified leaderboard with the same anonymised split and sMAPE protocol.
