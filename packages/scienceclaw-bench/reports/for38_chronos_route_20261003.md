# FoR38 Chronos-2 pretrained route (2026-10-03)

## Implementation

`chronos_fixed_predict` is an additional FoR38 tool. It accepts only the
label-free `history`, opaque `indicator`/`indicator_kinds`, and
`target_offsets` payloads from the existing tools. It calls the staged
`scilib.tsfm.forecast` Chronos-2 weights with a fixed median quantile and
32-period context, applies the existing nonnegative/unit caps, and returns
submit-ready forecasts with provenance. The route does not read targets or the
scorer and does not change the official scorer, reference, or acceptance.

Commit: `97b34c6`.

## Checks and server evidence

- FoR38 tests: **15 passed**.
- Leonardo `sc-run` compilation passed; remote source SHA-256 is
  `40a25156b5f30edc5df14de38a0bc0c0f50c4e11792874cc6bd0db5a7b1d9239`.
- On allocation `59277346` / node `lrdn2979`, direct Chronos-2 full-pool
  inference used the staged model and produced finite forecasts:
  - ID (64 items): mean sMAPE **16.9616**; naive reference **17.6282**;
    history-backtest reference **16.8299**.
  - OOD (141 items): mean sMAPE **11.7577**; naive reference **12.0508**;
    history-backtest reference **12.0392**.

The frozen macro ensemble remains stronger on the same trusted-side full-pool
diagnostic (13.6593/10.2101 in the corresponding reconstruction), so Chronos
is recorded as a reproducible pretrained comparison route rather than being
presented as an improvement or SOTA claim. Existing H38 engineering scores
and formal manifests are unchanged.
