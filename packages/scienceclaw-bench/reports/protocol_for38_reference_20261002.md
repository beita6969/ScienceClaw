# FoR38 reference comparability check (2026-10-02)

This note adds a trusted-side reference diagnostic for the World Bank macro task.  The official primary and
acceptance rule are unchanged: the submitted forecast is scored by mean sMAPE against the per-episode random-walk
reference with the existing 3% margin.

`damped_trend` fits a line to the last eight finite history values and damps the extrapolated increments.  The
`history_backtest_forecast` diagnostic selects between that forecast and the random walk using only rolling four-step
backtests wholly inside the visible history.  It never reads a post-origin value and is not exposed by any tool.
The evaluator payload records the official reference, the damped-trend candidate, and the history-selected
candidate.  `Adapter.pooled_diagnostics` aggregates these values over all supplied episodes, so a 64-item full-split
comparison does not mix a 16-item slice with a leaderboard number.

With the local recorded WDI data and seed 0, four id episodes (64 items) give:

| reference | pooled mean sMAPE (%) |
| --- | ---: |
| official random walk | 17.62823 |
| history-selected candidate | 16.82989 |

The same four OOD episodes give 14.52940 versus 15.25153, respectively.  The diagnostic therefore identifies an
IID improvement but an OOD regression; it is evidence for comparison, not a claim that this reference is SOTA or a
reason to relax acceptance.

Validation: `./.venv/bin/python -m pytest -q tests/test_task_for38.py tests/test_task_forecast_common.py` (26 passed;
only the existing NumPy timedelta deprecation warnings).
