# FoR51 visible-dev model selection (2026-10-04)

This is a label-isolated engineering comparison on the existing FoR51 `val` episode. It uses only the visible `load_train` structures/targets, frozen SevenNet feature-cache rows (`mesh=8`, `min_len=7`, `disp=0.01`), and the task's `score_dev` interface. No hidden ID/OOD target was read and no formal item was touched.

On the two deterministic `val` plans (same dev payload and reference), the current concatenated structure + SevenNet feature route gave:

| route | dev MAE (cm^-1) | visible reference MAE |
|---|---:|---:|
| ExtraTrees log-target ensemble (current `fit_sevennet_mlip`) | 24.2811 | 121.9300 |
| HistGradientBoosting log-target ensemble | 41.7812 | 121.9300 |
| fixed 50/50 ExtraTrees + Hist blend | 32.0593 | 121.9300 |
| fixed 60/40 ExtraTrees + Hist blend | 30.3517 | 121.9300 |

The result is identical on both repeated val plans because they expose the same deterministic dev pool. The existing ExtraTrees route is the strongest of these directly comparable frozen-feature options; a blend would reduce visible-dev performance. Therefore no unvalidated ensemble or hyperparameter change is promoted into the formal route. The existing SevenNet formal evidence and the CHGNet diagnostic route remain unchanged.
