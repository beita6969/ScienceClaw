# FoR50 fixed route engineering batch C (2026-10-03)

- Runner: `H50C` through the existing one-card Qwen service `59277346:37346`.
- Episode selection: `skip=8`, ID/OOD episodes `08–11`; `SOTA50` and `SOTA50B` consumed `00–07`, so this batch is disjoint.
- Tool: fixed `fit_predict` route from `scilib.valueeval`, with visible train arguments/labels and label-free evaluation arguments only.

| split | F1 per episode | mean | accepted | direct tool-to-submit |
|---|---|---:|---:|---:|
| ID | 0.5334, 0.3993, 0.5114, 0.4498 | 0.4735 | 4/4 | 4/4 |
| OOD | 0.3033, 0.2496, 0.2770, 0.4486 | 0.3196 | 2/4 | 4/4 |

The two OOD failures are ordinary acceptance failures and are retained without
rerun. This is engineering evidence; it does not change the formal manifest or
make accuracy directly comparable to literature NLL results.
