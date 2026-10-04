# FoR49 fixed Z3 route engineering batch D (2026-10-03)

- Runner: existing one-card Qwen service `59277346:37346` with the CPU-visible `z3_check -> submit.y` route.
- Episode selection: ID `04–07` and OOD `04–07`, disjoint from the earlier H49 `00–03` items. ID-05 completed before the parallel step was stopped; ID-04 and ID-07 plus all four OOD items completed in the corrected per-tag cache layout.

| split | completed item accuracies | mean | accepted | direct solver-to-submit |
|---|---|---:|---:|---:|
| ID | 0.8125 (04), 0.8750 (05), 0.8750 (07) | 0.8542 | 3/3 | 3/3 |
| OOD | 0.8125, 0.6250, 0.8750, 0.9375 | 0.8125 | 4/4 | 4/4 |

ID-06 was consumed but stopped after repeated deterministic solver timeouts
while the agent kept escalating Z3 limits; no final result was written and it
is retained as an incomplete item without rerun. The completed graphs all wire
`z3_check` output directly to `submit.y`. This remains engineering evidence;
there is no formal SOTA49 manifest yet.
