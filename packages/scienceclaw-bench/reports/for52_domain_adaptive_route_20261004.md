# FoR52 domain-adaptive psych route (2026-10-04)

## Why this route was added

The fixed formal route is q-learning for every item.  A trusted-side full-pool
comparison on the current reconstructed index showed a stable split-specific
pattern: q-learning is strongest when the evaluation study is represented in
the visible training sessions, while the pooled history GBDT is stronger for
held-out study IDs.  The new route uses that observable study-ID boundary; it
does not receive a split flag, targets, scorer output, or hidden fields.

The route calls `scilib.psych.fit_predict(method="all")` once on the two
label-free tool payloads.  It chooses `qlearn` for an item whose `study` occurs
in `load_train.sessions`, and `gbdt` otherwise.  It returns the selected legal
keys and provenance (`seen_method`, `unseen_method`, visible-study count, and
item counts) so the tool-to-submit edge remains auditable.  The existing
`psych_fixed_predict` q-learning route and all formal manifests are unchanged.

## Full-pool trusted-side comparison

The comparison used all visible training sessions (144), every item in each
trusted pool, and the stored targets only on the trusted side for this audit.
The target-free route itself saw only session histories, options, and study IDs.

| split | items | studies represented in train | qlearn | gbdt | auto | domain route |
|---|---:|---:|---:|---:|---:|---:|
| src | 481 | 481 | 0.63202 | 0.62786 | 0.64241 | 0.63202 |
| val | 132 | 132 | 0.66667 | 0.61364 | 0.63636 | 0.66667 |
| id | 263 | 263 | 0.63498 | 0.55894 | 0.62357 | **0.63498** |
| ood | 544 | 0 | 0.62868 | **0.65625** | 0.64706 | **0.65625** |

These are full-pool diagnostics, not new formal agent results.  They support a
single label-free route that preserves the current IID behavior and raises the
held-out-study diagnostic by 2.76 percentage points relative to fixed qlearn.
The OOD pool is still a within-corpus proxy split, so the result is not called
external SOTA.

## Validation and deployment

- `tests/test_task_FoR52.py tests/test_scilib_psych.py`: **33 passed**.
- `py_compile` of the FoR52 adapter: passed.
- The new tool validates the existing strict schemas and rejects target-bearing
  fields through the same `_validate_psych_inputs` path as the fixed route.
- No formal ID/OOD item was rerun and no scorer, acceptance margin, or hidden
  label surface was changed.

The route is available to future tool-on episodes as
`psych_domain_adaptive_predict`; formal manifests still require the existing
fixed route until a fresh, disjoint allocation is available.
