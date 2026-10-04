# FoR52 adaptive visible psych route (2026-10-03)

## Implementation

`psych_adaptive_predict` is an additional tool in `for52_psych201.py`. It validates
the same two label-free payloads as `psych_fixed_predict`, runs grouped out-of-fold
accuracy on the visible `load_train` sessions for the fixed candidates
`qlearn`, `personal`, and `wsls`, selects the highest-scoring candidate with a
stable tie rule, and refits that candidate on the visible sessions. The returned
prediction and provenance can be wired directly to `submit.y`. No evaluation
target, scorer output, or hidden field is accepted by the route. The existing
formal qlearn route is unchanged.

Commit: `ed8b507`.

## Checks and server smoke

- Local FoR52 tests: **14 passed**.
- Leonardo `sc-run` compilation: passed; remote source SHA-256 is
  `33ea6ad0b8435e4463de50620f14fa23f7532d3f2a00167606dd112053c7fe59`.
- On existing single-card allocation `59277346` / node `lrdn2979`, direct
  route execution used the new disjoint OOD items from indices 16 and 17:
  - OOD-16 selected `qlearn`, accuracy **0.6250**.
  - OOD-17 selected `personal`, accuracy **0.7500**.

These are trusted route-smoke measurements: the predictor received only the
visible session payloads; the hidden evaluator was called afterward to measure
the returned output. They are not a formal SOTA batch. The ordinary H52E agent
probe on OOD-16..19 did not call the new tool (`uses=[]`) and is retained as
separate evidence, without relabelling it as adaptive-route performance.

The FoR52 ID pool supports exactly 16 disjoint 16-item episodes; indices 0..15
are already consumed by the existing batches, so there is no unused ID episode
for an adaptive rerun under the one-item-once rule. OOD has additional capacity.
