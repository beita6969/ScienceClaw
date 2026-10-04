# Tool routing notes

Prefer a disclosed frozen tool when its checkpoint and preprocessing contract
are available. Use a structure-only or CPU baseline as a control, not as a
replacement for a claimed SOTA route. For remote tools, use the checked-in
Slurm launcher and content-addressed cache; do not make an agent shell call
with arbitrary paths.

When a task exposes `score_dev`, use it for selection. The formal evaluator is
only for an approved, mutually exclusive server-side batch. Record a tool's
provenance output and connect the final prediction directly to `submit.y`.
