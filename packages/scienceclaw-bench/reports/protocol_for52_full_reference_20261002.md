# FoR52 full-pool reference audit (2026-10-02)

`Psych201Adapter.full_split_reference(split)` adds a trusted-side, full-pool comparison for the rebuilt Psych-201
splits. It scores the existing participant-history reference and the study-conditioned reference using every item in
the requested split and the complete visible `train` pool. The method is diagnostic-only: it is not exposed through
an episode tool, does not change `evaluate`, `pooled_metric`, acceptance, or hidden-label visibility, and reads
targets only to form the audit aggregate.

Using the current verified index (144 visible training sessions across 24 IID studies), the full-pool values are:

| split | items | participant-history | study-conditioned |
| --- | ---: | ---: | ---: |
| src | 481 | 0.58836 | 0.48025 |
| val | 132 | 0.62121 | 0.50000 |
| id | 263 | 0.52852 | 0.47909 |
| ood | 544 | 0.60662 | 0.60662 |

The study-conditioned route does not improve the full id pool and falls back to participant history on OOD studies;
it therefore remains a comparison diagnostic rather than a replacement reference or a leaderboard claim. This closes
the comparability gap where a 16-item sampled episode was previously the only aggregate for the stronger reference.

Validation: `./.venv/bin/python -m py_compile scienceclaw/bench/tasks/for52_psych201.py tests/test_task_FoR52.py` and
`./.venv/bin/pytest -q tests/test_task_FoR52.py tests/test_task_for38.py tests/test_task_for45.py
tests/test_task_FoR49.py tests/test_task_for50.py tests/test_task_utils_for36_46_49_52.py` (59 passed).
