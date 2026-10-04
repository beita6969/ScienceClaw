# FoR37 Granite TTM frequency decision (2026-10-03)

This note records a read-only compatibility check for the staged Granite TTM r2
checkpoint.  It uses only the public `load_train` and `load_dev` outputs of the
FoR37 adapter.  It does not inspect an ID/OOD target, fit a model, change the
FoR37 evaluator, or create a formal manifest.

## Observed task cadence and visible inputs

The Leonardo check was run with the repository checkout and data root selected
by the normal tool-on environment:

```text
PYTHONPATH=$F/scienceclaw
SCIENCECLAW_DATA_ROOT=$L/scienceclaw-data/datasets
$F/envs/sc-run/bin/python
```

For one `src` and one `val` episode, the script called only
`ep.tool("load_train").fn({}, {})` and `ep.tool("load_dev").fn({}, {})` and
reported:

```text
src n_items 8 train_shape (1336, 64, 32) train_steps 1336 train_unique_dt_h [6]
src dev_context_shape (16, 4, 64, 32)
val n_items 8 train_shape (1336, 64, 32) train_steps 1336 train_unique_dt_h [6]
val dev_context_shape (16, 4, 64, 32)
context_finite True context_range 227.3588409423828 316.3547668457031
```

Thus the visible series is 6-hourly.  The forecast task is a 24-hour lead,
which is four native steps.  The dev interface exposes four fields at offsets
`-18, -12, -6, 0` hours; it does not expose a contiguous 512-hour history for
each evaluation initialisation.

## Checkpoint contract

The staged checkpoint is

```text
$L/p3-stage-20261003/granite/model.safetensors
bytes  3240592
sha256 a706726a7eb01bbcb42994b7dcb3c06ea9557898dbae8d480eb04fe8ccb89710
```

Its local `config.json` fixes `context_length=512`, `prediction_length=96`,
and `num_input_channels=1`.  The checkpoint README says that r2 supports
minutely and hourly resolutions and explicitly warns against upsampling or
zero-padding to increase the context.  The current read-only wrapper therefore
correctly exposes only `(n, 512, 1) -> (n, 96, 1)` smoke inference and does not
claim a FoR37 frequency adapter.

## Candidate policies and decision

* Passing the native 6-hour series unchanged would make the 512-point context
  cover 128 days and the 96-point output cover 24 days.  This does not match
  FoR37's 24-hour target and is outside the checkpoint's documented resolution
  range.
* Interpolating/repeating 6-hour fields to hourly cadence would create 512
  synthetic hourly points from only the four visible dev context fields.  It is
  explicitly discouraged by the model card and would introduce a new protocol
  choice that has not been fixed on `src`/`val`.  The formal evaluator does not
  expose the missing continuous history needed to make this a faithful 512-hour
  input for the 2019 ID and 2020 temporal-proxy OOD episodes.
* Taking the first four of the 96 native outputs would preserve a 24-hour
  horizon, but still feeds an unsupported 6-hour frequency and leaves the
  model's 96-step training contract unexplained.  It cannot be called a fair
  pretrained SOTA component without a frequency-tuned checkpoint or a
  predeclared, src/val-validated conversion protocol.

**Decision:** keep Granite as `smoke_validated_unintegrated`.  Do not add it to
the FoR37 formal tool pool or score it.  A safe next step requires either a
checkpoint documented for 6-hour data (with a 4-step output) or an explicitly
approved adapter protocol that is fixed and measured on visible `src`/`val`
before any new, non-overlapping formal episodes.  The current wrapper and
formal evaluator remain unchanged.

## Reproduction

The observation above can be reproduced on Leonardo with:

```bash
ssh leonardo
F=/leonardo_scratch/fast/AIFAC_F02_774/rqian000
V=$F/envs/sc-run/bin/python
export PYTHONPATH=$F/scienceclaw
export SCIENCECLAW_DATA_ROOT=/leonardo_scratch/large/userexternal/rqian000/scienceclaw-data/datasets
$V - <<'PY'
from scienceclaw.bench.tasks.for37_weatherbench import Adapter
import numpy as np
ad = Adapter()
for split in ("src", "val"):
    ep = ad.build_episodes(split, 1, 20261003, items_per_episode=8)[0]
    tr = ep.tool("load_train").fn({}, {})
    dv = ep.tool("load_dev").fn({}, {})
    t = np.array(tr["time"], dtype="datetime64[h]")
    print(split, tr["t2m"].shape, sorted(set(np.diff(t).astype(int).tolist())))
    print(split, dv["context"].shape, np.isfinite(dv["context"]).all())
PY
```

The adapter construction is only used to obtain the visible tool handles; the
script never calls `score_dev`, `load_eval_inputs`, or any target accessor.
