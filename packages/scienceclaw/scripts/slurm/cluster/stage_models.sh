#!/bin/bash
# Stage the pretrained tool weights on the cluster from the registry (scienceclaw/tools/weights.json).
# Layout = $SCIENCECLAW_MODELS/<subdir>, as scilib._pretrained expects. Pass asset ids to stage a subset.
set -u
F=${SCIENCECLAW_WORK_ROOT:?set SCIENCECLAW_WORK_ROOT}; L=${SCIENCECLAW_STORE_ROOT:-$F}
export SCIENCECLAW_MODELS=${SCIENCECLAW_MODELS:-$L/models} HF_HOME=$L/cache/hf HF_HUB_ENABLE_HF_TRANSFER=0
export PATH=$F/envs/sc-vllm/bin:$PATH
mkdir -p "$SCIENCECLAW_MODELS" "$L/sc-tools/logs"
PYTHONPATH=$F/scienceclaw "$F/envs/sc-harness/bin/python" -m scienceclaw.cli weights plan "$@" | nice -n 10 bash
PYTHONPATH=$F/scienceclaw "$F/envs/sc-harness/bin/python" -m scienceclaw.cli weights status
