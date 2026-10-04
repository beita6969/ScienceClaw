#!/bin/bash
# Remote-tool broker for the GPU tools (scilib.tsfm/audioenc/textenc/proteinplm/phenoseg_sam/...) running inside a Slurm allocation. Set SCIENCECLAW_SSH_HOST to the ssh alias of the cluster login node.
# usage: scripts/slurm/start_broker.sh <jobid> <gpu list inside the allocation, e.g. 2,3>
cd "$(dirname "$0")/../.." || exit 1
JOB=${1:?jobid}; GPUS=${2:-2,3}
F=${SCIENCECLAW_WORK_ROOT:?set SCIENCECLAW_WORK_ROOT}; L=${SCIENCECLAW_STORE_ROOT:-$F}
PYTHONPATH="$PWD" exec nice -n 5 .venv/bin/python -u scripts/remote/broker.py --spool cache/remote_spool --host "${SCIENCECLAW_SSH_HOST:?set SCIENCECLAW_SSH_HOST}" \
  --ssh-socket ~/.ssh/cm-scienceclaw --srun-job "$JOB" --gpu "$GPUS" --root $L/sc-tools --code-dir $F/scienceclaw \
  --python $F/envs/sc-harness/bin/python --models $L/models --hf-home $L/cache/hf --mlip-cache $L/sc-tools/mlip \
  --out-dir $L/sc-tools/remote_models --blobs $L/sc-tools/blobs
