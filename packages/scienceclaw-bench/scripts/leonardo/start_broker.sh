#!/bin/bash
# Remote-tool broker for the GPU tools (scilib.tsfm/audioenc/textenc/proteinplm/phenoseg_sam/...) running inside a Leonardo allocation.
# usage: scripts/leonardo/start_broker.sh <jobid> <gpu list inside the allocation, e.g. 2,3>
cd "$(dirname "$0")/../.." || exit 1
JOB=${1:?jobid}; GPUS=${2:-2,3}
F=/leonardo_scratch/fast/AIFAC_F02_774/rqian000; L=/leonardo_scratch/large/userexternal/rqian000
PYTHONPATH="$PWD" exec nice -n 5 .venv/bin/python -u scripts/remote/broker.py --spool cache/remote_spool --host leonardo \
  --ssh-socket ~/.ssh/cm-sc-leo --srun-job "$JOB" --gpu "$GPUS" --root $L/sc-tools --code-dir $F/scienceclaw \
  --python $F/envs/sc-harness/bin/python --models $L/models --hf-home $L/cache/hf --mlip-cache $L/sc-tools/mlip \
  --out-dir $L/sc-tools/remote_models --blobs $L/sc-tools/blobs
