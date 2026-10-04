#!/bin/bash
# long-lived srun step: the node-local GPU-tool broker (exec in foreground so the step lives as long as the broker)
F=/leonardo_scratch/fast/AIFAC_F02_774/rqian000; L=/leonardo_scratch/large/userexternal/rqian000; R=$L/sc-runs
cd $F/scienceclaw
mkdir -p "$R/remote_spool"
export PYTHONPATH=$F/scienceclaw SCIENCECLAW_DATA_ROOT=$L/scienceclaw-data/datasets SCIENCECLAW_MODELS=$L/models CUDA_VISIBLE_DEVICES=
# The two split batches start together and may also race this driver-launched
# broker.  Hold the same lock for the whole broker lifetime; a second broker
# exits cleanly instead of consuming the shared spool twice.
exec 9>"$R/remote_spool/.broker.lock"
/usr/bin/flock -n 9 || { echo "broker already owns $R/remote_spool"; exit 0; }
exec $F/envs/sc-harness/bin/python -u scripts/remote/broker.py --spool $R/remote_spool --local --gpu ${1:-2,3,2,3,2,3} --root $L/sc-tools \
  --code-dir $F/scienceclaw --python $F/envs/sc-harness/bin/python --models $L/models --hf-home $L/cache/hf --mlip-cache $L/sc-tools/mlip \
  --out-dir $L/sc-tools/remote_models --blobs $L/sc-tools/blobs
