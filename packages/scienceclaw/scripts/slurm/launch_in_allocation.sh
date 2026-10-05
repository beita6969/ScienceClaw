#!/bin/bash
# Start one single-GPU vLLM server per A100 inside an already running 4-GPU allocation (no new Slurm job).
# usage (on the login node): launch_in_allocation.sh <jobid> [n_gpus=4] [port_base=21000]
# Each server publishes $L/sc-serve/endpoints/<jobid>-g<i> once /health answers; locally run scripts/slurm/tunnel.sh.
JOB=${1:?jobid}; N=${2:-4}; PB=${3:-21000}
F=${SCIENCECLAW_WORK_ROOT:?set SCIENCECLAW_WORK_ROOT}; L=${SCIENCECLAW_STORE_ROOT:-$F}
mkdir -p $L/sc-serve/logs
for g in $(seq 0 $((N-1))); do
  G=$g EPID=$JOB-g$g PORT=$((PB+g)) MODEL=${MODEL:?set MODEL} NAME=${NAME:-sc-llm} TP=1 MAXLEN=${MAXLEN:-65536} MAXSEQS=${MAXSEQS:-48} \
  nohup srun --jobid=$JOB --overlap -N1 -n1 -c8 --gres=gpu:a100:4 --export=ALL \
    bash -c 'export CUDA_VISIBLE_DEVICES=$G; exec bash '$F'/sc-serve/serve_vllm.sbatch' > $L/sc-serve/logs/$JOB-g$g.out 2>&1 < /dev/null &
  sleep 25
done
echo launched $N servers in job $JOB
