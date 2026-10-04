#!/bin/bash
# Self-hosted OpenAI-compatible LLM server for ScienceClaw on one GPU node.
# EPID=<name> overrides the endpoint file name (several servers inside one allocation, started with srun --overlap).
# Submit with: MODEL=<dir under models/> NAME=<served name> TP=<gpus> sbatch --job-name=workN-sc-vllm -A <account> \
#   -p <partition> --qos=normal -N 1 --gres=gpu:$TP -c $((8*TP)) --mem=$((100*TP))G --time=... --time-min=... serve_vllm.sbatch
# Log files: pass -o/-e to sbatch (default: slurm-%j.out in the submit directory).
set -u
F=${SCIENCECLAW_WORK_ROOT:?set SCIENCECLAW_WORK_ROOT}
L=${SCIENCECLAW_STORE_ROOT:-$F}
MODEL=${MODEL:-Qwen3.8-27B-FP8}; NAME=${NAME:-sc-llm}; TP=${TP:-${SLURM_GPUS_ON_NODE:-1}}
MAXLEN=${MAXLEN:-65536}; MAXSEQS=${MAXSEQS:-48}; PORT=${PORT:-$((20000 + SLURM_JOB_ID % 20000))}; UTIL=${UTIL:-0.90}
mkdir -p $L/sc-serve/endpoints $L/sc-serve/logs
export HF_HOME=$L/cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_CACHE_ROOT=$L/sc-serve/vllm-cache
# Triton compiles a small C helper at start-up; the system python has no Python.h, so point gcc at extracted CPython headers
export CPATH=$F/pyinc/python3.11${CPATH:+:$CPATH}
# no nvcc on the compute nodes' PATH: FlashInfer JIT paths (sampler, all-reduce) would fail, use the native implementations
export PATH=$F/envs/sc-vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0
export TMPDIR=${TMPDIR:-/tmp} TRITON_CACHE_DIR=$L/sc-serve/triton-cache
# no `module load cuda`: the cu129 pip wheels bring their own CUDA libs and a module would shadow them
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$F/envs/sc-vllm/bin/python -c "import torch,vllm;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available(),'vllm',vllm.__version__)"
EP=$L/sc-serve/${EPDIR:-endpoints}/${EPID:-$SLURM_JOB_ID}; mkdir -p $(dirname $EP)
trap 'rm -f $EP $EP.starting; kill ${SRV:-0} 2>/dev/null' EXIT
echo "$(hostname):$PORT $NAME $MODEL $(date +%s)" > $EP.starting
$F/envs/sc-vllm/bin/vllm serve $L/models/$MODEL --served-model-name $NAME --host 0.0.0.0 --port $PORT \
  --tensor-parallel-size $TP --max-model-len $MAXLEN --max-num-seqs $MAXSEQS --gpu-memory-utilization $UTIL \
  --enable-prefix-caching --trust-remote-code ${VLLM_EXTRA:-} &
SRV=$!
# publish the endpoint only once the server answers, so tunnel.sh never forwards to a still-loading server
( until curl -sf http://127.0.0.1:$PORT/health >/dev/null; do kill -0 $SRV 2>/dev/null || exit 1; sleep 5; done
  mv $EP.starting $EP; echo "READY $(date)" ) &
wait $SRV
