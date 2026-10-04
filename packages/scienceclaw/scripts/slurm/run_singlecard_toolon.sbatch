#!/bin/bash
# Single-A100 tool-ON runner for CPU-visible tools (for example FoR38 macro).
# The same card hosts one Qwen endpoint and the probe; no GPU broker is started.
# Submit with: sbatch --export=ALL,DISC=FoR38,TAG=H38,PORT=37400,N=4,W=8 \
#   --time=4-00:00:00 scripts/slurm/run_singlecard_toolon.sbatch
# A split is selected by SPLITS=id,ood; each split is launched once with SKIP.
# This script is deliberately separate from the four-GPU broker launcher: it
# makes the single-card long-task route explicit and prevents a CPU-only tool
# from reserving three unused GPUs.
# Log files: pass -o/-e to sbatch (default: slurm-%j.out in the submit directory).
set -euo pipefail

F=${SCIENCECLAW_WORK_ROOT:?set SCIENCECLAW_WORK_ROOT}
L=${SCIENCECLAW_STORE_ROOT:-$F}
DISC=${DISC:-FoR38}
TAG=${TAG:-H38}
PORT=${PORT:-37400}
N=${N:-4}
W=${W:-8}
SKIP=${SKIP:-0}
SPLITS=${SPLITS:-id,ood}
MODEL=${MODEL:-Qwen3.8-27B-FP8}
NAME=${NAME:-sc-llm}

[[ "$DISC" =~ ^[A-Za-z0-9_,]+$ ]] || { echo "invalid DISC=$DISC" >&2; exit 2; }
[[ "$TAG" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid TAG=$TAG" >&2; exit 2; }
[[ "$PORT" =~ ^[1-9][0-9]*$ && "$N" =~ ^[1-9][0-9]*$ && "$W" =~ ^[1-9][0-9]*$ && "$SKIP" =~ ^[0-9]+$ ]] || {
  echo "PORT/N/W/SKIP must be numeric" >&2; exit 2;
}
[[ "$SPLITS" =~ ^(src|val|id|ood)(,(src|val|id|ood))*$ ]] || { echo "invalid SPLITS=$SPLITS" >&2; exit 2; }

R=$L/sc-runs
mkdir -p "$R/logs" "$R/remote_spool"
cd "$F/scienceclaw"
export HF_HOME=$L/cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT=$L/sc-serve/vllm-cache
export CPATH=$F/pyinc/python3.11${CPATH:+:$CPATH}
export PATH=$F/envs/sc-vllm/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0
export TRITON_CACHE_DIR=$L/sc-serve/triton-cache
export PYTHONPATH=$F/scienceclaw
export SCIENCECLAW_DATA_ROOT=$L/scienceclaw-data/datasets
export SCIENCECLAW_MODELS=$L/models
export SCIENCECLAW_REMOTE_SPOOL=$R/remote_spool
export SCIENCECLAW_MLIP_CACHE=$L/sc-tools/mlip
export SCIENCECLAW_MAX_SUBPROCS=8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

CFG=$R/config_singlecard_${SLURM_JOB_ID}.yaml
cat > "$CFG" <<YAML
name: singlecard_toolon
runs_root: $R
llm:
  endpoints: [http://127.0.0.1:${PORT}/v1]
  policy: {model: $NAME, max_tokens: 8000, temperature: 0.0, reasoning_effort: null, json_mode: true, extra_body: {chat_template_kwargs: {enable_thinking: false}}}
  executor: {model: $NAME, max_tokens: 2000, temperature: 0.0, reasoning_effort: null, json_mode: false, extra_body: {chat_template_kwargs: {enable_thinking: false}}}
  patch: {model: $NAME, max_tokens: 4000, temperature: 0.0, reasoning_effort: null, json_mode: false, extra_body: {chat_template_kwargs: {enable_thinking: false}}}
  concurrency: 32
  timeout_s: 600
  cache_path: $R/llm_cache_singlecard_${SLURM_JOB_ID}.sqlite
  use_cache: true
solver: {max_steps: 24}
evolution: {variant: full, qval: macrosr_then_score, qval_eps: 0.02}
bench:
  disciplines: []
  items_per_episode: 16
  rounds: 2
  n_val: 2
  n_id: 4
  n_ood: 4
  seed: 20260928
  data_root: $L/scienceclaw-data/datasets
YAML

ENDPOINT="$L/sc-serve/endpoints/${SLURM_JOB_ID}"
mkdir -p "$L/sc-serve/endpoints" "$L/sc-serve/logs"
rm -f "$ENDPOINT" "$ENDPOINT.starting"
trap 'rm -f "$CFG" "$ENDPOINT" "$ENDPOINT.starting"; kill ${SRV:-0} 2>/dev/null || true' EXIT
echo "$(hostname):$PORT $NAME $MODEL $(date +%s)" > "$ENDPOINT.starting"

"$F/envs/sc-vllm/bin/vllm" serve "$L/models/$MODEL" --served-model-name "$NAME" --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size 1 --max-model-len 65536 --max-num-seqs 48 --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --trust-remote-code > "$L/sc-serve/logs/${SLURM_JOB_ID}.out" 2>&1 &
SRV=$!
until curl -fsS --max-time 10 "http://127.0.0.1:${PORT}/health" >/dev/null; do
  kill -0 "$SRV" 2>/dev/null || { echo "vLLM exited before health" >&2; exit 1; }
  sleep 5
done
mv "$ENDPOINT.starting" "$ENDPOINT"
echo "READY $(date) host=$(hostname) port=$PORT"

for split in ${SPLITS//,/ }; do
  "$F/envs/sc-run/bin/python" scripts/probe.py --config "$CFG" --disciplines "$DISC" --split "$split" \
    --n "$N" --skip "$SKIP" --workers "$W" --out "$R/toolon_${TAG}_${split}" \
    > "$R/logs/toolon_${TAG}_${split}_${SLURM_JOB_ID}.log" 2>&1
done
echo "DONE $(date) $TAG $DISC $SPLITS"
