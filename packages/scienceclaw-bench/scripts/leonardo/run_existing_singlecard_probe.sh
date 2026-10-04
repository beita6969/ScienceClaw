#!/bin/bash
# Run CPU-visible tool-ON probes inside an already running one-card Qwen job.
# usage: run_existing_singlecard_probe.sh <tag> <discipline> <port> [splits] [n] [skip] [workers]
set -euo pipefail
F=/leonardo_scratch/fast/AIFAC_F02_774/rqian000
L=/leonardo_scratch/large/userexternal/rqian000
TAG=${1:?tag}; DISC=${2:?discipline}; PORT=${3:?port}; SPLITS=${4:-id,ood}; N=${5:-4}; SKIP=${6:-0}; W=${7:-8}
MODEL=${MODEL:-}
POLICY_MAX_TOKENS=${POLICY_MAX_TOKENS:-8000}
EXECUTOR_MAX_TOKENS=${EXECUTOR_MAX_TOKENS:-2000}
PATCH_MAX_TOKENS=${PATCH_MAX_TOKENS:-4000}
SOLVER_MAX_STEPS=${SOLVER_MAX_STEPS:-24}
[[ "$TAG" =~ ^[A-Za-z0-9_.-]+$ && "$DISC" =~ ^[A-Za-z0-9_,]+$ && "$PORT" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$SPLITS" =~ ^(src|val|id|ood)(,(src|val|id|ood))*$ ]] || exit 2
[[ "$N" =~ ^[1-9][0-9]*$ && "$SKIP" =~ ^[0-9]+$ && "$W" =~ ^[1-9][0-9]*$ ]] || exit 2
R=$L/sc-runs; mkdir -p "$R/logs" "$R/remote_spool"
if [[ "$TAG" == SOTA* ]]; then
  for split in ${SPLITS//,/ }; do
    if find "$R/toolon_${TAG}_${split}" -name result.json -print -quit 2>/dev/null | grep -q .; then
      echo "formal output already exists: $R/toolon_${TAG}_${split}; refusing to rerun consumed episodes" >&2
      exit 2
    fi
  done
fi
cd "$F/scienceclaw"
export HF_HOME=$L/cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$F/scienceclaw SCIENCECLAW_DATA_ROOT=$L/scienceclaw-data/datasets
export SCIENCECLAW_REMOTE_SPOOL=$R/remote_spool
export SCIENCECLAW_MLIP_CACHE=${SCIENCECLAW_MLIP_CACHE:-$L/sc-tools/mlip}
export SCIENCECLAW_MAX_SUBPROCS=8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
curl -fsS --max-time 10 "http://127.0.0.1:${PORT}/health" >/dev/null
if [ -z "$MODEL" ]; then
  MODEL=$(curl -fsS --max-time 10 "http://127.0.0.1:${PORT}/v1/models" \
    | "$F/envs/sc-run/bin/python" -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
  [ -n "$MODEL" ] || { echo "endpoint returned no model id" >&2; exit 2; }
fi
# Read required tool refs from the same manifest used by the launch gate.  The
# probe turns these into an explicit objective requirement so a formal run
# cannot silently choose a weaker sibling ToolSpec.
required_tool_args=()
if [[ "$TAG" == SOTA* ]]; then
  mapfile -t required_refs < <("$F/envs/sc-run/bin/python" - "$F/scienceclaw/configs/formal_toolon_manifest.json" "$TAG" <<'PY'
import json, sys
manifest, tag = sys.argv[1:]
for ref in next(row for row in json.load(open(manifest))['batches'] if row['tag'] == tag).get('required_tool_refs', []):
    print(ref)
PY
  )
  for ref in "${required_refs[@]}"; do
    [ -n "$ref" ] && required_tool_args+=(--required-tool "$ref")
  done
fi
if [[ -n "${REQUIRED_TOOL:-}" ]]; then
  required_tool_args+=(--required-tool "$REQUIRED_TOOL")
fi
# Multiple CPU-visible probes may share one long-lived Qwen allocation.  Keep
# their configs and SQLite caches disjoint by including the validated tag;
# otherwise concurrent runs can block each other while opening the same cache.
CFG=$R/config_existing_${TAG}_${SLURM_JOB_ID}_${PORT}.yaml
cat > "$CFG" <<YAML
name: existing_singlecard_toolon
runs_root: $R
llm:
  endpoints: [http://127.0.0.1:${PORT}/v1]
  policy: {model: $MODEL, max_tokens: $POLICY_MAX_TOKENS, temperature: 0.0, reasoning_effort: null, json_mode: true, extra_body: {chat_template_kwargs: {enable_thinking: false}}}
  executor: {model: $MODEL, max_tokens: $EXECUTOR_MAX_TOKENS, temperature: 0.0, reasoning_effort: null, json_mode: false, extra_body: {chat_template_kwargs: {enable_thinking: false}}}
  patch: {model: $MODEL, max_tokens: $PATCH_MAX_TOKENS, temperature: 0.0, reasoning_effort: null, json_mode: false, extra_body: {chat_template_kwargs: {enable_thinking: false}}}
  concurrency: 32
  timeout_s: 600
  cache_path: $R/llm_cache_existing_${TAG}_${SLURM_JOB_ID}_${PORT}.sqlite
  use_cache: true
solver: {max_steps: $SOLVER_MAX_STEPS}
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
trap 'rm -f "$CFG"' EXIT
for split in ${SPLITS//,/ }; do
  "$F/envs/sc-run/bin/python" scripts/probe.py --config "$CFG" --disciplines "$DISC" --split "$split" \
    --n "$N" --skip "$SKIP" --workers "$W" "${required_tool_args[@]}" --out "$R/toolon_${TAG}_${split}" \
    > "$R/logs/toolon_${TAG}_${split}_existing_${SLURM_JOB_ID}.log" 2>&1
  if [[ "$TAG" == SOTA* ]]; then
    VALIDATOR=$F/scienceclaw/scripts/leonardo/migration/validate_formal_results.py
    [ -r "$VALIDATOR" ] || { echo "missing formal post-run validator: $VALIDATOR" >&2; exit 2; }
    SCIENCECLAW_REPO=$F/scienceclaw SCIENCECLAW_RUN_ROOT=$R \
      "$F/envs/sc-run/bin/python" "$VALIDATOR" "$TAG" "$R" --split "$split" || {
        echo "formal post-run evidence failed for $TAG/$split" >&2
        exit 2
      }
  fi
done
echo "DONE $(date) $TAG $DISC $SPLITS"
