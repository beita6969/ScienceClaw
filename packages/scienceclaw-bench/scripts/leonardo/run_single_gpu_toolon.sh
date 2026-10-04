#!/bin/bash
# Run one ScienceClaw tool-on shard from a standalone 1-GPU allocation.
# Two independent 1-GPU Qwen jobs publish endpoint markers in SERVICE_DIR;
# this job waits for them, starts one GPU broker on GPU 0, and runs one shard.
set -euo pipefail
F=/leonardo_scratch/fast/AIFAC_F02_774/rqian000
L=/leonardo_scratch/large/userexternal/rqian000
TAG=${1:?tag is required}
DISC=${2:?disciplines are required}
SPLIT=${3:?split is required}
N=${4:-4}
W=${5:-8}
SKIP=${SKIP:-0}
RUN_ID=${RUN_ID:-${TAG}_${SPLIT}_sg$(date +%s)}
SPOOL_ID=${SPOOL_ID:-$RUN_ID}
if [ -n "${OUT_ROOT:-}" ]; then
  OUT_ROOT=$OUT_ROOT
elif [[ "$TAG" == SOTA* ]]; then
  OUT_ROOT="$L/sc-runs/toolon_${TAG}_${SPLIT}"
else
  OUT_ROOT="$L/sc-runs/$SPOOL_ID/toolon_${TAG}_${SPLIT}"
fi
SERVICE_DIR=${SERVICE_DIR:-single-gpu-qwen-l4-20261003}
# Keep IDs in separate variables: Slurm --export uses commas as separators.
SERVICE_ID_A=${SERVICE_ID_A:-59264793}
SERVICE_ID_B=${SERVICE_ID_B:-59264794}
SERVICE_IDS=${SERVICE_IDS:-$SERVICE_ID_A;$SERVICE_ID_B}
WAIT_S=${ENDPOINT_WAIT_S:-345600}
R="$L/sc-runs/$SPOOL_ID"
SPOOL="$R/remote_spool"
mkdir -p "$R/logs" "$SPOOL"
cd "$F/scienceclaw"
if [[ "$TAG" == SOTA* ]]; then
  MANIFEST_CHECK=$F/sc-tools/check_formal_manifest.py
  [ -r "$MANIFEST_CHECK" ] || { echo "missing formal manifest checker: $MANIFEST_CHECK" >&2; exit 2; }
  SCIENCECLAW_REPO=$F/scienceclaw "$F/envs/sc-run/bin/python" "$MANIFEST_CHECK" "$DISC" "$TAG" "$SPLIT" "$N" --skip "$SKIP"
  if find "$OUT_ROOT" -name result.json -print -quit 2>/dev/null | grep -q .; then
    echo "formal output already exists: $OUT_ROOT; refusing to overwrite existing episodes" >&2
    exit 2
  fi
  if [[ "$TAG" == SOTA51* ]]; then
    MLIP_AUDIT=$F/scienceclaw/scripts/f51/audit_mlip_cache.py
    [ -r "$MLIP_AUDIT" ] || { echo "missing MLIP cache auditor: $MLIP_AUDIT" >&2; exit 2; }
    PYTHONPATH=$F/scienceclaw:$F/sv_pkgs SCIENCECLAW_DATA_ROOT=$L/scienceclaw-data/datasets \
      SCIENCECLAW_MLIP_CACHE=$L/sc-tools/mlip "$F/envs/sc-harness/bin/python" "$MLIP_AUDIT" \
      --model sevennet --require-complete
  fi
fi
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
export PYTHONPATH=$F/scienceclaw SCIENCECLAW_DATA_ROOT=$L/scienceclaw-data/datasets
export SCIENCECLAW_REMOTE_SPOOL=$SPOOL SCIENCECLAW_MLIP_CACHE=$L/sc-tools/mlip
export SCIENCECLAW_MAX_SUBPROCS=${SUBPROCS:-8} OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ffmpeg_glob=("$F"/envs/sc-run/lib64/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-*)
if [ -f "${ffmpeg_glob[0]:-}" ]; then export SCIENCECLAW_FFMPEG="${ffmpeg_glob[0]}"; fi
IFS=',;' read -r -a service_ids <<< "$SERVICE_IDS"
IFS=',' read -r -a endpoint_urls <<< ""
deadline=$(( $(date +%s) + WAIT_S ))
while :; do
  endpoint_urls=()
  ready=1
  for sid in "${service_ids[@]}"; do
    marker="$L/sc-serve/$SERVICE_DIR/$sid"
    if [ ! -s "$marker" ]; then ready=0; break; fi
    spec=$(awk 'NR==1{print $1}' "$marker")
    host=${spec%:*}; port=${spec##*:}
    if ! curl -fsS --max-time 10 "http://$spec/health" >/dev/null; then ready=0; break; fi
    endpoint_urls+=("http://$host:$port/v1")
  done
  if [ "$ready" -eq 1 ]; then break; fi
  [ "$(date +%s)" -lt "$deadline" ] || { echo "timed out waiting for Qwen markers/health in $SERVICE_DIR" >&2; exit 1; }
  sleep 30
done
endpoints=$(IFS=,; echo "${endpoint_urls[*]}")
CONFIG="$R/toolon_leo.yaml"
sed "s#^[[:space:]]*endpoints:.*#  endpoints: [$endpoints]#" configs/toolon_leo.yaml > "$CONFIG"
sed -i "s|^  cache_path:.*|  cache_path: $R/llm_cache.sqlite|" "$CONFIG"
export SCIENCECLAW_CONFIG_PATH="$CONFIG"
# One broker owns this shard's spool and uses this allocation's only GPU.
exec 9>"$SPOOL/.broker.lock"
/usr/bin/flock -n 9 || { echo "broker lock is already held: $SPOOL" >&2; exit 2; }
"$F/envs/sc-harness/bin/python" -u scripts/remote/broker.py --spool "$SPOOL" --local --gpu 0 \
  --root "$L/sc-tools" --code-dir "$F/scienceclaw" --python "$F/envs/sc-harness/bin/python" \
  --models "$L/models" --hf-home "$L/cache/hf" --mlip-cache "$L/sc-tools/mlip" \
  --out-dir "$L/sc-tools/remote_models" --blobs "$L/sc-tools/blobs" > "$R/logs/broker.log" 2>&1 &
broker_pid=$!
cleanup() { kill "$broker_pid" 2>/dev/null || true; }
trap cleanup EXIT
sleep 5
"$F/envs/sc-run/bin/python" scripts/probe.py --config "$CONFIG" --disciplines "$DISC" --split "$SPLIT" --n "$N" --skip "$SKIP" --workers "$W" "${required_tool_args[@]}" --out "$OUT_ROOT"

if [[ "$TAG" == SOTA* ]]; then
  VALIDATOR=$F/sc-tools/validate_formal_results.py
  [ -r "$VALIDATOR" ] || { echo "missing formal post-run validator: $VALIDATOR" >&2; exit 2; }
  SCIENCECLAW_REPO=$F/scienceclaw SCIENCECLAW_RUN_ROOT=$L/sc-runs \
    "$F/envs/sc-run/bin/python" "$VALIDATOR" "$TAG" "$L/sc-runs" --split "$SPLIT"
fi
