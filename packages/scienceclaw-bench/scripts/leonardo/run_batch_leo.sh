#!/bin/bash
# Run a tool-ON batch ON Leonardo inside a running allocation (use: srun --jobid=<J> --overlap -N1 -n1 -c24 --gres=gpu:a100:4 --export=ALL this-script ...).
# usage: run_batch_leo.sh <tag> <disciplines,comma> <split> <n> <workers> [tool_gpus=2,3]
# Set SCIENCECLAW_CONFIG_PATH to use a checked-in execution profile (for example
# configs/toolon_f32_budgeted.yaml); the default remains configs/toolon_leo.yaml.
set -euo pipefail
F=/leonardo_scratch/fast/AIFAC_F02_774/rqian000; L=/leonardo_scratch/large/userexternal/rqian000
TAG=${1:?tag is required}; DISC=${2:?disciplines are required}; SPLIT=${3:?split is required}; N=${4:-4}; W=${5:-12}; GPUS=${6:-2,3}; SKIP=${SKIP:-0}
# A caller-provided profile is checked before touching the remote checkout so a
# typo cannot fall through to the default profile or consume an allocation.
if [[ -n "${SCIENCECLAW_CONFIG_PATH:-}" && ! -r "$SCIENCECLAW_CONFIG_PATH" ]]; then
  echo "config file is not readable: $SCIENCECLAW_CONFIG_PATH" >&2
  exit 2
fi
[[ "$DISC" =~ ^[A-Za-z0-9_,]+$ ]] || { echo "invalid disciplines: $DISC" >&2; exit 2; }
[[ "$SPLIT" =~ ^(src|val|id|ood)$ ]] || { echo "invalid split: $SPLIT" >&2; exit 2; }
[[ "$N" =~ ^[1-9][0-9]*$ ]] || { echo "n must be a positive integer: $N" >&2; exit 2; }
[[ "$W" =~ ^[1-9][0-9]*$ ]] || { echo "workers must be a positive integer: $W" >&2; exit 2; }
[[ "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "invalid tool GPUs: $GPUS" >&2; exit 2; }
R=$L/sc-runs; mkdir -p $R/logs $R/remote_spool
cd $F/scienceclaw || exit 1
# AAC stems (FoR36) need ffmpeg: static binary bundled with the imageio-ffmpeg wheel in the run env
ffmpeg_glob=("$F"/envs/sc-run/lib64/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-*)
if [ -f "${ffmpeg_glob[0]:-}" ]; then
  export SCIENCECLAW_FFMPEG="${ffmpeg_glob[0]}"
else
  unset SCIENCECLAW_FFMPEG || true
fi
export PYTHONPATH=$F/scienceclaw SCIENCECLAW_DATA_ROOT=$L/scienceclaw-data/datasets SCIENCECLAW_REMOTE_SPOOL=$R/remote_spool \
       SCIENCECLAW_MLIP_CACHE=$L/sc-tools/mlip \
       SCIENCECLAW_MAX_SUBPROCS=${SUBPROCS:-24} OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CONFIG_PATH=${SCIENCECLAW_CONFIG_PATH:-configs/toolon_leo.yaml}
[ -r "$CONFIG_PATH" ] || { echo "config file is not readable: $CONFIG_PATH" >&2; exit 2; }

# Keep direct srun entry equivalent to leo_chain/leo_driver: a formal SOTA tag
# cannot bypass the immutable manifest by invoking this batch script directly.
MANIFEST_CHECK=$F/sc-tools/check_formal_manifest.py
if [ -r "$MANIFEST_CHECK" ]; then
  SCIENCECLAW_REPO=$F/scienceclaw "$F/envs/sc-run/bin/python" "$MANIFEST_CHECK" "$DISC" "$TAG" "$SPLIT" "$N" --skip "$SKIP"
elif [[ "$TAG" == SOTA* ]]; then
  echo "missing formal manifest checker: $MANIFEST_CHECK" >&2
  exit 2
fi

# A formal SevenNet run must not silently compute a partial cache on the first
# episode.  Debug/prewarm tags may populate the cache; only the immutable
# SOTA51 path is fail-closed on exact full-dataset coverage.
if [[ "$TAG" == SOTA51 ]]; then
  MLIP_AUDIT=$F/scienceclaw/scripts/f51/audit_mlip_cache.py
  [ -r "$MLIP_AUDIT" ] || { echo "missing MLIP cache auditor: $MLIP_AUDIT" >&2; exit 2; }
  PYTHONPATH=$F/scienceclaw:$F/sv_pkgs SCIENCECLAW_DATA_ROOT=$L/scienceclaw-data/datasets \
    SCIENCECLAW_MLIP_CACHE=$L/sc-tools/mlip "$F/envs/sc-harness/bin/python" "$MLIP_AUDIT" \
    --model sevennet --require-complete || {
      echo "SOTA51 requires a complete SevenNet cache before formal episodes" >&2
      exit 2
    }
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

# Marker files are published after /health answers, but this protects against
# stale markers or a crashed server before any broker request is created.
command -v curl >/dev/null 2>&1 || { echo "curl is required for endpoint health checks" >&2; exit 2; }
ENDPOINT_PORTS=${LEO_ENDPOINT_PORTS:-21000,21001}
[[ "$ENDPOINT_PORTS" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || { echo "invalid LEO_ENDPOINT_PORTS: $ENDPOINT_PORTS" >&2; exit 2; }
IFS=',' read -r -a endpoint_ports <<< "$ENDPOINT_PORTS"
for port in "${endpoint_ports[@]}"; do
  curl -fsS --max-time "${LEO_HEALTH_TIMEOUT_S:-10}" "http://127.0.0.1:${port}/health" >/dev/null || {
    echo "Qwen endpoint 127.0.0.1:${port} is not healthy; refusing to consume the shared spool" >&2
    exit 2
  }
done
if [[ ",$DISC," == *,FoR36,* ]] && { [ -z "${SCIENCECLAW_FFMPEG:-}" ] || [ ! -x "$SCIENCECLAW_FFMPEG" ]; }; then
  echo "FoR36 requested but the bundled ffmpeg is missing or not executable" >&2
  exit 2
fi
# one broker per allocation (idempotent): runs the GPU-tool worker on this node, one request per listed GPU
if ! pgrep -u $USER -f "broker.py --spool $R/remote_spool" > /dev/null; then
  # id and ood batches are launched concurrently.  The pgrep check alone has
  # a race: both shells can observe no broker and start duplicate consumers of
  # the same spool.  Keep an advisory lock open for the broker lifetime so
  # exactly one process owns this allocation's spool.
  nohup bash -c 'exec 9>"$1"; /usr/bin/flock -n 9 || exit 0; export CUDA_VISIBLE_DEVICES=; exec "$2" -u scripts/remote/broker.py --spool "$3" --local --gpu "$4" --root "$5" --code-dir "$6" --python "$2" --models "$7" --hf-home "$8" --mlip-cache "$9" --out-dir "${10}" --blobs "${11}"' \
    _ "$R/remote_spool/.broker.lock" "$F/envs/sc-harness/bin/python" "$R/remote_spool" "$GPUS" "$L/sc-tools" "$F/scienceclaw" "$L/models" "$L/cache/hf" "$L/sc-tools/mlip" "$L/sc-tools/remote_models" "$L/sc-tools/blobs" \
    > $R/logs/broker_$(hostname).log 2>&1 &
  sleep 3
fi
set +e
$F/envs/sc-run/bin/python scripts/probe.py --config "$CONFIG_PATH" --disciplines "$DISC" --split "$SPLIT" --n "$N" --skip "$SKIP" --workers "$W" "${required_tool_args[@]}" \
  --out $R/toolon_${TAG}_${SPLIT} > $R/logs/toolon_${TAG}_${SPLIT}.log 2>&1
probe_rc=$?
set -e
if [ "$probe_rc" -ne 0 ]; then
  echo "probe failed for $TAG/$SPLIT (rc=$probe_rc); formal evidence validation skipped" >&2
  exit "$probe_rc"
fi

# Formal batches validate their own split after probe finishes.  leo_driver
# launches id and ood concurrently, so validating only the completed split
# avoids rejecting a valid id batch while the sibling split is still running.
if [[ "$TAG" == SOTA* ]]; then
  VALIDATOR=$F/sc-tools/validate_formal_results.py
  [ -r "$VALIDATOR" ] || { echo "missing formal post-run validator: $VALIDATOR" >&2; exit 2; }
  SCIENCECLAW_REPO=$F/scienceclaw SCIENCECLAW_RUN_ROOT=$R \
    "$F/envs/sc-run/bin/python" "$VALIDATOR" "$TAG" "$R" --split "$SPLIT" || {
      echo "formal post-run evidence failed for $TAG/$SPLIT" >&2
      exit 2
    }
fi
