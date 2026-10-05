#!/bin/bash
# Stage the audited OpenAI CLIP ViT-B/32 component for the remote tool worker.
# This intentionally installs Python packages without dependencies: sc-harness
# already owns the compatible Torch/CUDA stack.
set -euo pipefail
F=${SCIENCECLAW_WORK_ROOT:?set SCIENCECLAW_WORK_ROOT}
L=${SCIENCECLAW_STORE_ROOT:-$F}
PY=$F/envs/sc-harness/bin/python
D=$L/sc-tools/clipdeps
M=$L/models/clip/open_clip_vit_b32
URL=https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
SHA=40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af

mkdir -p "$D" "$M"
"$PY" -m pip install --disable-pip-version-check --no-input --no-deps --target "$D" \
  open_clip_torch==2.31.0 timm==1.0.30 ftfy==6.3.1 wcwidth==0.2.13 regex==2024.11.6
if [ ! -f "$M/ViT-B-32.pt" ]; then
  curl -L --fail --retry 3 --max-time 900 -o "$M/ViT-B-32.pt" "$URL"
fi
echo "$SHA  $M/ViT-B-32.pt" | sha256sum -c -
PYTHONPATH="$F/scienceclaw:$D:$F/sv_pkgs" SCIENCECLAW_MODELS="$L/models" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES= \
  "$PY" - <<'PY'
import numpy as np
from scilib import clip_retrieval
p = clip_retrieval.provenance()
assert p["available"] and p["weights_sha256"] == "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
z = clip_retrieval.encode_images(np.zeros((1, 64, 64, 3), dtype=np.uint8), device="cpu")
assert z.shape == (1, 512)
print("frozen CLIP stage OK", p["weights_sha256"], z.shape)
PY
