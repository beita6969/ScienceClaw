#!/bin/bash
# Build the ScienceClaw tool/harness environment on the cluster (login node, network allowed). Log: $L/sc-tools/logs/env.log
set -u
F=${SCIENCECLAW_WORK_ROOT:?set SCIENCECLAW_WORK_ROOT}; L=${SCIENCECLAW_STORE_ROOT:-$F}
mkdir -p $L/sc-tools/logs $F/sc-tools; cd $F/sc-tools
E=$F/envs/sc-harness
[ -d $E ] || /usr/bin/python3.11 -m venv $E
. $E/bin/activate
pip install -q -U pip wheel setuptools 2>&1 | tail -n 2
echo "== torch"; pip install -q torch==2.9.1 torchaudio==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -n 3
echo "== harness"; pip install -q -r req_harness_u.txt 2>&1 | tail -n 5
echo "== tools"; pip install -q -r req_tools_u.txt 2>&1 | tail -n 8
echo "== check"; python - <<'PY'
import importlib
for m in ("torch","transformers","chronos","stanza","demucs","chgnet","ase","pymatgen","phonopy","sklearn","scipy","rdkit","lightgbm","jiwer","conllu","z3","h5py","skimage","xarray","safetensors","accelerate","nibabel"):
    try:
        mod = importlib.import_module(m); print("ok  ", m, getattr(mod, "__version__", ""))
    except Exception as e:
        print("FAIL", m, str(e)[:100])
import torch; print("cuda", torch.version.cuda)
PY
echo ENV_DONE
