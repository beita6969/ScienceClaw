#!/bin/bash
S=/home/bedicloud/sharestore2/zxc/scienceclaw
cd $S/code/f43
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=5 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  nice -n 5 $S/env/bin/python infer_llm.py $S/models/ocr/ocronos $S/data/f43/units_ocr.json $S/data/f43/ocronos_c700.jsonl 700 16 > $S/logs/f43_ocronos_c700.log 2>&1
