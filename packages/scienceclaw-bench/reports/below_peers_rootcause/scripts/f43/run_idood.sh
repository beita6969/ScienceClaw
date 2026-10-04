#!/bin/bash
S=/home/bedicloud/sharestore2/zxc/scienceclaw
cd $S/code/f43
run() { g=$1; sp=$2
  PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$g HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  nice -n 5 $S/env/bin/python infer_llm.py $S/models/ocr/ocronos $S/data/f43/units_ocr_$sp.json $S/data/f43/ocronos_c700_$sp.jsonl 700 16 > $S/logs/f43_ocronos_$sp.log 2>&1 &
}
run 5 id
run 2 ood
wait
