#!/bin/bash
S=/home/bedicloud/sharestore2/zxc/scienceclaw
cd $S/code/f43
run() { g=$1; c=$2; bs=$3
  PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$g HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  nice -n 5 $S/env/bin/python infer_llm.py $S/models/ocr/ocronos $S/data/f43/units_ocr.json $S/data/f43/ocronos_c$c.jsonl $c $bs > $S/logs/f43_ocronos_c$c.log 2>&1 &
}
run 5 300 24
run 2 1200 8
wait
