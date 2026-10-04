#!/bin/bash
# One 4-GPU node: GPUs 0..N-1 serve the LLM (one vLLM per GPU), the remaining GPUs stay free for the GPU tools (srun --overlap from the broker).
# Submit: sbatch --job-name=work2-sc-hold -A <account> -p <partition> --qos=normal -N1 --gres=gpu:4 -c 32 --mem=400G --time=08:00:00 --time-min=03:00:00 hold_and_serve.sbatch
# Log files: pass -o/-e to sbatch (default: slurm-%j.out in the submit directory).
F=${SCIENCECLAW_WORK_ROOT:?set SCIENCECLAW_WORK_ROOT}; L=${SCIENCECLAW_STORE_ROOT:-$F}
mkdir -p $L/sc-serve/logs
NLLM=${NLLM:-2}
for g in $(seq 0 $((NLLM-1))); do
  ( export CUDA_VISIBLE_DEVICES=$g EPID=$SLURM_JOB_ID-g$g PORT=$((21000+g)) MODEL=${MODEL:?set MODEL} NAME=sc-llm TP=1 MAXLEN=65536 MAXSEQS=48
    exec bash $F/sc-serve/serve_vllm.sbatch > $L/sc-serve/logs/$SLURM_JOB_ID-g$g.out 2>&1 ) &
  sleep 25
done
echo "serving $NLLM models on GPUs 0..$((NLLM-1)); tools may use the other GPUs of job $SLURM_JOB_ID on $(hostname)"
wait
