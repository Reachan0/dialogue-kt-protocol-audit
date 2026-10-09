#!/usr/bin/env bash
# MathDial Prefix/Full 2x2 for LLMKT, seed 222 (same settings as the QATD2k main LLMKT runs).
set -euo pipefail
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate causal-dialogkt
export PATH=/usr/lib/wsl/lib:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$PROJECT_ROOT/hf-cache CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=$BASE_MODEL
cd $PROJECT_ROOT/edu-paper-0831/dialogue-kt
for train in causal leakage; do
  MODEL=mathdial_${train}_s222_llmkt
  other=$([ "$train" = causal ] && echo leakage || echo causal)
  PYTHONPATH=. python -m dialogue_kt.main train --dataset mathdial --tag_src "$train" --eval_tag_src "$train" --seed 222 --model_type lmkt --model_name "$MODEL" --base_model "$BASE" --epochs 5 --batch_size 1 --grad_accum_steps 64 --r 16 --lora_alpha 16 --pack_kcs true --quantize false
  PYTHONPATH=. python -m dialogue_kt.main test --dataset mathdial --tag_src "$train" --eval_tag_src "$other" --seed 222 --model_type lmkt --model_name "$MODEL" --base_model "$BASE" --batch_size 1 --pack_kcs true --quantize false
done
cd ..
python causal_annotation/evaluate_2x2.py --bootstrap 2000 --output dialogue-kt/results/mathdial_llmkt_s222_metrics_2x2.json \
  --cell C-C=dialogue-kt/results/predictions_mathdial_causal_s222_llmkt_eval-causal.csv \
  --cell C-L=dialogue-kt/results/predictions_mathdial_causal_s222_llmkt_eval-leakage.csv \
  --cell L-C=dialogue-kt/results/predictions_mathdial_leakage_s222_llmkt_eval-causal.csv \
  --cell L-L=dialogue-kt/results/predictions_mathdial_leakage_s222_llmkt_eval-leakage.csv
