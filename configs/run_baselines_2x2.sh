#!/usr/bin/env bash
# MathDial Prefix/Full 2x2 for DKT, simpleKT, DKT-Sem (same settings as the QATD2k runs).
set -euo pipefail
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate causal-dialogkt
export PATH=/usr/lib/wsl/lib:$PATH CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HOME=$PROJECT_ROOT/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $PROJECT_ROOT/edu-paper-0831/dialogue-kt
for mt in dkt simplekt dkt-sem; do
  tag=${mt//-/}
  for train in causal leakage; do
    model=mathdial_${train}_${tag}
    other=$([ "$train" = causal ] && echo leakage || echo causal)
    PYTHONPATH=. python -m dialogue_kt.main train --dataset mathdial --tag_src "$train" --eval_tag_src "$train" --model_type "$mt" --model_name "$model" --epochs 100 --batch_size 64 --emb_size 64
    PYTHONPATH=. python -m dialogue_kt.main test --dataset mathdial --tag_src "$train" --eval_tag_src "$other" --model_type "$mt" --model_name "$model" --batch_size 64 --emb_size 64
  done
  (cd .. && python causal_annotation/evaluate_2x2.py --bootstrap 2000 --output dialogue-kt/results/mathdial_${tag}_metrics_2x2.json \
    --cell C-C=dialogue-kt/results/predictions_mathdial_causal_${tag}_eval-causal.csv \
    --cell C-L=dialogue-kt/results/predictions_mathdial_causal_${tag}_eval-leakage.csv \
    --cell L-C=dialogue-kt/results/predictions_mathdial_leakage_${tag}_eval-causal.csv \
    --cell L-L=dialogue-kt/results/predictions_mathdial_leakage_${tag}_eval-leakage.csv)
done
