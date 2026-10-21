#!/usr/bin/env bash
# Optional GPU recipe for the extra baseline training seeds (222, 223); NOT invoked by verify_release.py.
# Same settings as configs/train_2x2.sh for dkt/simplekt/dkt-sem; only --seed and model names differ.
# Usage: bash configs/run_baseline_seeds.sh   (then: python code/summarize_seeds.py)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$ROOT/dialogue-kt/results" "$ROOT/dialogue-kt/saved_models" "$ROOT/metrics/seeds"
cd "$ROOT/dialogue-kt"
COMMON=(--batch_size 64 --emb_size 64)
for seed in 222 223; do
 for corpus in qatd2k mathdial; do
  for mt in dkt simplekt dkt-sem; do
   tag=${mt//-/}
   for train in causal leakage; do
    model=${corpus}_${train}_${tag}_s${seed}
    other=$([ "$train" = causal ] && echo leakage || echo causal)
    PYTHONPATH=. python -m dialogue_kt.main train --dataset "$corpus" --tag_src "$train" --eval_tag_src "$train" --model_type "$mt" --model_name "$model" --seed "$seed" --epochs 100 "${COMMON[@]}"
    PYTHONPATH=. python -m dialogue_kt.main test --dataset "$corpus" --tag_src "$train" --eval_tag_src "$other" --model_type "$mt" --model_name "$model" --seed "$seed" "${COMMON[@]}"
   done
   (cd "$ROOT" && python code/evaluate_2x2.py --bootstrap 2000 --seed 221 --output "metrics/seeds/${corpus}_${tag}_s${seed}.json" \
     --cell "C-C=dialogue-kt/results/predictions_${corpus}_causal_${tag}_s${seed}_eval-causal.csv" \
     --cell "C-L=dialogue-kt/results/predictions_${corpus}_causal_${tag}_s${seed}_eval-leakage.csv" \
     --cell "L-C=dialogue-kt/results/predictions_${corpus}_leakage_${tag}_s${seed}_eval-causal.csv" \
     --cell "L-L=dialogue-kt/results/predictions_${corpus}_leakage_${tag}_s${seed}_eval-leakage.csv")
  done
 done
done
