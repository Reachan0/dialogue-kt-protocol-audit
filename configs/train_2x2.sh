#!/usr/bin/env bash
# Optional GPU training recipe; NOT invoked by verify_release.py.
# Usage: BASE_MODEL=/path/to/Llama-3.1-8B-Instruct bash configs/train_2x2.sh mathdial lmkt
#        bash configs/train_2x2.sh qatd2k dkt
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATASET=${1:?provide qatd2k or mathdial}
MODEL_TYPE=${2:?provide dkt, simplekt, dkt-sem or lmkt}
[[ "$DATASET" == qatd2k || "$DATASET" == mathdial ]] || exit 2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$ROOT/dialogue-kt/results" "$ROOT/dialogue-kt/saved_models"
cd "$ROOT/dialogue-kt"
case "$MODEL_TYPE" in
 lmkt)
  : "${BASE_MODEL:?Set BASE_MODEL to a licensed Llama-3.1-8B-Instruct checkpoint}"
  TAG=s222_llmkt
  COMMON=(--seed 222 --model_type lmkt --base_model "$BASE_MODEL" --batch_size 1 --pack_kcs true --quantize false)
  TRAIN=(--epochs 5 --grad_accum_steps 64 --r 16 --lora_alpha 16)
  ;;
 dkt|simplekt|dkt-sem)
  TAG=${MODEL_TYPE//-/}
  COMMON=(--seed 221 --model_type "$MODEL_TYPE" --batch_size 64 --emb_size 64)
  TRAIN=(--epochs 100)
  ;;
 *) echo 'Unsupported model type' >&2; exit 2 ;;
esac
for protocol in causal leakage; do
 model=${DATASET}_${protocol}_${TAG}
 other=$([ "$protocol" = causal ] && echo leakage || echo causal)
 PYTHONPATH=. python -m dialogue_kt.main train --dataset "$DATASET" --tag_src "$protocol" --eval_tag_src "$protocol" --model_name "$model" "${COMMON[@]}" "${TRAIN[@]}"
 PYTHONPATH=. python -m dialogue_kt.main test --dataset "$DATASET" --tag_src "$protocol" --eval_tag_src "$other" --model_name "$model" "${COMMON[@]}"
done
cd "$ROOT"
python code/evaluate_2x2.py --bootstrap 2000 --seed 221 --output "dialogue-kt/results/${DATASET}_${TAG}_metrics_2x2.json" \
 --cell "C-C=dialogue-kt/results/predictions_${DATASET}_causal_${TAG}_eval-causal.csv" \
 --cell "C-L=dialogue-kt/results/predictions_${DATASET}_causal_${TAG}_eval-leakage.csv" \
 --cell "L-C=dialogue-kt/results/predictions_${DATASET}_leakage_${TAG}_eval-causal.csv" \
 --cell "L-L=dialogue-kt/results/predictions_${DATASET}_leakage_${TAG}_eval-leakage.csv"
