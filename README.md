# Dialogue KT Protocol Audit — anonymous supplementary material

Paired Prefix/Full annotation labels, explicit evaluation supports, completed-run
predictions and offline verification for two tutoring corpora. No credentials,
service addresses, model weights or raw API responses are distributed.

## Quick offline verification (no GPU or API)

From this repository root, in an isolated environment:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-analysis.txt
python verify_release.py
python verify_release.py --bootstrap-labels
```

The first command checks six raw-label disagreement summaries, both corpora's
annotation/prediction IDs, and every main matrix's AUC, accuracy, ECE and paired
AUC point differences from the included predictions. The optional second mode
also recomputes all six label-statistic confidence intervals (2,000
dialogue-clustered percentile resamples, seed 20261007).

A successful verification record is in `configs/verification.txt`. This does not
claim independent human validation of the labels or a fresh training replication.

To recompute a complete main matrix's AUC/accuracy/ECE intervals and paired AUC
intervals, for example:

```bash
python code/evaluate_2x2.py --seed 221 --bootstrap 2000 \
  --cell C-C=predictions/mathdial_llmkt_C-C.csv \
  --cell C-L=predictions/mathdial_llmkt_C-L.csv \
  --cell L-C=predictions/mathdial_llmkt_L-C.csv \
  --cell L-L=predictions/mathdial_llmkt_L-L.csv \
  --output /tmp/mathdial_llmkt_verified.json
```

The eight matrices and exact cell/file mappings are specified in
`configs/main_matrices.json`. C/L correspond to Prefix/Full; the first letter is
the **training** protocol, the second the **evaluation** protocol.

## Files and scope

- `labels/qatd2k_primary/`: both protocols, train/test, primary annotator.
- `labels/mathdial/`: both protocols, train/test.
- `labels/qatd2k_second_annotator/`: both protocols on the QATD2k test split.
- `labels/qatd2k_third_prefix_test.jsonl`: the earlier partial third-model check
  (1,016 turns); no Full labels for this partial run are claimed.
- `support/{corpus}_effective_test_labels.jsonl`: test labels **after** KC
  filtering and, for MathDial, the teacher's actual-last-turn outcome override.
  Raw generated labels in `labels/` are a separate layer.
- `support/{corpus}_support_ids.json`: exact `(dialogue_id, turn)` lists for
  annotation support, prediction support, their intersections, exclusive targets
  and jointly-scoreable targets excluded by the first-scoreable-turn rule.
- `support/{dataset}_{annotator}_{split}.json` and `support/mathdial_*.json`:
  six distinct label-level summaries, with bootstrap settings. For combined
  train/test summaries, dialogue IDs are namespaced as `split:dialogue_id`.
- `support/review_followup.json`: existing ECE results and an offline MathDial
  final-turn rule sensitivity count. The override-off comparison **does not**
  include new AUCs or newly trained models.
- `metrics/`: eight main matrix archives, plus archived common-training,
  history-label, seed-221 and rationale results for supplementary experiments.
- `predictions/`: 32 main-cell CSVs, one per corpus/model/train/eval combination.
  Only key, label, probability and protocol columns are included.
- `code/`: annotation drivers and offline analyses. `review_followup.py` exports the archived ECE summaries and the MathDial final-turn rule-count sensitivity. The main analyses are
  runnable with the processed inputs supplied here. The original raw-data
  preprocessor is also supplied for reference; its source-specific metadata must
  be obtained from the public dataset providers if it is rerun.
- `configs/`: written full-run protocols, main-run file map and portable training
  recipes. Plans followed small pilots; they are not formal preregistrations.
- `dialogue-kt/`: patched upstream training package (upstream MIT licence
  preserved), dependency specification, KC dictionaries, validation IDs and
  compact processed CSVs used by the main experiments.
- `data/`: processed QATD2k inputs for the annotation drivers.
- `FILE_HASHES.json`: SHA256 digests of the released files.

Raw-label files use `dialogue_id`, `turn`, `label` (`true`, `false`, `na`), and
where applicable split and dialogue-text fields. Service response IDs, usage and
raw reasoning are omitted. Effective-label files and prediction files explicitly
identify their postprocessing layer.

## Optional annotation regeneration (paid hosted API)

These commands are **not** part of the offline verification. Hosted model
identifiers are those reported by a third-party service; availability, actual
upstream snapshots, and equality to the original runs are not guaranteed.

```bash
pip install -r requirements-annotation.txt
export GATEWAY_BASE_URL=...  # your own authorised endpoint
export GATEWAY_API_KEY=...   # your own credential; never commit it
bash code/run_mathdial.sh --protocol prefix --split test \
  --run-dir runs/new_mathdial --max-calls 4000 --concurrency 8
bash code/run_e2.sh --protocol full --run-dir runs/new_second_annotator \
  --max-calls 600 --concurrency 8
```

The drivers read the processed inputs supplied here. Do not use released
stripped-label files as annotation-driver resume checkpoints: raw assistant
history is intentionally absent. Start a fresh run directory. MathDial's exporter
uses the original driver directory layout; `python prepare_label_layout.py`
creates that layout from the flat released files for offline export, followed by
`python code/export_mathdial.py --run-dir runs/mathdial`.

## Optional main KT retraining (GPU)

The upstream package is from <https://github.com/umass-ml4ed/dialogue-kt>, adapted
for the paired protocols; its MIT licence is included. The paper's archived
outputs are provided independently of this expensive operation. Install the
training dependencies in a separate environment from the light analysis one:

```bash
pip install -r dialogue-kt/requirements.txt
bash configs/train_2x2.sh mathdial dkt
BASE_MODEL=/your/licensed/Llama-3.1-8B-Instruct \
  bash configs/train_2x2.sh mathdial lmkt
```

Either corpus (`qatd2k`, `mathdial`) and any main model (`dkt`, `simplekt`,
`dkt-sem`, `lmkt`) may be selected. LLMKT uses seed 222, five epochs, LoRA r=16,
alpha=16, batch 1 and accumulation 64; baselines use seed 221 and 100 epochs.
The best validation-loss checkpoint is used. Embedding weights for DKT-Sem and
base Llama weights are not included; obtain them under their own terms. Hardware,
dependency and numerical differences may affect fresh training.

The main recipes do not rerun the supplementary seed-221, history or rationale
experiments. Their archived result summaries are included, but their complete
request histories, checkpoints and interruption ledgers are not; those checks
are not claimed to be fully regenerable from this release.

## Licences and data provenance

QATD2k-derived labels, text and processed inputs: **CC BY-NC 4.0** (research,
non-commercial reuse with attribution):
<https://huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k>.

MathDial-derived labels, text and processed inputs: **CC BY-SA 4.0**, retaining
attribution and share-alike conditions:
<https://github.com/eth-nlped/mathdial>.

The included upstream dialogue-kt source is covered by its own MIT licence in
`dialogue-kt/LICENSE`. Source-provider attribution is not author identification
for this submission. No author-identifying service addresses or host paths are
included in this release.
