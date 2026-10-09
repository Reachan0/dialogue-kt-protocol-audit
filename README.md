# Anonymous supplementary material

Paired prefix-based (`prefix`) and full-dialogue (`full`) turn-level correctness
labels for two tutoring corpora, the support/statistics files behind the paper's
tables, and the code used to produce and analyse them.

## Labels

Labels are JSONL, one row per student turn: `dialogue_id`, `split`, `turn`,
`student`, `teacher`, `label` with `label in {true, false, na}`.
No model responses, response identifiers or usage records are included.

- `labels/qatd2k_primary/`  QATD2k, primary annotator, both protocols, train+test.
- `labels/mathdial/`        MathDial, both protocols, train+test.
- `labels/qatd2k_second_annotator/`  QATD2k test split, second annotator, both protocols.

## Support and statistics (`support/`)

`h1h3_*.json` hold the disagreement decomposition and dialogue-clustered
bootstrap intervals for each label pair; `support_test.json` and
`qatd2k_h1h3_*.json` hold the annotation/prediction supports and the
first-scoreable-turn decomposition.

## Code (`code/`)

`annotate.py` (prefix protocol), `annotate_leakage.py` (full protocol),
`annotate_mathdial.py` and `annotate_second_annotator.py` (run drivers with call
budgets), plus the offline analysis: `analyze_protocol_pair.py` (H1-H3),
`support_decomposition.py` (supports), `evaluate_2x2.py` (AUC/bootstrap),
`export_mathdial.py` and `preprocess.py` (dataset construction).

`run_mathdial.sh` / `run_e2.sh` expect gateway credentials in environment
variables; no credentials, endpoints or host paths are included anywhere in this
bundle.

## Configs (`configs/`)

Frozen protocols written before the corresponding runs, and the training scripts
for the 2x2 Knowledge Tracing experiments (`run_*_2x2.sh`), with absolute paths
replaced by `$HOME` / `$BASE_MODEL`.

## Licences

Derived labels inherit the source datasets' terms: QATD2k CC BY-NC 4.0,
MathDial CC BY-SA 4.0.

## Repository layout

Anonymous supplementary material for a submission under review. Prepared so that
no author-identifying information is contained in the files or in the history.
