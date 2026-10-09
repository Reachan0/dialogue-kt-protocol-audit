"""Offline follow-up: archived calibration and MathDial final-turn rule sensitivity.

No model inference, annotation API, GPU use, or retraining. Run from any directory:
    python causal_annotation/review_followup.py

The without-override comparison is a rule-count sensitivity check, not an AUC
ablation: the existing models were trained and evaluated with the override on.
"""
import ast
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "audit_materials" / "review_followup_20261010"
csv.field_size_limit(sys.maxsize)


def metric_paths():
    paths = {}
    for corpus in ("qatd2k", "mathdial"):
        paths[corpus] = {}
        for model in ("dkt", "simplekt", "dktsem", "llmkt"):
            if corpus == "mathdial":
                suffix = "_s222" if model == "llmkt" else ""
                path = ROOT / "remote_archive/runs_mathdial/results" / f"mathdial_{model}{suffix}_metrics_2x2.json"
            else:
                name = "llmkt_seed222_causal_vs_leakage.json" if model == "llmkt" else f"{model}_metrics_2x2.json"
                path = ROOT / "remote_archive/dialogue-kt/results" / name
            paths[corpus][model] = path
    return paths


def effective_labels(dialogue, annotation, meta, override):
    if "error" in annotation:
        return None
    labels = {}
    for turn in dialogue:
        idx = turn["turn"]
        entry = annotation[f"turn {idx}"]
        labels[idx] = entry["correct"] if entry["kcs"] else None
    last = dialogue[-1]["turn"]
    # Matches apply_annotations: only the ACTUAL last turn, originally non-NA,
    # with KCs, is overridden; there is no backward search for a scoreable turn.
    if override and labels[last] is not None:
        outcome = meta["self_correctness"]
        if outcome == "Yes":
            labels[last] = True
        elif outcome == "No":
            labels[last] = False
        elif outcome == "Yes, but I had to reveal the answer":
            labels[last] = None
    return labels


def support_snapshot(rows, override):
    scoreable, predicted, labels = set(), set(), {}
    skipped = 0
    for row in rows:
        dialogue = ast.literal_eval(row["dialogue"])
        annotation = ast.literal_eval(row["annotation"])
        meta = ast.literal_eval(row["meta_data"])
        effective = effective_labels(dialogue, annotation, meta, override)
        if effective is None:
            skipped += 1
            continue
        did = str(row["dialogue_id"])
        turns = [t["turn"] for t in dialogue if effective[t["turn"]] is not None]
        scoreable.update((did, t) for t in turns)
        predicted.update((did, t) for t in turns[1:])
        labels.update({(did, t): value for t, value in effective.items()})
    return scoreable, predicted, labels, skipped


def decomposition(prefix, full):
    sp, pp, _, _ = prefix
    sf, pf, _, _ = full
    return {
        "annotation_support": {"prefix": len(sp), "full": len(sf), "common": len(sp & sf)},
        "prediction_support": {"prefix": len(pp), "full": len(pf), "common": len(pp & pf)},
        "exclusive_targets": len(pp ^ pf),
        "prefix_only_targets": {"total": len(pp - pf), "not_scoreable_under_full": len((pp - pf) - sf),
                                "jointly_scoreable_rule_excluded": len((pp - pf) & sf)},
        "full_only_targets": {"total": len(pf - pp), "not_scoreable_under_prefix": len((pf - pp) - sp),
                              "jointly_scoreable_rule_excluded": len((pf - pp) & sp)},
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "calibration_definition": "10 equal-width bins over predicted P(correct); no post-hoc calibration",
        "calibration_scope": "agreement with each protocol's evaluation labels on the existing common prediction support; not calibration against human truth",
        "metric_sources": {}, "calibration": {}, "auc_shifts": {},
    }
    for corpus, paths in metric_paths().items():
        result["metric_sources"][corpus] = {m: str(p.relative_to(ROOT)) for m, p in paths.items()}
        result["calibration"][corpus] = {}
        result["auc_shifts"][corpus] = {}
        for model, path in paths.items():
            archived = json.loads(path.read_text())
            cells = {k: v["common_mask"] for k, v in archived["cells"].items()}
            result["calibration"][corpus][model] = {
                k: {"n": v["n"], "ece": v["ece"], "ci95": v["ci95"]["ece"]} for k, v in cells.items()
            }
            auc = lambda k: cells[k]["auc"]
            result["auc_shifts"][corpus][model] = {
                "train_change_fixed_eval_prefix_pp": 100 * (auc("C-C") - auc("L-C")),
                "train_change_fixed_eval_full_pp": 100 * (auc("C-L") - auc("L-L")),
                "eval_change_fixed_train_prefix_pp": 100 * (auc("C-C") - auc("C-L")),
                "eval_change_fixed_train_full_pp": 100 * (auc("L-C") - auc("L-L")),
            }

    sensitivity = {"scope": "MathDial test, existing generated labels and KC annotations; rule-count sensitivity only, no retraining or no-override AUC"}
    paired = {}
    for protocol in ("causal", "leakage"):
        path = ROOT / "dialogue-kt/data/annotated" / f"mathdial_test_{protocol}.csv"
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        paired[protocol] = {"on": support_snapshot(rows, True), "off": support_snapshot(rows, False)}
        on, off = paired[protocol]["on"], paired[protocol]["off"]
        polarity = sum(on[2][k] is not None and off[2][k] is not None and on[2][k] != off[2][k] for k in on[2])
        to_na = sum(on[2][k] is None and off[2][k] is not None for k in on[2])
        sensitivity[protocol] = {"input_dialogues": len(rows), "kc_error_dialogues": on[3],
                                 "polarity_rewrites": polarity, "changed_to_na": to_na,
                                 "prediction_targets_removed_by_override": len(off[1] - on[1])}
    sensitivity["with_override"] = decomposition(paired["causal"]["on"], paired["leakage"]["on"])
    sensitivity["without_override"] = decomposition(paired["causal"]["off"], paired["leakage"]["off"])
    result["mathdial_final_turn_sensitivity"] = sensitivity
    (OUT / "followup_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(sensitivity, indent=2))


if __name__ == "__main__":
    main()
