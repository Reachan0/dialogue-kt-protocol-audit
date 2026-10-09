"""Verify the released numbers from paired labels, effective labels and predictions.

Runs offline, without API credentials or GPUs:
    python verify_release.py
    python verify_release.py --bootstrap-labels

The optional bootstrap mode recomputes all six label summaries (2,000 resamples,
seed 20261007). To regenerate the full AUC/ECE intervals, see README.md; this
quick verifier checks all main metric point estimates and paired AUC deltas.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
from analyze_protocol_pair import load
from evaluate_2x2 import metrics


def read(path):
    return json.loads((ROOT / path).read_text())


def assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)


def label_checks(full_bootstrap):
    specs = [
        ("qatd2k_primary_all", "qatd2k_primary", ["train", "test"]),
        ("qatd2k_primary_test", "qatd2k_primary", ["test"]),
        ("mathdial_all", "mathdial", ["train", "test"]),
        ("mathdial_train", "mathdial", ["train"]),
        ("mathdial_test", "mathdial", ["test"]),
        ("qatd2k_second_test", "qatd2k_second_annotator", ["test"]),
    ]
    for name, subdir, splits in specs:
        arms = {}
        for protocol in ("prefix", "full"):
            pair = {}
            for split in splits:
                raw = load(ROOT / "labels" / subdir / f"{protocol}_{split}.jsonl")
                for (did, turn), label in raw.items():
                    pair[(f"{split}:{did}" if len(splits) > 1 else did, turn)] = label
            arms[protocol] = pair
        P, F = arms["prefix"], arms["full"]
        assert P.keys() == F.keys()
        saved = read(f"support/{name}.json")
        p_only = sum(P[k] != "na" and F[k] == "na" for k in P)
        f_only = sum(P[k] == "na" and F[k] != "na" for k in P)
        flips = sum(P[k] != "na" and F[k] != "na" and P[k] != F[k] for k in P)
        assert (len(P), p_only, f_only, flips) == (saved["targets"], saved["p_only_scoreable"],
                                                  saved["f_only_scoreable"], saved["polarity_flips"])
        if full_bootstrap:
            with tempfile.TemporaryDirectory() as tmp:
                for protocol, pair in arms.items():
                    with open(Path(tmp) / f"{protocol}.jsonl", "w") as f:
                        for (did, turn), label in sorted(pair.items()):
                            f.write(json.dumps({"dialogue_id": did, "turn": turn, "label": label}) + "\n")
                text = subprocess.check_output([
                    sys.executable, str(ROOT / "code/analyze_protocol_pair.py"),
                    "--prefix", str(Path(tmp) / "prefix.jsonl"),
                    "--full", str(Path(tmp) / "full.jsonl"),
                    "--seed", "20261007", "--n-boot", "2000",
                ], text=True)
                calculated = json.loads(text)
                for field in ("H1_ci", "H2_ci"):
                    assert_close(calculated[field], saved[field])
        print(f"PASS labels: {name}, n={len(P)}" + (" including bootstrap intervals" if full_bootstrap else ""))


def support_checks():
    for corpus in ("qatd2k", "mathdial"):
        saved = read(f"support/{corpus}_support_ids.json")
        labels = pd.read_json(ROOT / f"support/{corpus}_effective_test_labels.jsonl", lines=True)
        for protocol in ("prefix", "full"):
            rows = labels[labels.protocol == protocol]
            score = set(); pred = set()
            for did, group in rows.groupby("dialogue_id"):
                group = group.sort_values("turn")
                turns = group.loc[group.effective_label != "na", "turn"].tolist()
                score.update((str(did), int(t)) for t in turns)
                pred.update((str(did), int(t)) for t in turns[1:])
            recorded = lambda kind: {(str(r["dialogue_id"]), r["turn"]) for r in saved[kind][protocol]}
            assert score == recorded("annotation") and pred == recorded("prediction")
        print(f"PASS support IDs: {corpus}")


def prediction_checks():
    manifest = read("configs/main_matrices.json")
    for corpus, models in manifest["cells"].items():
        for model, spec in models.items():
            archive = read(spec["metrics"])
            cells = {k: pd.read_csv(ROOT / path, dtype={"dialogue_id": str})
                     for k, path in spec["predictions"].items()}
            common = None
            for key, df in cells.items():
                assert not df.duplicated(["dialogue_id", "turn"]).any()
                assert df.label.isin([0, 1]).all() and df.probability.between(0, 1).all()
                keys = set(zip(df.dialogue_id, df.turn))
                common = keys if common is None else common & keys
            assert len(common) == archive["common_mask_n"]
            common_df = pd.DataFrame(sorted(common), columns=["dialogue_id", "turn"])
            for key, df in cells.items():
                for scope, rows in [("own_mask", df), ("common_mask", common_df.merge(df, on=["dialogue_id", "turn"]))]:
                    actual = metrics(rows)
                    for field in ["auc", "accuracy", "ece", "n", "dialogues"]:
                        assert_close(actual[field], archive["cells"][key][scope][field])
            for key, value in archive["paired_deltas_same_eval"].items():
                a, b = key.split("_minus_")
                delta = archive["cells"][a]["common_mask"]["auc"] - archive["cells"][b]["common_mask"]["auc"]
                assert_close(delta, value["delta_auc"])
            print(f"PASS predictions: {corpus}/{model}, 4 cells, n={len(common)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-labels", action="store_true")
    args = parser.parse_args()
    label_checks(args.bootstrap_labels)
    support_checks()
    prediction_checks()
    print("All released main point estimates and keyed support checks passed.")


if __name__ == "__main__":
    main()
