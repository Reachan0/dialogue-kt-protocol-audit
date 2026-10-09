"""Convert MathDial Prefix/Full annotation runs to dialogue-kt annotated CSVs.

Rows, dialogue text, metadata and KCs come unchanged from LLMKT's
mathdial_{split}_atc.csv; only per-turn correctness is replaced by the protocol's
label. Dialogues missing any label under EITHER protocol are dropped from BOTH
outputs, so the two protocols share the same rows and validation split.

Usage:
  python export_mathdial.py --run-dir audit_materials/mathdial_formal_20261007
"""
import argparse
import ast
import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dialogue-kt" / "data" / "annotated"
PROTOCOLS = {"causal": "prefix", "leakage": "full"}  # dialogue-kt tag_src -> run name
csv.field_size_limit(sys.maxsize)


def load_labels(path: Path) -> dict:
    labels = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("turn") is not None:
            labels[(r["dialogue_id"], r["turn"])] = r["label"]
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    args = ap.parse_args()

    for split in ("train", "test"):
        src = pd.read_csv(OUT / f"mathdial_{split}_atc.csv")
        labels = {p: load_labels(args.run_dir / f"{run}_{split}" / "labels.jsonl") for p, run in PROTOCOLS.items()}
        keep, dropped = [], []
        for _, row in src.iterrows():
            did = str(row["index"])
            turns = [t["turn"] for t in ast.literal_eval(row["dialogue"])]
            ok = all((did, t) in labels[p] for p in PROTOCOLS for t in turns)
            (keep if ok else dropped).append(did)
        for protocol in PROTOCOLS:
            rows = []
            for _, row in src.iterrows():
                did = str(row["index"])
                if did not in keep:
                    continue
                anno = ast.literal_eval(row["annotation"])
                # LLMKT's KC step failed on some dialogues ({"error": ...}); dialogue-kt skips
                # those rows in KT, so keep them unchanged under both protocols.
                for key, turn_anno in ({} if "error" in anno else anno).items():
                    label = labels[protocol][(did, int(key.split()[1]))]
                    turn_anno["correct"] = True if label == "true" else False if label == "false" else None
                out = row.to_dict()
                out["dialogue_id"] = did
                out["annotation"] = repr(anno)
                rows.append(out)
            pd.DataFrame(rows).to_csv(OUT / f"mathdial_{split}_{protocol}.csv", index=False)
        n_error = sum("error" in ast.literal_eval(a) for a in src["annotation"])
        print(f"{split}: kept {len(keep)} / {len(src)} dialogues; dropped {len(dropped)}: {dropped[:20]}; "
              f"LLMKT KC-error rows (skipped by dialogue-kt): {n_error}")


if __name__ == "__main__":
    main()
