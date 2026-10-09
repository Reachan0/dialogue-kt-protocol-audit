"""Rule-based prediction-support decomposition for a Prefix/Full test pair (offline).

Uses dialogue-kt's own apply_annotations (NA/KC handling, MathDial final-turn override)
and the first-scoreable-turn exclusion used for next-interaction evaluation.

Usage (from dialogue-kt/):
  python ../causal_annotation/support_decomposition.py --dataset qatd2k
"""
import argparse
import json
import sys
from ast import literal_eval
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
from dialogue_kt.kt_data_loading import apply_annotations  # noqa: E402


def supports(path):
    df = pd.read_csv(path, converters={c: literal_eval for c in ["dialogue", "meta_data", "annotation"]})
    if "dialogue_id" not in df:
        df["dialogue_id"] = df["index"]
    scoreable, predicted = set(), set()
    for _, row in df.iterrows():
        dialogue = apply_annotations(row)
        if not dialogue:
            continue
        turns = [t["turn"] for t in dialogue if t["correct"] is not None]
        did = str(row["dialogue_id"])
        scoreable |= {(did, t) for t in turns}
        predicted |= {(did, t) for t in turns[1:]}
    return scoreable, predicted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["qatd2k", "mathdial"])
    ap.add_argument("--output")
    args = ap.parse_args()
    sp, pp = supports(f"data/annotated/{args.dataset}_test_causal.csv")
    sf, pf = supports(f"data/annotated/{args.dataset}_test_leakage.csv")
    p_only, f_only = pp - pf, pf - pp
    result = {
        "annotation_support": {"prefix": len(sp), "full": len(sf), "common": len(sp & sf)},
        "prediction_support": {"prefix": len(pp), "full": len(pf), "common": len(pp & pf)},
        "dialogues_with_targets": {"prefix": len({d for d, _ in pp}), "full": len({d for d, _ in pf}),
                                   "both": len({d for d, _ in pp} & {d for d, _ in pf}),
                                   "either": len({d for d, _ in pp} | {d for d, _ in pf})},
        "prefix_only_targets": {"total": len(p_only), "not_scoreable_under_full": len(p_only - sf),
                                "jointly_scoreable_rule_excluded": len(p_only & sf)},
        "full_only_targets": {"total": len(f_only), "not_scoreable_under_prefix": len(f_only - sp),
                              "jointly_scoreable_rule_excluded": len(f_only & sp)},
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
