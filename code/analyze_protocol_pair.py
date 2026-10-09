"""Offline H1-H3 analysis for a Prefix/Full label pair (no API calls).

H1: share of disagreements that are NA-boundary changes.
H2: Prefix-only-scoreable minus Full-only-scoreable, with dialogue-clustered bootstrap CI.
H3: correctness agreement where both labels are non-NA.
Optional --ref-prefix/--ref-full give same-protocol cross-annotator agreement.

Labels are JSONL rows with dialogue_id, turn, label (rows with turn null are skipped).
Only targets labeled under both protocols are compared.

Usage:
  python analyze_protocol_pair.py --prefix P.jsonl --full F.jsonl [--ref-prefix ... --ref-full ...]
"""
import argparse
import json
from collections import defaultdict

import numpy as np


def load(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("turn") is not None:
            out[(str(r["dialogue_id"]), int(r["turn"]))] = r["label"]
    return out


def category(p, f):
    if p == f:
        return "agree"
    if p == "na" or f == "na":
        return "p_only" if f == "na" else "f_only"
    return "flip"


def bootstrap(by_dialogue, stat, n_boot, seed):
    ids = list(by_dialogue)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sample = rng.choice(len(ids), size=len(ids), replace=True)
        vals.append(stat([c for i in sample for c in by_dialogue[ids[i]]]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--full", required=True)
    ap.add_argument("--ref-prefix")
    ap.add_argument("--ref-full")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--output")
    args = ap.parse_args()

    P, F = load(args.prefix), load(args.full)
    keys = sorted(P.keys() & F.keys())
    by_dialogue = defaultdict(list)
    for k in keys:
        by_dialogue[k[0]].append(category(P[k], F[k]) + ("|both" if P[k] != "na" and F[k] != "na" else ""))

    cats = [c for cs in by_dialogue.values() for c in cs]
    count = lambda name: sum(c.split("|")[0] == name for c in cats)
    p_only, f_only, flip = count("p_only"), count("f_only"), count("flip")
    disagree = p_only + f_only + flip
    both = [c for c in cats if c.endswith("|both")]

    def h2(cs):
        return (sum(c.startswith("p_only") for c in cs) - sum(c.startswith("f_only") for c in cs)) / len(cs)

    def h1(cs):
        d = sum(not c.startswith("agree") for c in cs)
        return sum(c.startswith(("p_only", "f_only")) for c in cs) / d if d else float("nan")

    result = {
        "targets": len(keys), "dialogues": len(by_dialogue),
        "missing_prefix": len(F.keys() - P.keys()), "missing_full": len(P.keys() - F.keys()),
        "overall_agreement": 1 - disagree / len(keys),
        "disagreements": disagree, "p_only_scoreable": p_only, "f_only_scoreable": f_only, "polarity_flips": flip,
        "H1_na_boundary_share": (p_only + f_only) / disagree if disagree else None,
        "H1_ci": bootstrap(by_dialogue, h1, args.n_boot, args.seed),
        "H2_net_p_minus_f_per_target": (p_only - f_only) / len(keys),
        "H2_ci": bootstrap(by_dialogue, h2, args.n_boot, args.seed),
        "H3_both_scoreable": len(both),
        "H3_polarity_agreement": 1 - flip / len(both) if both else None,
        "nonNA_rate_prefix": sum(P[k] != "na" for k in keys) / len(keys),
        "nonNA_rate_full": sum(F[k] != "na" for k in keys) / len(keys),
    }
    for name, ref, own in (("prefix", args.ref_prefix, P), ("full", args.ref_full, F)):
        if ref:
            R = load(ref)
            shared = own.keys() & R.keys()
            result[f"cross_annotator_{name}_agreement"] = sum(own[k] == R[k] for k in shared) / len(shared)
            result[f"cross_annotator_{name}_n"] = len(shared)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        open(args.output, "w").write(text + "\n")


if __name__ == "__main__":
    main()
