"""Evaluate 2x2 protocol predictions with own and common masks.

Each prediction CSV must contain:
  dialogue_id, turn, label, probability, train_protocol, eval_protocol

Example:
  python evaluate_2x2.py \
    --cell C-C=results/predictions_causal_eval-causal.csv \
    --cell C-L=results/predictions_causal_eval-leakage.csv \
    --cell L-C=results/predictions_leakage_eval-causal.csv \
    --cell L-L=results/predictions_leakage_eval-leakage.csv

The common mask is the intersection of (dialogue_id, turn) prediction keys
across every supplied cell. Bootstrap resamples dialogue IDs, never individual
turns, and paired deltas reuse the same resample for both cells.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


def ece(labels, probs, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ids = np.minimum(np.digitize(probs, bins[1:-1]), n_bins - 1)
    result = 0.0
    for b in range(n_bins):
        mask = ids == b
        if mask.any():
            result += mask.mean() * abs(labels[mask].mean() - probs[mask].mean())
    return result


def metrics(df):
    y = df["label"].to_numpy(dtype=float)
    p = df["probability"].to_numpy(dtype=float)
    return {
        "auc": roc_auc_score(y, p),
        "accuracy": accuracy_score(y, p >= 0.5),
        "ece": ece(y, p),
        "n": len(df),
        "dialogues": df["dialogue_id"].nunique(),
    }


def bootstrap_indices(df, rng, n_boot):
    grouped = {did: group.index.to_numpy() for did, group in df.groupby("dialogue_id")}
    dialogue_ids = np.array(list(grouped), dtype=object)
    for _ in range(n_boot):
        sampled = rng.choice(dialogue_ids, size=len(dialogue_ids), replace=True)
        yield np.concatenate([grouped[did] for did in sampled])


def bootstrap_metrics(df, seed, n_boot):
    base = df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    values = {"auc": [], "accuracy": [], "ece": []}
    for idxs in bootstrap_indices(base, rng, n_boot):
        sample = base.iloc[idxs]
        if sample["label"].nunique() < 2:
            continue
        cur = metrics(sample)
        for name in values:
            values[name].append(cur[name])
    return {name: [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]
            for name, vals in values.items()}


def paired_delta(df_a, df_b, seed, n_boot):
    key = ["dialogue_id", "turn"]
    a = df_a[key + ["label", "probability"]].rename(columns={"label": "label_a", "probability": "prob_a"})
    b = df_b[key + ["label", "probability"]].rename(columns={"label": "label_b", "probability": "prob_b"})
    merged = a.merge(b, on=key, validate="one_to_one")
    # For a paired model delta, both cells must be scored against the same label.
    if not (merged["label_a"] == merged["label_b"]).all():
        raise ValueError("Paired delta requires identical ground-truth labels on the common mask")
    base = merged.rename(columns={"label_a": "label"}).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    deltas = []
    for idxs in bootstrap_indices(base, rng, n_boot):
        sample = base.iloc[idxs]
        y = sample["label"].to_numpy(float)
        if len(np.unique(y)) < 2:
            continue
        deltas.append(roc_auc_score(y, sample["prob_a"]) - roc_auc_score(y, sample["prob_b"]))
    point = roc_auc_score(base["label"], base["prob_a"]) - roc_auc_score(base["label"], base["prob_b"])
    return {"delta_auc": float(point), "ci95": [float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", required=True, help="NAME=path.csv; repeat for all cells")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=221)
    ap.add_argument("--output", default="results/metrics_2x2.json")
    args = ap.parse_args()

    cells = {}
    for spec in args.cell:
        name, path = spec.split("=", 1)
        df = pd.read_csv(path, dtype={"dialogue_id": str})
        required = {"dialogue_id", "turn", "label", "probability"}
        if not required <= set(df.columns):
            raise ValueError(f"{name} missing columns: {required - set(df.columns)}")
        if df.duplicated(["dialogue_id", "turn"]).any():
            raise ValueError(f"{name} has duplicate prediction keys")
        cells[name] = df

    common = None
    for df in cells.values():
        keys = set(zip(df["dialogue_id"], df["turn"]))
        common = keys if common is None else common & keys
    common_df = pd.DataFrame(sorted(common), columns=["dialogue_id", "turn"])

    result = {"common_mask_n": len(common_df), "cells": {}, "paired_deltas_same_eval": {}}
    for name, df in cells.items():
        own = metrics(df)
        own["ci95"] = bootstrap_metrics(df, args.seed, args.bootstrap)
        fixed = common_df.merge(df, on=["dialogue_id", "turn"], validate="one_to_one")
        common_metrics = metrics(fixed)
        common_metrics["ci95"] = bootstrap_metrics(fixed, args.seed, args.bootstrap)
        result["cells"][name] = {"own_mask": own, "common_mask": common_metrics}

    # Compare training protocols under each fixed evaluation protocol (e.g. C-C vs L-C).
    names = list(cells)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            eval_a = a.rsplit("-", 1)[-1]
            eval_b = b.rsplit("-", 1)[-1]
            if eval_a != eval_b:
                continue
            da = common_df.merge(cells[a], on=["dialogue_id", "turn"])
            db = common_df.merge(cells[b], on=["dialogue_id", "turn"])
            result["paired_deltas_same_eval"][f"{a}_minus_{b}"] = paired_delta(da, db, args.seed, args.bootstrap)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
