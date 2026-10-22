"""End-to-end protocol comparison: Train P/Eval P vs Train F/Eval F, each on its own prediction support.

Each bootstrap resample draws dialogue IDs from the union of both cells' dialogues;
each pipeline is scored on its own targets within the resampled dialogues, and the
F-P AUC difference is recorded. Also reports the same diagonal gap on common support,
and point estimates of both gaps for the baselines at seeds 221-223.
Standard library only.  Usage: python code/end_to_end_bootstrap.py [n_boot]
"""
import csv, json, random, sys
from collections import defaultdict

R = 'predictions'


def auc(rows):
    s = sorted(rows, key=lambda r: r[1])
    n, ranks, i = len(s), [0.0] * len(s), 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1][1] == s[i][1]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    pos = sum(r[0] for r in s)
    rank_sum = sum(ranks[k] for k in range(n) if s[k][0] == 1)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * (n - pos))


def load(path):
    g = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        g[r['dialogue_id']][r['turn']] = (int(float(r['label'])), float(r['probability']))
    return g


def flat(g, keys=None):
    return [x for d, turns in g.items() for t, x in turns.items() if keys is None or (d, t) in keys]


n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
out = {}
for corp in ['qatd2k', 'mathdial']:
    for m in ['dkt', 'simplekt', 'dktsem', 'llmkt']:
        P, F = load(f'{R}/{corp}_{m}_C-C.csv'), load(f'{R}/{corp}_{m}_L-L.csv')
        common = {(d, t) for d in P for t in P[d]} & {(d, t) for d in F for t in F[d]}
        dids = sorted(set(P) | set(F))
        rng = random.Random(20261022)
        deltas = []
        for _ in range(n_boot):
            s = [rng.choice(dids) for _ in dids]
            a = [x for d in s for x in P.get(d, {}).values()]
            b = [x for d in s for x in F.get(d, {}).values()]
            deltas.append(auc(b) - auc(a))
        deltas.sort()
        own_p, own_f = auc(flat(P)), auc(flat(F))
        com_p, com_f = auc(flat(P, common)), auc(flat(F, common))
        out[f'{corp}_{m}'] = {
            'own_auc_P': round(own_p * 100, 2), 'own_auc_F': round(own_f * 100, 2),
            'delta_F_minus_P': round((own_f - own_p) * 100, 2),
            'ci95': [round(deltas[int(0.025 * n_boot)] * 100, 2), round(deltas[int(0.975 * n_boot) - 1] * 100, 2)],
            'common_delta_F_minus_P': round((com_f - com_p) * 100, 2),
            'n_P': len(flat(P)), 'n_F': len(flat(F)), 'n_common': len(common), 'dialogues': len(dids),
        }
        print(corp, m, out[f'{corp}_{m}'], flush=True)
seeds = {}
for corp in ['qatd2k', 'mathdial']:
    for m in ['dkt', 'simplekt', 'dktsem']:
        for seed in [221, 222, 223]:
            pre = f'{R}/{corp}_{m}' if seed == 221 else f'{R}/seeds/{corp}_{m}_s{seed}'
            P, F = load(pre + '_C-C.csv'), load(pre + '_L-L.csv')
            common = {(d, t) for d in P for t in P[d]} & {(d, t) for d in F for t in F[d]}
            seeds[f'{corp}_{m}_s{seed}'] = {
                'delta_F_minus_P': round((auc(flat(F)) - auc(flat(P))) * 100, 2),
                'common_delta_F_minus_P': round((auc(flat(F, common)) - auc(flat(P, common))) * 100, 2),
            }
            print(corp, m, seed, seeds[f'{corp}_{m}_s{seed}'], flush=True)
json.dump({'n_boot': n_boot, 'seed': 20261022, 'results': out, 'baseline_seeds': seeds}, open('metrics/end_to_end.json', 'w'), indent=2)
