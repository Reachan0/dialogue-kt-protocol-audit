"""Summarize baseline training-protocol deltas across seeds 221 (archived), 222, 223."""
import json, statistics as st
R = 'metrics'
D = 'metrics/seeds'
summary = {}
for corp in ['qatd2k', 'mathdial']:
    deltas, excl, cell_ranges = [], [], []
    per_model = {}
    for m in ['dkt', 'simplekt', 'dktsem']:
        aucs = {k: [] for k in ['C-C', 'C-L', 'L-C', 'L-L']}
        md = []
        for seed in [221, 222, 223]:
            f = f'{R}/{corp}_{m}.json' if seed == 221 else f'{D}/{corp}_{m}_s{seed}.json'
            d = json.load(open(f))
            assert d['common_mask_n'] == {'qatd2k': 1607, 'mathdial': 1751}[corp]
            for k in aucs:
                aucs[k].append(d['cells'][k]['common_mask']['auc'] * 100)
            for name, v in d['paired_deltas_same_eval'].items():
                x = v['delta_auc'] * 100
                lo, hi = (c * 100 for c in v['ci95'])
                deltas.append(x); md.append(x)
                if lo > 0 or hi < 0:
                    excl.append(f'{m} s{seed} {name} {x:.2f} [{lo:.2f},{hi:.2f}]')
        for k, v in aucs.items():
            cell_ranges.append(max(v) - min(v))
        per_model[m] = {'mean_delta': round(st.mean(md), 2), 'min': round(min(md), 2), 'max': round(max(md), 2),
                        'cell_auc_by_seed': {k: [round(x, 2) for x in v] for k, v in aucs.items()}}
    summary[corp] = {
        'n_deltas': len(deltas), 'mean_delta': round(st.mean(deltas), 2),
        'min_delta': round(min(deltas), 2), 'max_delta': round(max(deltas), 2),
        'max_abs_delta': round(max(abs(x) for x in deltas), 2),
        'n_negative': sum(x < 0 for x in deltas),
        'intervals_excluding_zero': excl,
        'max_seed_range_same_cell': round(max(cell_ranges), 2),
        'median_seed_range_same_cell': round(st.median(cell_ranges), 2),
        'per_model': per_model}
json.dump(summary, open('metrics/seeds/seed_summary.json', 'w'), indent=1)
for c, s in summary.items():
    print(c, {k: v for k, v in s.items() if k != 'per_model'})
    for m, v in s['per_model'].items():
        print('  ', m, v['mean_delta'], v['min'], v['max'])
