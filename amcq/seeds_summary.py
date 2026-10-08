"""Compare full-precision training variants over seeds (Step 9): mean +- SD, and a Welch t-test against the
baseline seeds. Every run folder needs metrics.json from train.py.

  python amcq/seeds_summary.py --base 'runs/iqformer_s[1-5]' --variants 'runs/iqf_*_s[0-9]' --out results/step9_summary.csv

Groups run folders by name without the trailing _s<seed> (runs/iqf_rot_s1..s5 -> iqf_rot).
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np
from scipy import stats

SNRS = list(range(-20, 20, 2))


def load(pattern_list):
    groups = {}
    for pat in pattern_list:
        for r in sorted(glob.glob(pat)):
            f = os.path.join(r, 'metrics.json')
            if not os.path.exists(f):
                continue
            d = json.load(open(f))
            key = re.sub(r'_s\d+$', '', os.path.basename(r.rstrip('/')))
            groups.setdefault(key, []).append(d)
    return groups


def row(name, runs, base):
    ov = np.array([100 * r['overall'] for r in runs])
    band = lambda r, lo, hi: 100 * np.mean([r['per_snr'][str(k)] for k in SNRS if lo <= k <= hi])
    tr = np.array([band(r, -6, 0) for r in runs])
    low = np.array([band(r, -20, -8) for r in runs])
    high = np.array([band(r, 2, 18) for r in runs])
    out = dict(variant=name, n=len(runs), overall=ov.mean(), sd=ov.std(ddof=1) if len(ov) > 1 else 0.0,
               peak=100 * np.mean([r['peak'] for r in runs]), band_m20_m8=low.mean(), band_m6_0=tr.mean(),
               band_2_18=high.mean(), macro_f1=np.mean([r['macro_f1'] for r in runs]),
               epochs=np.mean([r.get('epochs_run', np.nan) for r in runs]))
    if base is not None and name != 'BASE' and len(ov) > 1:
        b = np.array([100 * r['overall'] for r in base])
        t = stats.ttest_ind(ov, b, equal_var=False)
        bt = np.array([band(r, -6, 0) for r in base])
        out.update(diff_pp=ov.mean() - b.mean(), welch_p=t.pvalue, diff_band_m6_0=tr.mean() - bt.mean(),
                   welch_p_band=stats.ttest_ind(tr, bt, equal_var=False).pvalue)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', nargs='+', required=True)
    p.add_argument('--variants', nargs='+', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    base = sum(load(a.base).values(), [])
    var = load(a.variants)
    rows = [row('BASE', base, None)] + [row(k, v, base) for k, v in sorted(var.items())]
    keys = ['variant', 'n', 'overall', 'sd', 'diff_pp', 'welch_p', 'peak', 'band_m20_m8', 'band_m6_0',
            'diff_band_m6_0', 'welch_p_band', 'band_2_18', 'macro_f1', 'epochs']
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    print(f"{'variant':16s} {'n':>2s} {'overall':>8s} {'sd':>5s} {'diff':>6s} {'p':>6s}  {'-6..0 dB':>8s} {'diff':>6s} {'p':>6s}")
    for r in rows:
        print(f"{r['variant']:16s} {r['n']:2d} {r['overall']:8.2f} {r['sd']:5.2f} "
              f"{r.get('diff_pp', float('nan')):+6.2f} {r.get('welch_p', float('nan')):6.3f}  "
              f"{r['band_m6_0']:8.2f} {r.get('diff_band_m6_0', float('nan')):+6.2f} {r.get('welch_p_band', float('nan')):6.3f}")
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
