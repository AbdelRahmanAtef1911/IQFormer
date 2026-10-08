"""Summarize quantization-aware training runs (Step 10): accuracy before (PTQ) and after QAT, the paired change
against each model's own full-precision version, TOST equivalence within +-0.5 pp, and the change per SNR band.

  python amcq/qat_summary.py 'runs/qat*' --out results/step10_summary.csv

Groups run folders by name without the trailing _s<seed> (runs/qatkd_w4a4_iqformer_s1..s3 -> qatkd_w4a4_iqformer).
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SNRS = list(range(-20, 20, 2))
NOISE = 0.61            # 2 x seed-to-seed SD of the three original FP32 IQFormer models


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    from amcq.metrics import tost
    groups = {}
    for pat in a.runs:
        for r in sorted(glob.glob(pat)):
            f = os.path.join(r, 'metrics.json')
            if os.path.exists(f):
                d = json.load(open(f))
                if 'paired_vs_fp32' in d:
                    groups.setdefault(re.sub(r'_s\d+$', '', os.path.basename(r.rstrip('/'))), []).append(d)
    rows = []
    for k, runs in sorted(groups.items()):
        dl = np.array([r['paired_vs_fp32']['delta_pp'] for r in runs])
        per = np.array([[r['paired_vs_fp32']['delta_pp_per_snr'][str(s)] for s in SNRS] for r in runs])
        band = lambda lo, hi: per[:, [i for i, s in enumerate(SNRS) if lo <= s <= hi]].mean()
        t = tost(dl)
        rows.append(dict(config=k, n=len(runs), spec=runs[0]['args']['spec'], kd=bool(runs[0]['args'].get('teacher')),
                         fp32=100 * np.mean([r.get('fp32_overall', np.nan) for r in runs]),
                         ptq=100 * np.mean([r['ptq_overall'] for r in runs]), qat=100 * np.mean([r['overall'] for r in runs]),
                         delta_pp=dl.mean(), delta_sd=dl.std(ddof=1) if len(dl) > 1 else 0.0,
                         within_noise=bool(abs(dl.mean()) < NOISE), p_tost=t.get('p_tost'),
                         band_m6_0=band(-6, 0), band_2_18=band(2, 18),
                         wrong=np.mean([r['paired_vs_fp32']['worse'] for r in runs]),
                         right=np.mean([r['paired_vs_fp32']['better'] for r in runs]),
                         minutes=np.mean([r.get('minutes', np.nan) for r in runs])))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{'config':28s} {'n':>2s} {'FP32':>6s} {'PTQ':>6s} {'QAT':>6s} {'change':>7s} {'sd':>5s} {'TOST p':>7s} {'-6..0':>6s} {'2..18':>6s} {'min':>5s}")
    for r in rows:
        print(f"{r['config']:28s} {r['n']:2d} {r['fp32']:6.2f} {r['ptq']:6.2f} {r['qat']:6.2f} {r['delta_pp']:+7.2f} "
              f"{r['delta_sd']:5.2f} {r['p_tost'] if r['p_tost'] is not None else float('nan'):7.3f} "
              f"{r['band_m6_0']:+6.2f} {r['band_2_18']:+6.2f} {r['minutes']:5.0f}")
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
