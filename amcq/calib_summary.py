"""Combine ptq.py results obtained with different calibration sets (--calib-seed) into one table.

  python amcq/calib_summary.py results/step4_whole_calib*      ->  printed + <first dir>/../<name>_calib_summary.csv
For every (spec, model): accuracy for each calibration set, its mean and its range (max - min).
The range is the part of a post-training-quantization result that depends only on which frames calibrated it.
"""
import glob
import os
import re
import sys

import pandas as pd

dirs = sorted(set(sum([glob.glob(a) for a in sys.argv[1:]], [])))
if not dirs:
    raise SystemExit('no result folders given')
frames = []
for d in dirs:
    m = re.search(r'calib(\d+)$', d.rstrip('/'))
    f = pd.read_csv(os.path.join(d, 'ptq_runs.csv'))
    f['calib'] = int(m.group(1)) if m else 0
    frames.append(f)
d = pd.concat(frames)
fp = d[d.spec == 'FP32'].groupby('run').overall.first() * 100
d = d[d.spec != 'FP32']
t = d.pivot_table(index=['spec', 'run'], columns='calib', values='overall') * 100
cols = list(t.columns)
t['mean'] = t[cols].mean(axis=1)
t['range_pp'] = t[cols].max(axis=1) - t[cols].min(axis=1)
t['fp32'] = [fp.get(r, float('nan')) for _, r in t.index]
t['change_pp'] = t['mean'] - t['fp32']
order = list(dict.fromkeys(d.spec))
t = t.reindex(order, level=0)
pd.set_option('display.width', 200)
print(t.round(2).to_string())
s = t.groupby(level=0, sort=False).agg(models=('mean', 'size'), acc_mean=('mean', 'mean'), acc_sd_models=('mean', 'std'),
                                       change_pp=('change_pp', 'mean'), worst_range_pp=('range_pp', 'max'))
print('\nPer precision (mean over models of the mean over calibration sets):')
print(s.round(2).to_string())
base = re.sub(r'_calib\d+$', '', dirs[0].rstrip('/'))
t.round(4).to_csv(base + '_calib_detail.csv')
s.round(4).to_csv(base + '_calib_summary.csv')
print(f'\nwrote {base}_calib_detail.csv and {base}_calib_summary.csv')
