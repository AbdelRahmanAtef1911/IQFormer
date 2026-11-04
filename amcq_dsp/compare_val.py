"""Compare architectures on the VALIDATION set from the saved val_pred.npz files (no data or test set needed).

  python -m amcq_dsp.compare_val --roots runs_arch runs_dsp
  python -m amcq_dsp.compare_val --roots runs_arch runs_dsp --variants iqf_ms iqf_cum iqf_pls --detail iqf_ms

Runs are folders <variant>_s<seed> containing val_pred.npz. The primary metric is the mean accuracy over
-6, -4, -2, 0 dB; overall accuracy is the guardrail. Different architectures are independently trained, so the
main test is Welch's t-test between the groups of seeds; the seed-matched differences are shown for information.
"""
import argparse
import csv
import os
import re
from collections import defaultdict
import numpy as np
from scipy import stats

CLASSES = ['8PSK', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'AM-DSB', 'AM-SSB', 'WBFM']
BAND = (-6, -4, -2, 0)


def load(roots):
    runs = defaultdict(dict)
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            m = re.fullmatch(r'(.+)_s(\d+)', name)
            f = os.path.join(root, name, 'val_pred.npz')
            if m and os.path.isfile(f):
                with np.load(f) as z:
                    runs[m.group(1)][int(m.group(2))] = (z['pred'], z['y'], z['snr'].astype(int))
    return runs


def scores(pred, y, snr):
    b = np.isin(snr, BAND)
    return 100 * (pred[b] == y[b]).mean(), 100 * (pred == y).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--roots', nargs='+', default=['runs_arch', 'runs_dsp'])
    p.add_argument('--baseline', default='iqformer')
    p.add_argument('--variants', nargs='*', default=None, help='default: every variant found')
    p.add_argument('--detail', default=None, help='variant for per-SNR and per-class detail')
    p.add_argument('--out', default='results_dsp/compare_val.csv')
    a = p.parse_args()
    runs = load(a.roots)
    if a.baseline not in runs:
        raise SystemExit(f'no {a.baseline}_s* runs with val_pred.npz under {a.roots}')
    variants = a.variants or [v for v in runs if v != a.baseline]
    base = {s: scores(*r) for s, r in runs[a.baseline].items()}
    bb = np.array([base[s][0] for s in sorted(base)]); bo = np.array([base[s][1] for s in sorted(base)])
    print(f'{a.baseline:14s} n={len(bb)}  -6..0 dB {bb.mean():.2f} +- {bb.std(ddof=1) if len(bb) > 1 else 0:.2f}   '
          f'overall {bo.mean():.2f} +- {bo.std(ddof=1) if len(bo) > 1 else 0:.2f}   seeds {sorted(base)}')
    rows = []
    for v in variants:
        if v not in runs:
            print(f'{v}: no runs found'); continue
        sc = {s: scores(*r) for s, r in runs[v].items()}
        vb = np.array([sc[s][0] for s in sorted(sc)]); vo = np.array([sc[s][1] for s in sorted(sc)])
        pb = stats.ttest_ind(vb, bb, equal_var=False).pvalue if len(vb) > 1 and len(bb) > 1 else float('nan')
        po = stats.ttest_ind(vo, bo, equal_var=False).pvalue if len(vo) > 1 and len(bo) > 1 else float('nan')
        common = sorted(set(sc) & set(base))
        seedwise = [float(round(sc[s][0] - base[s][0], 2)) for s in common]
        print(f'{v:14s} n={len(vb)}  -6..0 dB {vb.mean():.2f} ({vb.mean() - bb.mean():+.2f}, Welch p={pb:.3f})   '
              f'overall {vo.mean():.2f} ({vo.mean() - bo.mean():+.2f}, p={po:.3f})   seed-matched band diffs {seedwise}')
        rows.append(dict(variant=v, n=len(vb), band=vb.mean(), band_diff=vb.mean() - bb.mean(), band_p=pb,
                         overall=vo.mean(), overall_diff=vo.mean() - bo.mean(), overall_p=po))
    if a.detail and a.detail in runs:
        def per(rs, fn):
            return np.mean([fn(*r) for r in rs.values()], axis=0)
        snrs = list(range(-12, 4, 2))
        acc_snr = lambda pr, y, s: np.array([100 * (pr[s == k] == y[s == k]).mean() for k in snrs])
        acc_cls = lambda pr, y, s: np.array([100 * (pr[np.isin(s, BAND) & (y == c)] == c).mean() for c in range(11)])
        d_snr = per(runs[a.detail], acc_snr) - per(runs[a.baseline], acc_snr)
        d_cls = per(runs[a.detail], acc_cls) - per(runs[a.baseline], acc_cls)
        print(f'\n{a.detail} - {a.baseline}, change per SNR (pp):  ' + '  '.join(f'{k}:{v:+.2f}' for k, v in zip(snrs, d_snr)))
        print(f'{a.detail} - {a.baseline}, change per class in -6..0 dB (pp):  ' +
              '  '.join(f'{c}:{v:+.1f}' for c, v in zip(CLASSES, d_cls)))
    if rows:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        with open(a.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print('\nwrote', a.out, '- choose on these VALIDATION numbers, then run final_test once.')


if __name__ == '__main__':
    main()
