"""Where does test-time augmentation (TTA) gain and lose?  Per-SNR and per-class change, averaged over runs.

Validation (default, may be run as often as you like):
  python -m amcq_dsp.tta_detail --runs runs_arch/iqformer_s1 ... runs_arch/iqformer_s5
Held-out test (run ONCE, after the method is fixed on validation):
  python -m amcq_dsp.tta_detail --runs ... --methods guard --final-test

Methods (all use the same trained model; only the way the views are combined differs):
  mean    average the softmax of the 4 rotations (the original rot4)
  logmean average the log-softmax (geometric mean) of the 4 rotations
  guard   'mean', but if TTA says AM-SSB (the class the model uses for "no signal") while the plain
          prediction says some other class, keep the plain prediction.  Uses only the input window.
  rot2    average the original and the 180-degree rotation (half the cost of rot4)
The same model is scored with and without TTA, so each per-run change is a true paired difference.
"""
import argparse
import csv
import os
import numpy as np
import torch
from scipy import stats
import amcq_dsp  # noqa: F401
from amcq import setup_repo, default_data
from amcq.data import load_rml2016a, CLASSES_2016A
from amcq_arch.ptq import load_run
from amcq_dsp.tta_val import BAND

METHODS = ['mean', 'logmean', 'guard', 'rot2']
SSB = CLASSES_2016A.index('AM-SSB')


def rotations(xb):
    i, q = xb[:, 0], xb[:, 1]
    return [xb, torch.stack([-q, i], 1), -xb, torch.stack([q, -i], 1)]


@torch.no_grad()
def predict_all(model, X, methods, batch=1024):
    out = {m: [] for m in ['none'] + methods}
    for k in range(0, len(X), batch):
        logp = [torch.log_softmax(model(v), 1) for v in rotations(X[k:k + batch])]
        prob = [lp.exp() for lp in logp]
        plain = logp[0].argmax(1)
        mean = sum(prob).argmax(1)
        out['none'].append(plain.cpu())
        if 'mean' in methods:
            out['mean'].append(mean.cpu())
        if 'logmean' in methods:
            out['logmean'].append(sum(logp).argmax(1).cpu())
        if 'guard' in methods:
            g = torch.where((mean == SSB) & (plain != SSB), plain, mean)
            out['guard'].append(g.cpu())
        if 'rot2' in methods:
            out['rot2'].append((prob[0] + prob[2]).argmax(1).cpu())
    return {m: torch.cat(v).numpy() for m, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS)
    p.add_argument('--final-test', action='store_true', help='score the held-out TEST set (run once)')
    p.add_argument('--repo', default=None)
    p.add_argument('--data', default=None)
    p.add_argument('--out', default='results_dsp/tta')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    repo = setup_repo(a.repo)
    d = load_rml2016a(a.data or default_data(repo))
    split = 'te' if a.final_test else 'va'
    name = 'TEST' if a.final_test else 'validation'
    dev = torch.device(a.device)
    X = torch.from_numpy(d['X' + split]).to(dev)
    y, snr = d['y' + split], d['s' + split]
    band = np.isin(snr, BAND)
    groups = ([('overall', np.ones_like(y, bool)), ('-6..0 dB', band),
               ('-20..-12', snr <= -12), ('-10..-8', (snr >= -10) & (snr <= -8)), ('2..18', snr >= 2)]
              + [(f'{s:+d} dB', snr == s) for s in sorted(np.unique(snr))]
              + [(f'{c} (-6..0)', band & (y == i)) for i, c in enumerate(CLASSES_2016A)])
    acc = {m: {g: [] for g, _ in groups} for m in ['none'] + a.methods}
    for run in a.runs:
        preds = predict_all(load_run(run, dev, a), X, a.methods)
        for m, pr in preds.items():
            for g, mask in groups:
                acc[m][g].append(100 * (pr[mask] == y[mask]).mean())
        print(run, '  '.join(f'{m}: {acc[m]["overall"][-1]:.2f}/{acc[m]["-6..0 dB"][-1]:.2f}'
                             for m in ['none'] + a.methods), f'  (overall/-6..0 dB, {name})')
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f'tta_detail_{"test" if a.final_test else "val"}.csv')
    print(f'\n{name}, {len(a.runs)} runs: accuracy without TTA, then the mean paired change of each method '
          f'(* = p < 0.01)')
    print(f'{"group":18s} {"no TTA":>7s}' + ''.join(f' {m:>9s}' for m in a.methods))
    with open(fn, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'no_tta'] + [f'{m}_{k}' for m in a.methods for k in ('change', 'sd', 'p')])
        for g, _ in groups:
            b = np.array(acc['none'][g])
            line, row = f'{g:18s} {b.mean():7.2f}', [g, f'{b.mean():.3f}']
            for m in a.methods:
                dl = np.array(acc[m][g]) - b
                sd = dl.std(ddof=1) if len(dl) > 1 else float('nan')
                pv = stats.ttest_1samp(dl, 0).pvalue if len(dl) > 1 and sd > 0 else float('nan')
                line += f' {dl.mean():+8.2f}{"*" if pv < 0.01 else " "}'
                row += [f'{dl.mean():+.3f}', f'{sd:.3f}', f'{pv:.4g}']
            print(line)
            w.writerow(row)
    print('wrote', fn)


if __name__ == '__main__':
    main()
