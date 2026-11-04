"""Test-time augmentation on the VALIDATION set (no retraining, no test data).

Every view is a valid example of the same class: a rotation by k x 90 degrees is a different carrier phase,
and complex conjugation (Q -> -Q) mirrors the spectrum. The softmax outputs of all views are averaged.

  python -m amcq_dsp.tta_val --runs runs_arch/iqformer_s1 runs_arch/iqformer_s2 ... --views rot4 rot8 --out results_dsp/tta
Prints, per run, overall and -6..0 dB validation accuracy without and with TTA, and the paired change
(same trained model with and without TTA, so the paired t-test is valid here).
"""
import argparse
import csv
import os
import numpy as np
import torch
from scipy import stats
import amcq_dsp  # noqa: F401
from amcq import setup_repo, default_data
from amcq.data import load_rml2016a
from amcq_arch.ptq import load_run

BAND = (-6, -4, -2, 0)


def views(xb, kind):
    i, q = xb[:, 0], xb[:, 1]
    rot = [xb, torch.stack([-q, i], 1), -xb, torch.stack([q, -i], 1)]
    if kind == 'none':
        return [xb]
    if kind == 'rot4':
        return rot
    if kind == 'rot8':
        return rot + [torch.stack([v[:, 0], -v[:, 1]], 1) for v in rot]
    raise ValueError(kind)


@torch.no_grad()
def predict_views(model, X, kind, batch=1024):
    out = []
    for k in range(0, len(X), batch):
        xb = X[k:k + batch]
        out.append(sum(torch.softmax(model(v), 1) for v in views(xb, kind)).argmax(1).cpu())
    return torch.cat(out).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--views', nargs='+', default=['rot4', 'rot8'], choices=['rot4', 'rot8'])
    p.add_argument('--repo', default=None)
    p.add_argument('--data', default=None)
    p.add_argument('--out', default='results_dsp/tta')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    repo = setup_repo(a.repo)
    d = load_rml2016a(a.data or default_data(repo))
    dev = torch.device(a.device)
    X = torch.from_numpy(d['Xva']).to(dev)
    y, snr = d['yva'], d['sva']
    band = np.isin(snr, BAND)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for run in a.runs:
        model = load_run(run, dev, a)
        row = {'run': run}
        for kind in ['none'] + a.views:
            pred = predict_views(model, X, kind)
            row[f'{kind}_overall'] = 100 * (pred == y).mean()
            row[f'{kind}_band'] = 100 * (pred[band] == y[band]).mean()
        rows.append(row)
        print(run, '  '.join(f"{k}: {row[k + '_overall']:.2f} / {row[k + '_band']:.2f}" for k in ['none'] + a.views),
              '  (overall / -6..0 dB, validation)')
    with open(os.path.join(a.out, 'tta_val.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if len(rows) > 1:
        for kind in a.views:
            for m in ('overall', 'band'):
                dlt = np.array([r[f'{kind}_{m}'] - r[f'none_{m}'] for r in rows])
                pval = stats.ttest_1samp(dlt, 0).pvalue
                print(f'{kind} {m:7s}: change {dlt.mean():+.2f} +- {dlt.std(ddof=1):.2f} pp over {len(dlt)} models '
                      f'(paired t-test p = {pval:.3g})')
    print('wrote', os.path.join(a.out, 'tta_val.csv'))


if __name__ == '__main__':
    main()
