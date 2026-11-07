"""D1 - ensemble ceiling on the VALIDATION set (no training).

Averages the softmax outputs of several trained runs, with and without the guarded 4-rotation TTA, and compares
with the single models. If the ensemble beats a single model by >= 1 pp, distillation from it has room.

  python -m amcq_next.d1_ensemble --runs runs_arch/iqformer_s1 runs_arch/iqformer_s2 runs_arch/iqformer_s3 runs_arch/iqformer_s4 runs_arch/iqformer_s5
"""
import argparse
import csv
import os
import numpy as np
import torch
import amcq_next  # noqa: F401
from amcq import setup_repo, default_data
from amcq.data import load_rml2016a, CLASSES_2016A
from amcq_arch.ptq import load_run

BAND = (-6, -4, -2, 0)
SSB = CLASSES_2016A.index('AM-SSB')


def rotations(xb):
    i, q = xb[:, 0], xb[:, 1]
    return [xb, torch.stack([-q, i], 1), -xb, torch.stack([q, -i], 1)]


@torch.no_grad()
def probs(model, X, batch=1024):
    """Returns (plain softmax, mean softmax over the 4 rotations), both (N, C) numpy."""
    model.eval()
    p0, p4 = [], []
    for k in range(0, len(X), batch):
        ps = [torch.softmax(model(v), 1) for v in rotations(X[k:k + batch])]
        p0.append(ps[0].cpu()); p4.append((sum(ps) / 4).cpu())
    return torch.cat(p0).numpy(), torch.cat(p4).numpy()


def guard(p0, p4):
    plain, tta = p0.argmax(1), p4.argmax(1)
    return np.where((tta == SSB) & (plain != SSB), plain, tta)


def scores(pred, y, snr):
    acc = lambda m: 100 * (pred[m] == y[m]).mean()          # noqa: E731
    return dict(overall=acc(np.ones_like(y, bool)), band=acc(np.isin(snr, BAND)), low=acc(snr <= -8),
                high=acc(snr >= 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--repo', default=None)
    p.add_argument('--data', default=None)
    p.add_argument('--out', default='results_next/d1_ensemble.csv')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    repo = setup_repo(a.repo)
    d = load_rml2016a(a.data or default_data(repo))
    dev = torch.device(a.device)
    X = torch.from_numpy(d['Xva']).to(dev)
    y, snr = d['yva'], d['sva']
    P0, P4, rows = [], [], []
    for run in a.runs:
        p0, p4 = probs(load_run(run, dev, a), X)
        P0.append(p0); P4.append(p4)
        rows.append((os.path.basename(run.rstrip('/')), scores(p0.argmax(1), y, snr), scores(guard(p0, p4), y, snr)))
    e0, e4 = np.mean(P0, 0), np.mean(P4, 0)
    ens = (f'ensemble of {len(a.runs)}', scores(e0.argmax(1), y, snr), scores(guard(e0, e4), y, snr))
    single = {k: np.mean([r[1][k] for r in rows]) for k in rows[0][1]}
    single_tta = {k: np.mean([r[2][k] for r in rows]) for k in rows[0][1]}
    print('VALIDATION accuracy (%)          overall   -6..0 dB  <=-8 dB   >=+2 dB')
    for name, s0, s4 in rows + [ens]:
        print(f'{name:28s} plain  {s0["overall"]:7.2f}  {s0["band"]:8.2f}  {s0["low"]:7.2f}  {s0["high"]:7.2f}')
        print(f'{"":28s} +TTA   {s4["overall"]:7.2f}  {s4["band"]:8.2f}  {s4["low"]:7.2f}  {s4["high"]:7.2f}')
    print(f'{"mean single model":28s} plain  {single["overall"]:7.2f}  {single["band"]:8.2f}  {single["low"]:7.2f}  {single["high"]:7.2f}')
    gain = ens[1]['overall'] - single['overall']
    print(f'\nensemble - mean single model: overall {gain:+.2f} pp, -6..0 dB {ens[1]["band"] - single["band"]:+.2f} pp'
          f'   (with TTA on both: overall {ens[2]["overall"] - single_tta["overall"]:+.2f} pp)')
    print('DECISION: ' + ('distillation from the ensemble has room (>= 1 pp)' if gain >= 1 else
                          'ensemble gain < 1 pp: distillation is unlikely to add much'))
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'tta', 'overall', 'band', 'low_le_m8', 'high_ge_2'])
        for name, s0, s4 in rows + [ens]:
            w.writerow([name, 'no', *[f'{s0[k]:.3f}' for k in ('overall', 'band', 'low', 'high')]])
            w.writerow([name, 'guard', *[f'{s4[k]:.3f}' for k in ('overall', 'band', 'low', 'high')]])
    print('wrote', a.out)


if __name__ == '__main__':
    main()
