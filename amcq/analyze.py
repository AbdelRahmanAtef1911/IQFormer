"""Aggregate runs into report-ready tables and figures.

  python amcq/analyze.py --runs 'runs/*' --out results/analysis
Groups run folders by name without the trailing _s<seed> (e.g. runs/iqformer_s1..s3 -> 'iqformer'),
reports mean +- std over seeds, and draws:
  fig_acc_vs_snr.pdf        accuracy vs SNR, one line per model/variant (mean over seeds)
  fig_perclass_<run>.pdf    per-class accuracy vs SNR for one run (cf. IQFormer paper Fig. 6)
  fig_confusion_<run>.pdf   confusion matrices at -4, +2 and +18 dB (cf. paper Fig. 7)
  fig_ptq_<name>.pdf        accuracy vs precision from ptq.py summaries (if --ptq given)
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', default=['runs/*'])
    p.add_argument('--ptq', nargs='*', default=[], help='ptq output folders')
    p.add_argument('--detail', nargs='*', default=None, help='runs to draw per-class / confusion figures for')
    p.add_argument('--out', required=True)
    args = p.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from amcq.data import CLASSES_2016A
    from amcq.metrics import confusion_at

    os.makedirs(args.out, exist_ok=True)
    runs = sorted(set(sum([glob.glob(r) for r in args.runs], [])))
    runs = [r for r in runs if os.path.exists(os.path.join(r, 'metrics.json')) and 'args' in json.load(open(os.path.join(r, 'metrics.json')))]
    groups = {}
    for r in runs:
        tag = re.sub(r'_s\d+$', '', os.path.basename(r.rstrip('/')))
        groups.setdefault(tag, []).append(r)

    lines = ['| Model / variant | n | Params | Overall (%) | Peak (%) | -20..0 dB (%) | 0..18 dB (%) | Macro F1 |',
             '|---|---|---|---|---|---|---|---|']
    curves = {}
    fmt = lambda v: f'{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}' if len(v) > 1 else f'{np.mean(v):.2f}'
    for tag, rs in sorted(groups.items()):
        ms = [json.load(open(os.path.join(r, 'metrics.json'))) for r in rs]
        get = lambda k: [100 * m[k] for m in ms if m.get(k) is not None]
        f1 = [m['macro_f1'] for m in ms]
        lines.append(f"| {tag} | {len(ms)} | {ms[0].get('params', 0):,} | {fmt(get('overall'))} | {fmt(get('peak'))} | "
                     f"{fmt(get('mean_snr_le0'))} | {fmt(get('mean_snr_ge0'))} | "
                     f"{np.mean(f1):.4f} |")
        snrs = sorted(int(s) for s in ms[0]['per_snr'])
        curves[tag] = (snrs, np.mean([[m['per_snr'][str(s)] for s in snrs] for m in ms], 0))
    open(os.path.join(args.out, 'summary_models.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    if curves:
        plt.figure(figsize=(6, 4))
        for tag, (s, a) in curves.items():
            plt.plot(s, 100 * a, marker='o', ms=3, label=tag)
        plt.axhline(100 / 11, ls=':', c='gray', lw=1, label='chance (9.1%)')
        plt.xlabel('SNR (dB)'); plt.ylabel('Accuracy (%)'); plt.grid(alpha=.3); plt.legend(fontsize=7)
        plt.tight_layout(); plt.savefig(os.path.join(args.out, 'fig_acc_vs_snr.pdf')); plt.close()

    detail = args.detail if args.detail is not None else [rs[0] for rs in groups.values()][:1]
    for r in detail:
        z = np.load(os.path.join(r, 'test_pred.npz'))
        pred, y, snr = z['pred'], z['y'], z['snr']
        name = os.path.basename(r.rstrip('/'))
        snrs = sorted(np.unique(snr))
        plt.figure(figsize=(6, 4))
        for k, c in enumerate(CLASSES_2016A):
            acc = [100 * (pred[(snr == s) & (y == k)] == k).mean() for s in snrs]
            plt.plot(snrs, acc, marker='.', label=c)
        plt.xlabel('SNR (dB)'); plt.ylabel('Accuracy (%)'); plt.grid(alpha=.3); plt.legend(fontsize=6, ncol=2)
        plt.tight_layout(); plt.savefig(os.path.join(args.out, f'fig_perclass_{name}.pdf')); plt.close()
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
        for a, s in zip(ax, (-4, 2, 18)):
            cm = confusion_at(pred, y, snr, s)
            a.imshow(cm, cmap='Blues', vmin=0, vmax=1)
            a.set_xticks(range(11)); a.set_xticklabels(CLASSES_2016A, rotation=90, fontsize=6)
            a.set_yticks(range(11)); a.set_yticklabels(CLASSES_2016A, fontsize=6)
            for i in range(11):
                for j in range(11):
                    if cm[i, j] >= 0.05:
                        a.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center', fontsize=5,
                               color='white' if cm[i, j] > .5 else 'black')
            a.set_title(f'{s} dB'); a.set_xlabel('predicted'); a.set_ylabel('true')
        plt.tight_layout(); plt.savefig(os.path.join(args.out, f'fig_confusion_{name}.pdf')); plt.close()

    for pdir in args.ptq:
        import csv
        rows = list(csv.DictReader(open(os.path.join(pdir, 'ptq_summary.csv'))))
        name = os.path.basename(pdir.rstrip('/'))
        plt.figure(figsize=(7, 3.6))
        x = np.arange(len(rows))
        plt.bar(x, [float(r['delta_mean_pp']) for r in rows], yerr=[float(r['delta_std_pp']) for r in rows], capsize=2)
        plt.xticks(x, [r['spec'] for r in rows], rotation=60, ha='right', fontsize=6)
        plt.ylabel('Accuracy change vs FP32 (pp)'); plt.axhline(0, c='k', lw=.6); plt.grid(axis='y', alpha=.3)
        plt.tight_layout(); plt.savefig(os.path.join(args.out, f'fig_ptq_{name}.pdf')); plt.close()
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
