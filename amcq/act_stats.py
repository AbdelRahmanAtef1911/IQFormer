"""Step 6 — why do 4-bit activations fail? Error analysis on the trained models.

Part 1, ranges (calibration set 0, per model):
  activation_ranges_<run>.csv  per quantized layer input: max |a|, 99.9th / 99th percentile of |a|, and
                               'usable 4-bit levels' = 16 * p99 / range (levels the bulk of the values uses)
  channel_ranges_<run>.csv     per conv/linear input: the range of each channel compared with the range of
                               the whole tensor (one shared scale): 'imbalance' = widest / median channel range,
                               'levels_median_channel_4bit' = 4-bit levels a typical channel gets
  fusion_branches.csv          the fusion layer's input = [IQ branch | STFT branch] concatenated: range of each
                               branch and the 4-bit levels each gets under the shared scale

Part 2, accuracy tests (every model x every calibration set, same frames as ptq.py --calib-seed):
  whole model, W4A4 and W32A4 : min-max vs percentile clipping (99.99, 99.9, 99)  -> are outliers the cause?
  FUSION only, W32A4          : min-max (= Step 5), clipping, one scale per branch, one scale per channel
  whole model, W32A4 / W4A4   : with one scale per fusion branch (deployable fix) / per channel everywhere
                                (upper bound)
  -> step6_tests.csv (every run) and step6_summary.csv (mean over models and calibration sets)

  python amcq/act_stats.py --runs runs/iqformer_s1 runs/iqformer_s2 runs/iqformer_s3 --calib-seeds 0 1 2 --out results/step6
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amcq import setup_repo, default_data                     # noqa: E402

FUSION_IN = 'fusion.Conv.0'          # first 1x1 convolution of the fusion layer (input = IQ | STFT channels)

TESTS = [  # (test name, spec, calibration method, percentile, fix)
    ('whole: min-max', 'all=W4A4', 'minmax', 100.0, None),
    ('whole: clip 99.99', 'all=W4A4', 'percentile', 99.99, None),
    ('whole: clip 99.9', 'all=W4A4', 'percentile', 99.9, None),
    ('whole: clip 99', 'all=W4A4', 'percentile', 99.0, None),
    ('whole: min-max', 'all=W32A4', 'minmax', 100.0, None),
    ('whole: clip 99.99', 'all=W32A4', 'percentile', 99.99, None),
    ('whole: clip 99.9', 'all=W32A4', 'percentile', 99.9, None),
    ('whole: clip 99', 'all=W32A4', 'percentile', 99.0, None),
    ('fusion only: min-max (= Step 5)', 'FUSION=W32A4', 'minmax', 100.0, None),
    ('fusion only: clip 99.9', 'FUSION=W32A4', 'percentile', 99.9, None),
    ('fusion only: clip 99', 'FUSION=W32A4', 'percentile', 99.0, None),
    ('fusion only: one scale per branch', 'FUSION=W32A4', 'minmax', 100.0, 'branch'),
    ('fusion only: one scale per channel', 'FUSION=W32A4', 'minmax', 100.0, 'fusion_channel'),
    ('whole: one scale per fusion branch', 'all=W32A4', 'minmax', 100.0, 'branch'),
    ('whole: one scale per fusion branch', 'all=W4A4', 'minmax', 100.0, 'branch'),
    ('whole: one scale per channel (upper bound)', 'all=W32A4', 'minmax', 100.0, 'all_channel'),
]


def branch_split(model):
    """Channel groups of the fusion input: IQ stem channels first, then STFT stem channels (torch.cat order)."""
    m = model.m
    n_stft = m.patch_embedSTFT[0].out_channels
    n_iq = m.fusion.Conv[0].in_channels - n_stft
    return [list(range(n_iq)), list(range(n_iq, n_iq + n_stft))]


def range_tables(model, x_cal, tag):
    from amcq.quant import quantize_model, ActQuant, QConv, QLinear, QLSTM
    q = quantize_model(model, 'all=W32A8')
    for m in q.modules():
        if isinstance(m, ActQuant) and m.bits is not None:
            m.mode = 'observe'
    chan, hooks = {}, []
    for name, m in q.named_modules():
        if isinstance(m, (QConv, QLinear)):
            def h(mod, inp, name=name):
                x = inp[0].detach().float()
                dim = 1 if isinstance(mod, QConv) else -1
                xm = x.movedim(dim, -1).reshape(-1, x.shape[dim])
                lo, hi = xm.min(0).values, xm.max(0).values
                if name in chan:
                    chan[name] = (torch.minimum(chan[name][0], lo), torch.maximum(chan[name][1], hi), mod.group)
                else:
                    chan[name] = (lo, hi, mod.group)
            hooks.append(m.register_forward_pre_hook(h))
    with torch.no_grad():
        for i in range(0, len(x_cal), 256):
            q(x_cal[i:i + 256])
    for hk in hooks:
        hk.remove()

    rows = []
    for name, m in q.named_modules():
        aqs = []
        if isinstance(m, (QConv, QLinear)):
            aqs = [('in', m.aq, m.group)]
        elif isinstance(m, QLSTM):
            aqs = [(f'in_l{k}', a, 'LSTM') for k, a in enumerate(m.in_q)] + [(f'h_{k}', a, 'LSTM') for k, a in enumerate(m.h_q)]
        for t, a, g in aqs:
            s = a.stats()
            rng_ = max(max(s['max'], 0) - min(s['min'], 0), 1e-12)
            rows.append(dict(layer=f'{name}.{t}', group=g, min=s['min'], max=s['max'], absmax=s['absmax'],
                             p999=s['p999'], p99=s['p99'], max_over_p999=s['absmax'] / max(s['p999'], 1e-12),
                             usable_levels_4bit=16 * s['p99'] / rng_))
    crow = []
    for name, (lo, hi, g) in chan.items():
        lo0, hi0 = lo.clamp(max=0), hi.clamp(min=0)
        r = (hi0 - lo0).cpu().numpy()
        tensor_range = float(hi0.max() - lo0.min())
        med = float(np.median(r))
        crow.append(dict(layer=name, group=g, channels=len(r), tensor_range=tensor_range, median_channel_range=med,
                         widest_channel_range=float(r.max()), narrowest_channel_range=float(r.min()),
                         imbalance=float(r.max()) / max(med, 1e-12),
                         levels_median_channel_4bit=16 * med / max(tensor_range, 1e-12)))
    fus = None
    for name, (lo, hi, g) in chan.items():
        if name.endswith(FUSION_IN):
            fus = []
            iq, st = branch_split(model)
            t_lo, t_hi = float(lo.min().clamp(max=0)), float(hi.max().clamp(min=0))
            for bname, idx in (('IQ stem', iq), ('STFT stem', st)):
                b_lo, b_hi = float(lo[idx].min()), float(hi[idx].max())
                b_rng = max(b_hi, 0) - min(b_lo, 0)
                fus.append(dict(run=tag, branch=bname, channels=len(idx), min=b_lo, max=b_hi, range=b_rng,
                                shared_range=t_hi - t_lo, levels_4bit_shared_scale=16 * b_rng / max(t_hi - t_lo, 1e-12),
                                levels_8bit_shared_scale=256 * b_rng / max(t_hi - t_lo, 1e-12)))
    return rows, crow, fus


def write_csv(path, rows):
    if rows:
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--repo', default=None)
    p.add_argument('--data', default=None)
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--calib-seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--calib', type=int, default=1024)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args(argv)
    repo = setup_repo(args.repo)
    from amcq.data import load_rml2016a, to_device
    from amcq.metrics import predict
    from amcq.quant import quantize_model, calibrate, use_channel_scales
    from ptq import load_run

    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    d = load_rml2016a(args.data or default_data(repo))
    t = to_device(d, dev)

    def cal_set(seed):                                    # exactly the frames ptq.py --calib-seed uses
        rng = np.random.default_rng(seed)
        return t['Xtr'][torch.from_numpy(rng.choice(len(d['ytr']), args.calib, replace=False)).to(dev)]

    res, fus_all = [], []
    for run in args.runs:
        tag = os.path.basename(run.rstrip('/'))
        model = load_run(run, dev, args)
        rows, crow, fus = range_tables(model, cal_set(args.calib_seeds[0]), tag)
        write_csv(os.path.join(args.out, f'activation_ranges_{tag}.csv'), rows)
        write_csv(os.path.join(args.out, f'channel_ranges_{tag}.csv'), crow)
        fus_all += fus or []
        by_group = {}
        for r in rows:
            by_group.setdefault(r['group'], []).append(r['usable_levels_4bit'])
        print(f'\n{tag}: median usable 4-bit levels per group:',
              {g: round(float(np.median(v)), 2) for g, v in by_group.items()})
        for f in fus or []:
            print(f"  fusion input, {f['branch']:9s}: range {f['min']:+.3f} .. {f['max']:+.3f}  "
                  f"-> {f['levels_4bit_shared_scale']:.1f} of 16 levels with one shared scale")
        fp = (predict(model, t['Xte']).argmax(1).numpy() == d['yte']).mean()
        for c in args.calib_seeds:
            x_cal = cal_set(c)
            for name, spec, method, pct, fix in TESTS:
                q = quantize_model(model, spec)
                if fix == 'branch':
                    use_channel_scales(q, names=[FUSION_IN], groups=branch_split(model))
                elif fix == 'fusion_channel':
                    use_channel_scales(q, names=[FUSION_IN])
                elif fix == 'all_channel':
                    use_channel_scales(q)
                calibrate(q, x_cal, method, pct)
                acc = (predict(q, t['Xte']).argmax(1).numpy() == d['yte']).mean()
                res.append(dict(run=tag, calib=c, test=name, spec=spec, method=method, pct=pct, fix=fix or '',
                                acc=100 * acc, fp32=100 * fp, change_pp=100 * (acc - fp)))
                print(f'  calib {c}  {spec:13s} {name:44s} {100 * acc:6.2f}%  ({100 * (acc - fp):+7.2f} pp)')
                del q
                if dev.type == 'cuda':
                    torch.cuda.empty_cache()
    write_csv(os.path.join(args.out, 'fusion_branches.csv'), fus_all)
    write_csv(os.path.join(args.out, 'step6_tests.csv'), res)

    summ = []
    for name, spec, method, pct, fix in TESTS:
        rr = [r for r in res if r['test'] == name and r['spec'] == spec]
        ch = np.array([r['change_pp'] for r in rr])
        per_model = [np.mean([r['change_pp'] for r in rr if r['run'] == m]) for m in dict.fromkeys(r['run'] for r in rr)]
        summ.append(dict(spec=spec, test=name, n=len(rr), acc_mean=np.mean([r['acc'] for r in rr]),
                         change_pp=ch.mean(), sd_models=np.std(per_model, ddof=1) if len(per_model) > 1 else 0.0,
                         min_pp=ch.min(), max_pp=ch.max()))
    write_csv(os.path.join(args.out, 'step6_summary.csv'), summ)
    print('\nMean over models and calibration sets (change vs FP32, pp):')
    for s in summ:
        print(f"  {s['spec']:13s} {s['test']:44s} {s['acc_mean']:6.2f}%  {s['change_pp']:+7.2f}  "
              f"(sd models {s['sd_models']:.2f}, runs {s['min_pp']:+.2f} .. {s['max_pp']:+.2f})")

    # cross-check: the 'fusion only: min-max' rows must equal Step 5's FUSION=W32A4 numbers exactly
    diffs = []
    for c in args.calib_seeds:
        f5 = os.path.join('results', f'step5_fine_calib{c}', 'ptq_runs.csv')
        if os.path.exists(f5):
            for r5 in csv.DictReader(open(f5)):
                if r5['spec'] == 'FUSION=W32A4':
                    tag = os.path.basename(r5['run'].rstrip('/'))
                    for r in res:
                        if r['run'] == tag and r['calib'] == c and r['test'].startswith('fusion only: min-max'):
                            diffs.append(abs(r['acc'] - 100 * float(r5['overall'])))
    if diffs:
        print(f'\nCross-check with Step 5 (FUSION=W32A4, {len(diffs)} runs): largest difference '
              f'{max(diffs):.4f} pp -> {"PASS" if max(diffs) < 1e-6 else "CHECK"}')
    print(f'\nwrote {args.out}/: step6_summary.csv, step6_tests.csv, fusion_branches.csv, '
          f'activation_ranges_*.csv, channel_ranges_*.csv')


if __name__ == '__main__':
    main()
