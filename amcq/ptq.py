"""Post-training quantization (PTQ) sweeps on trained checkpoints.

Every quantized model is compared with ITS OWN full-precision version on the same 44,000 test frames
(paired design), so the run-to-run variation of training cancels.

Presets
  whole   : whole model at W8A8, W6A6, W4A8, W4A4, W3A8, W2A2 and weight-only W8/W4  (RQ1, H1, control)
  blocks  : one block family at a time (CONV, ATTN, LSTM, HEAD) at 8/6/4 bits: W+A, W only, A only (RQ2, H2)
  fine    : the CONV family split into STEM, FUSION, CONVENC, LOCAL, FFN at W4A4 and A4 only
  mixed   : mixed-precision candidates for the proposed configuration (RQ3)
  a8path  : W4 weights; activations raised to 8 bits group by group, most sensitive per MAC first (step 5b, RQ3)
  custom  : --specs "all=W8A8" "CONV=W4A8,LSTM=W8A8" ...

Examples
  python amcq/ptq.py --runs runs/iqformer_s1 runs/iqformer_s2 runs/iqformer_s3 --preset whole --out results/ptq_whole
  python amcq/ptq.py --runs runs/iqformer_s* --preset blocks --out results/ptq_blocks
  python amcq/ptq.py --runs runs/iqformer_s* --preset whole --calib-method percentile --pct 99.9 --out results/ptq_whole_p999
  # cross-check against the earlier Brevitas numbers (released weight.pt files are accepted directly):
  python amcq/ptq.py --runs save_models/*/weight.pt --preset custom --specs all=W8A8 all=W4A4 CONV=W4A4 \
         --act-scheme symmetric --no-fold --calib-method percentile --pct 99.999 --out results/ptq_brevitas_check
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amcq import setup_repo, default_data                     # noqa: E402

PRESETS = {
    'whole': ['all=W8A8', 'all=W6A6', 'all=W4A8', 'all=W4A6', 'all=W4A4', 'all=W3A8', 'all=W2A2',
              'all=W8A32', 'all=W4A32'],
    'blocks': [f'{g}={q}' for g in ('CONV', 'ATTN', 'LSTM', 'HEAD')
               for q in ('W8A8', 'W6A6', 'W4A4', 'W8A32', 'W6A32', 'W4A32', 'W3A32', 'W32A8', 'W32A6', 'W32A4')],
    'fine': [f'{g}={q}' for g in ('STEM', 'FUSION', 'CONVENC', 'LOCAL', 'FFN', 'ATTN', 'LSTM', 'HEAD')
             for q in ('W4A4', 'W32A4')],
    'mixed': ['all=W8A8', 'all=W4A8', 'all=W4A8,LSTM=W8A8', 'all=W4A8,HEAD=W8A8',
              'CONV=W4A8,ATTN=W4A6,LSTM=W8A8,HEAD=W8A8', 'CONV=W4A6,ATTN=W4A6,LSTM=W8A8,HEAD=W8A8',
              'all=W6A6', 'all=W4A6'],
    # Step 5b: 4-bit weights everywhere; activations raised to 8 bits one group at a time, in the order of
    # Step 5's loss per share of compute (FUSION, STEM+HEAD, ATTN, FFN, CONVENC, LSTM; LOCAL last),
    # plus the literature rule alone (first and last layer at 8 bits, 0.1 % of MACs) to compare with FUSION alone (0.4 %)
    'a8path': ['all=W4A4',
               'all=W4A4,STEM=W4A8,HEAD=W4A8',
               'all=W4A4,FUSION=W4A8',
               'all=W4A4,FUSION=W4A8,STEM=W4A8,HEAD=W4A8',
               'all=W4A4,FUSION=W4A8,STEM=W4A8,HEAD=W4A8,ATTN=W4A8',
               'all=W4A4,FUSION=W4A8,STEM=W4A8,HEAD=W4A8,ATTN=W4A8,FFN=W4A8',
               'all=W4A4,FUSION=W4A8,STEM=W4A8,HEAD=W4A8,ATTN=W4A8,FFN=W4A8,CONVENC=W4A8',
               'all=W4A4,FUSION=W4A8,STEM=W4A8,HEAD=W4A8,ATTN=W4A8,FFN=W4A8,CONVENC=W4A8,LSTM=W4A8',
               'all=W4A8'],
}


def model_bits(qmodel, base_model):
    """Weight storage in bytes: quantized tensors at their bit width, everything else at 32 bits."""
    from amcq.quant import QConv, QLinear, QLSTM
    bits, counted = 0, set()
    for m in qmodel.modules():
        if isinstance(m, (QConv, QLinear)):
            w = m.conv.weight if isinstance(m, QConv) else m.lin.weight
            bits += w.numel() * (m.wq.bits or 32); counted.add(id(w))
        elif isinstance(m, QLSTM):
            for name, p in m.lstm.named_parameters():
                if name.startswith('weight'):
                    bits += p.numel() * (m.wih_q[0].bits or 32); counted.add(id(p))
    for p in qmodel.parameters():
        if id(p) not in counted:
            bits += p.numel() * 32
    return bits / 8


def load_run(run, dev, args):
    from amcq.models import build_model
    if os.path.isdir(run):
        meta = json.load(open(os.path.join(run, 'metrics.json')))['args']
        model = build_model(meta['model'], stft=meta.get('stft', 'real'), iq=meta.get('iq', 'iq'),
                            act=meta.get('act', 'gelu'), input_norm=meta.get('input_norm', False)).to(dev)
        model.load_state_dict(torch.load(os.path.join(run, 'best.pt'), map_location=dev))
    else:                                                          # a released weight.pt
        model = build_model('iqformer').to(dev)
        model.load_legacy(run, dev)
    return model.eval()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--repo', default=None)
    p.add_argument('--data', default=None)
    p.add_argument('--runs', nargs='+', required=True, help='run folders from train.py or released weight.pt files')
    p.add_argument('--preset', default='whole', choices=list(PRESETS) + ['custom'])
    p.add_argument('--specs', nargs='*', default=[])
    p.add_argument('--calib', type=int, default=1024, help='calibration frames, from the TRAINING set')
    p.add_argument('--calib-seed', type=int, default=0, help='which 1,024 training frames calibrate (0 = the frames used so far)')
    p.add_argument('--calib-method', default='minmax', choices=['minmax', 'percentile', 'auto'],
                   help="auto = percentile (--pct) for activations of 4 bits or fewer, min-max for wider ones")
    p.add_argument('--pct', type=float, default=99.99)
    p.add_argument('--no-fold', action='store_true')
    p.add_argument('--no-bare', action='store_true', help='leave layer_scale and w_g in floating point (as Brevitas did)')
    p.add_argument('--act-scheme', default='affine', choices=['affine', 'symmetric'],
                   help="'symmetric' + --no-fold + --calib-method percentile --pct 99.999 ~ the earlier Brevitas setup")
    p.add_argument('--save-pred', action='store_true', help='also save every test prediction (per-class / confusion analysis)')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args(argv)
    repo = setup_repo(args.repo)
    from amcq.data import load_rml2016a, to_device
    from amcq.metrics import predict, summarize, paired, tost, save_json
    from amcq.quant import quantize_model, calibrate

    runs = sorted(set(sum([glob.glob(r) or [r] for r in args.runs], [])))
    specs = args.specs if args.preset == 'custom' else PRESETS[args.preset] + args.specs
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    d = load_rml2016a(args.data or default_data(repo))
    t = to_device(d, dev)
    rng = np.random.default_rng(args.calib_seed)
    cal_idx = torch.from_numpy(rng.choice(len(d['ytr']), args.calib, replace=False)).to(dev)
    x_cal = t['Xtr'][cal_idx]

    rows, per_spec = [], {}
    for run in runs:
        model = load_run(run, dev, args)
        fp = predict(model, t['Xte']).argmax(1).numpy()
        fp_sum = summarize(fp, d['yte'], d['ste'])
        fp_bytes = sum(q.numel() for q in model.parameters()) * 4
        rows.append(dict(run=run, spec='FP32', overall=fp_sum['overall'], macro_f1=fp_sum['macro_f1'], delta_pp=0.0,
                         changed=0, worse=0, better=0, mcnemar_p=1.0, weight_MB=fp_bytes / 1e6,
                         per_snr=json.dumps(fp_sum['per_snr'])))
        print(f'\n{run}: FP32 {100 * fp_sum["overall"]:.2f}%')
        for spec in specs:
            q = quantize_model(model, spec, fold=not args.no_fold, act_scheme=args.act_scheme,
                               quantize_bare=not args.no_bare)
            calibrate(q, x_cal, args.calib_method, args.pct)
            pr = predict(q, t['Xte']).argmax(1).numpy()
            s = summarize(pr, d['yte'], d['ste'])
            pc = paired(fp, pr, d['yte'], d['ste'])
            row = dict(run=run, spec=spec, overall=s['overall'], macro_f1=s['macro_f1'], delta_pp=pc['delta_pp'],
                       changed=pc['changed'], worse=pc['worse'], better=pc['better'], mcnemar_p=pc['mcnemar_p'],
                       weight_MB=model_bits(q, model) / 1e6, per_snr=json.dumps(s['per_snr']),
                       delta_per_snr=json.dumps(pc['delta_pp_per_snr']))
            rows.append(row)
            per_spec.setdefault(spec, []).append(pc['delta_pp'])
            if args.save_pred:
                tag = os.path.basename(run.rstrip('/')) + '__' + spec.replace('=', '-').replace(',', '_')
                np.savez_compressed(os.path.join(args.out, f'pred_{tag}.npz'), pred=pr.astype(np.int8),
                                    fp32=fp.astype(np.int8), y=np.asarray(d['yte'], np.int8), snr=np.asarray(d['ste'], np.int8))
            print(f'  {spec:42s} {100 * s["overall"]:6.2f}%  delta {pc["delta_pp"]:+7.2f} pp  '
                  f'flips {pc["changed"]:5d} (worse {pc["worse"]}, better {pc["better"]}, McNemar p={pc["mcnemar_p"]:.3g})  '
                  f'{row["weight_MB"]:.3f} MB')
            del q
            if dev.type == 'cuda':
                torch.cuda.empty_cache()

    keys = list(rows[0].keys()) + ['delta_per_snr']
    with open(os.path.join(args.out, 'ptq_runs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    summary = []
    for spec, deltas in per_spec.items():
        acc = [r['overall'] for r in rows if r['spec'] == spec]
        summary.append(dict(spec=spec, n=len(acc), acc_mean=100 * np.mean(acc), acc_std=100 * np.std(acc, ddof=1) if len(acc) > 1 else 0,
                            delta_mean_pp=np.mean(deltas), delta_std_pp=np.std(deltas, ddof=1) if len(deltas) > 1 else 0,
                            **{k: v for k, v in tost(deltas).items() if k in ('p_tost', 'equivalent')}))
    with open(os.path.join(args.out, 'ptq_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    save_json(dict(args=vars(args), summary=summary), os.path.join(args.out, 'ptq_summary.json'))
    print(f'\nwrote {args.out}/ptq_runs.csv and ptq_summary.csv')


if __name__ == '__main__':
    main()
