"""Quantization-aware training (QAT) with learnable step sizes (LSQ, Esser et al., ICLR 2020).

Starts from a trained full-precision run, folds batch normalization, inserts the quantizers of
`--spec`, calibrates them on 1,024 training frames, then fine-tunes weights AND step sizes with the
straight-through estimator. Batch-norm statistics stay frozen. Ranges start from the bit-width-aware
rule (Steps 5d/6: clip 4-bit activations at the 99.9th percentile, min-max for 8-bit ones). The best epoch on the validation set
is tested and compared (paired) with the same model in full precision.

Examples
  python amcq/qat.py --run runs/iqformer_s1 --spec all=W4A4 --epochs 15 --out runs/qat_w4a4_s1
  # with distillation from the three FP32 runs (teacher = their averaged logits):
  python amcq/qat.py --run runs/iqformer_s1 --spec all=W4A4 --teacher runs/iqformer_s1 runs/iqformer_s2 runs/iqformer_s3 \
         --out runs/qatkd_w4a4_s1
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amcq import setup_repo, default_data                     # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--repo', default=None)
    p.add_argument('--data', default=None)
    p.add_argument('--run', required=True)
    p.add_argument('--spec', default='all=W4A4')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--bs', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-4, help='weights')
    p.add_argument('--lr-scale', type=float, default=1e-3, help='step sizes')
    p.add_argument('--aug', default='')
    p.add_argument('--calib-method', default='auto', choices=['minmax', 'percentile', 'auto'],
                   help="auto = percentile (--pct) for activations of 4 bits or fewer, min-max for wider ones")
    p.add_argument('--pct', type=float, default=99.9, help='Step 6: 99.9 is the best clipping point for 4-bit activations')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--calib-seed', type=int, default=0, help='which 1,024 training frames initialize the ranges')
    p.add_argument('--teacher', nargs='*', default=[], help='FP32 runs whose averaged logits teach the student (KD)')
    p.add_argument('--kd-alpha', type=float, default=0.7, help='weight of the distillation term')
    p.add_argument('--kd-T', type=float, default=3.0, help='distillation temperature')
    p.add_argument('--act-scheme', default='affine', choices=['affine', 'symmetric'])
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args(argv)
    repo = setup_repo(args.repo)
    from amcq.data import load_rml2016a, to_device, augment
    from amcq.metrics import predict, summarize, paired, save_json
    from amcq.quant import quantize_model, calibrate, set_learnable, ActQuant, WeightQuant
    from ptq import load_run

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    d = load_rml2016a(args.data or default_data(repo))
    t = to_device(d, dev)
    fp_model = load_run(args.run, dev, args)
    fp_pred = predict(fp_model, t['Xte']).argmax(1).numpy()

    teachers = [load_run(r, dev, args) for r in args.teacher]
    q = quantize_model(fp_model, args.spec, act_scheme=args.act_scheme)
    rng = np.random.default_rng(args.calib_seed)
    cal = torch.from_numpy(rng.choice(len(d['ytr']), 1024, replace=False)).to(dev)
    calibrate(q, t['Xtr'][cal], args.calib_method, args.pct)
    ptq_pred = predict(q, t['Xte']).argmax(1).numpy()
    ptq_acc = float((ptq_pred == d['yte']).mean())
    print(f'FP32 {100 * (fp_pred == d["yte"]).mean():.2f}%   PTQ {args.spec} {100 * ptq_acc:.2f}%  '
          f'(start of QAT; ranges: {args.calib_method}, pct {args.pct})', flush=True)

    set_learnable(q, True)
    scale_params = [m.scale for m in q.modules() if isinstance(m, (ActQuant, WeightQuant))
                    and isinstance(getattr(m, 'scale', None), nn.Parameter) and m.scale.requires_grad]
    sp = set(id(s) for s in scale_params)
    weights = [w for w in q.parameters() if id(w) not in sp and w.requires_grad]
    opt = torch.optim.AdamW([{'params': weights, 'lr': args.lr, 'weight_decay': 0.0},
                             {'params': scale_params, 'lr': args.lr_scale, 'weight_decay': 0.0}])
    steps = args.epochs * math.ceil(len(d['ytr']) / args.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    crit = nn.CrossEntropyLoss()
    aug = set(a for a in args.aug.split(',') if a)
    best, ckpt, n = -1, os.path.join(args.out, 'qat_best.pt'), len(d['ytr'])
    log, t_start = [], time.time()
    for ep in range(args.epochs):
        t0 = time.time()
        q.train()
        for m in q.modules():                                      # frozen BN statistics
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, args.bs):
            idx = perm[i:i + args.bs]
            if len(idx) < 2:
                continue
            xb = augment(t['Xtr'][idx], aug)
            out = q(xb)
            loss = crit(out, t['ytr'][idx])
            if teachers:                                           # Hinton et al. distillation
                with torch.no_grad():
                    tl = sum(tm(xb) for tm in teachers) / len(teachers)
                T = args.kd_T
                kd = F.kl_div(F.log_softmax(out / T, 1), F.softmax(tl / T, 1), reduction='batchmean') * T * T
                loss = (1 - args.kd_alpha) * loss + args.kd_alpha * kd
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        va = float((predict(q, t['Xva']).argmax(1) == t['yva'].cpu()).float().mean())
        sec = time.time() - t0
        log.append(dict(epoch=ep, val_acc=va, lr=opt.param_groups[0]['lr'], sec=round(sec, 1)))
        print(f'ep {ep:2d}  val {100 * va:.2f}%   {sec:.0f} s/epoch (about {sec * (args.epochs - ep - 1) / 60:.0f} min left)', flush=True)
        with open(os.path.join(args.out, 'log.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(log[0])); w.writeheader(); w.writerows(log)
        if va > best:
            best = va
            torch.save(q.state_dict(), ckpt)
    q.load_state_dict(torch.load(ckpt, map_location=dev))
    pr = predict(q, t['Xte']).argmax(1).numpy()
    res = summarize(pr, d['yte'], d['ste'])
    res['paired_vs_fp32'] = paired(fp_pred, pr, d['yte'], d['ste'])
    res['ptq_overall'] = ptq_acc
    res['fp32_overall'] = float((fp_pred == d['yte']).mean())
    res['best_val_acc'] = best
    res['minutes'] = round((time.time() - t_start) / 60, 1)
    res['args'] = vars(args)
    save_json(res, os.path.join(args.out, 'metrics.json'))
    np.savez_compressed(os.path.join(args.out, 'test_pred.npz'), pred=pr, y=d['yte'], snr=d['ste'])
    print(f"QAT {args.spec}: {100 * res['overall']:.2f}%  (PTQ {100 * ptq_acc:.2f}%, "
          f"delta vs FP32 {res['paired_vs_fp32']['delta_pp']:+.2f} pp)")


if __name__ == '__main__':
    main()
