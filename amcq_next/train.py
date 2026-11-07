"""Train one model for the next-round experiments. VALIDATION ONLY unless --final-test is given.

Same data split, optimiser and selection as amcq_arch/train.py (AdamW, batch 256, best epoch by validation
accuracy). Additions:
  --aug phase              continuous random carrier-phase rotation (uniform 0..360 degrees), per window
                           (can be combined with the toolkit's rot,flip,rev)
  --snr-loss W             weight of the SNR-bin cross-entropy for iqfx_*snr* models (true SNR used ONLY as a
                           training target; 0 = branch without SNR supervision)
  --ctr-weight W           weight of the I/Q-STFT InfoNCE loss for iqfx_*ctr* models (training only)
  --teachers RUN ...       distillation from the averaged outputs of these trained runs (--kd-alpha, --kd-t)
Writes log.csv, best.pt, metrics.json + val_metrics.json, val_pred.npz and val_logits.npz (float16, for ensembles).

Examples
  python -m amcq_next.train --model iqformer --sched cosine --epochs 100 --warmup 5 --patience 0 --seed 1 --out runs_next/e1_cos100_s1
  python -m amcq_next.train --model iqfx_snr --snr-loss 0.05 --seed 1 --out runs_next/e2a_snr_s1
  python -m amcq_next.train --model iqfx_oracle --seed 1 --out runs_next/d3_oracle_s1
"""
import argparse
import csv
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import amcq_next  # noqa: F401  (registers iqfx_* and sets the path to amcq)
from amcq import setup_repo, default_data
from amcq_next.models import snr_bin


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--repo', default=None)
    p.add_argument('--data', default=None)
    p.add_argument('--model', default='iqformer')
    p.add_argument('--stft', default='real', choices=['real', 'complex', 'iq_complex', 'mag', 'cplx_logmag'])
    p.add_argument('--iq', default='iq', choices=['iq', 'iqap'])
    p.add_argument('--act', default='gelu', choices=['gelu', 'relu', 'hswish', 'silu', 'relu6'])
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--bs', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=0.01)
    p.add_argument('--sched', default='plateau', choices=['plateau', 'cosine', 'none'])
    p.add_argument('--warmup', type=int, default=0)
    p.add_argument('--min-lr', type=float, default=5e-5)
    p.add_argument('--ls', type=float, default=0.0, help='label smoothing')
    p.add_argument('--aug', default='', help='comma list of phase,rot,flip,rev')
    p.add_argument('--patience', type=int, default=10, help='early stopping (0 = off)')
    p.add_argument('--snr-loss', type=float, default=0.05)
    p.add_argument('--ctr-weight', type=float, default=0.1)
    p.add_argument('--ctr-temp', type=float, default=0.07)
    p.add_argument('--teachers', nargs='*', default=[])
    p.add_argument('--kd-alpha', type=float, default=0.7)
    p.add_argument('--kd-t', type=float, default=3.0)
    p.add_argument('--subset', type=float, default=1.0, help='fraction of training frames (smoke tests)')
    p.add_argument('--final-test', action='store_true', help='also score the held-out TEST set (run once, at the end)')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args(argv)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def phase_rotate(x):
    """Rotate every window by its own uniform random angle: (I, Q) -> (I cos t - Q sin t, I sin t + Q cos t)."""
    t = torch.rand(x.shape[0], 1, device=x.device) * (2 * math.pi)
    c, s = torch.cos(t), torch.sin(t)
    i, q = x[:, 0], x[:, 1]
    return torch.stack([i * c - q * s, i * s + q * c], 1)


def info_nce(a, b, temp):
    """Symmetric InfoNCE: the I/Q and STFT views of the same window are the positive pair."""
    logits = a @ b.t() / temp
    target = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


def forward(model, xb, sb=None, aux=False):
    if getattr(model, 'needs_snr', False) or aux:
        return model(xb, snr=sb, return_aux=aux) if hasattr(model, 'opts') else model(xb)
    return model(xb)


@torch.no_grad()
def logits_of(model, X, S, batch=1024):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        out.append(forward(model, X[i:i + batch], S[i:i + batch]).float().cpu())
    return torch.cat(out)


def main(argv=None):
    a = get_args(argv)
    repo = setup_repo(a.repo)
    from amcq.data import load_rml2016a, to_device, augment
    from amcq_arch.models import build_model
    from amcq_arch.ptq import load_run
    from amcq.metrics import summarize, save_json
    is_oracle = 'oracle' in a.model
    if is_oracle and a.final_test:
        raise SystemExit('iqfx_oracle is a diagnostic: it needs the true SNR and is never run on the test set')

    os.makedirs(a.out, exist_ok=True)
    set_seed(a.seed)
    dev = torch.device(a.device)
    d = load_rml2016a(a.data or default_data(repo))
    if a.subset < 1.0:
        keep = np.random.default_rng(0).permutation(len(d['ytr']))[:int(a.subset * len(d['ytr']))]
        for k in ('Xtr', 'ytr', 'str'):
            d[k] = d[k][keep]
    t = to_device(d, dev)

    model = build_model(a.model, stft=a.stft, iq=a.iq, act=a.act).to(dev)
    opts = getattr(model, 'opts', set())
    use_snr = 'snr' in opts and a.snr_loss > 0
    use_ctr = 'ctr' in opts and a.ctr_weight > 0
    n_params = sum(p.numel() for p in model.parameters())
    n_infer = n_params - sum(p.numel() for n_, p in model.named_parameters() if n_.startswith('proj_'))
    print(f'[model] {a.model}  params {n_params:,} (inference {n_infer:,})  snr-loss {a.snr_loss if "snr" in opts else "-"}'
          f'  ctr {a.ctr_weight if "ctr" in opts else "-"}  device {dev}')

    teachers = [load_run(r, dev, None) for r in a.teachers]
    for tm in teachers:
        for p_ in tm.parameters():
            p_.requires_grad_(False)
    if teachers:
        print(f'[kd] {len(teachers)} teachers, alpha {a.kd_alpha}, T {a.kd_t}')

    crit = nn.CrossEntropyLoss(label_smoothing=a.ls)
    val_crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    if a.sched == 'plateau':
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=3, min_lr=a.min_lr)
    elif a.sched == 'cosine':
        f_min = a.min_lr / a.lr
        def lam(e):
            if e < a.warmup:
                return (e + 1) / a.warmup
            prog = (e - a.warmup) / max(1, a.epochs - a.warmup)
            return f_min + (1 - f_min) * 0.5 * (1 + math.cos(math.pi * prog))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    else:
        sched = None
    aug = set(x for x in a.aug.split(',') if x)
    phase = 'phase' in aug
    aug.discard('phase')

    best_acc, bad, log = -1.0, 0, []
    ckpt = os.path.join(a.out, 'best.pt')
    n = len(t['ytr'])
    for ep in range(a.epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(n, device=dev)
        tot, corr, seen, snr_hit = 0.0, 0, 0, 0
        for i in range(0, n, a.bs):
            idx = perm[i:i + a.bs]
            if len(idx) < 2:
                continue
            xb, yb, sb = t['Xtr'][idx], t['ytr'][idx], t['str'][idx]
            if phase:
                xb = phase_rotate(xb)
            xb = augment(xb, aug)
            want_aux = use_snr or use_ctr
            if want_aux:
                out, aux = forward(model, xb, sb, aux=True)
            else:
                out, aux = forward(model, xb, sb), {}
            loss = crit(out, yb)
            if use_snr:
                loss = loss + a.snr_loss * F.cross_entropy(aux['snr_logits'], snr_bin(sb))
                snr_hit += (aux['snr_logits'].argmax(1) == snr_bin(sb)).sum().item()
            if use_ctr:
                loss = loss + a.ctr_weight * info_nce(aux['z_iq'], aux['z_tf'], a.ctr_temp)
            if teachers:
                with torch.no_grad():
                    tl = sum(tm(xb) for tm in teachers) / len(teachers)
                kd = F.kl_div(F.log_softmax(out / a.kd_t, 1), F.softmax(tl / a.kd_t, 1), reduction='batchmean') * a.kd_t ** 2
                loss = (1 - a.kd_alpha) * loss + a.kd_alpha * kd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx); corr += (out.argmax(1) == yb).sum().item(); seen += len(idx)
        vlog = logits_of(model, t['Xva'], t['sva'])
        vl = val_crit(vlog, d_y := torch.from_numpy(d['yva'])).item()
        va = (vlog.argmax(1) == d_y).float().mean().item()
        if a.sched == 'plateau':
            sched.step(vl)
        elif sched is not None:
            sched.step()
        lr = opt.param_groups[0]['lr']
        improved = va > best_acc
        if improved:
            best_acc, bad = va, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
        row = dict(epoch=ep, train_loss=tot / seen, train_acc=corr / seen, val_loss=vl, val_acc=va, lr=lr,
                   sec=round(time.time() - t0, 1))
        if use_snr:
            row['snr_bin_acc'] = snr_hit / seen
        log.append(row)
        print(f"ep {ep:3d}  train {row['train_loss']:.4f}/{row['train_acc']:.4f}  val {vl:.4f}/{va:.4f}  lr {lr:.2e}"
              + (f"  snr-bin {row['snr_bin_acc']:.3f}" if use_snr else '') + f"  {row['sec']}s" + ('  *' if improved else ''),
              flush=True)
        if a.patience and bad >= a.patience:
            print(f'early stop at epoch {ep}')
            break
    if log:
        with open(os.path.join(a.out, 'log.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(log[-1])); w.writeheader()
            for r in log:
                w.writerow({k: r.get(k, '') for k in log[-1]})

    model.load_state_dict(torch.load(ckpt, map_location=dev))
    vlog = logits_of(model, t['Xva'], t['sva'])
    vpred = vlog.argmax(1).numpy()
    res = summarize(vpred, d['yva'], d['sva'])
    res.update(model=a.model, params=n_params, inference_params=n_infer, args=vars(a), best_val_acc=best_acc,
               epochs_run=len(log), split='validation')
    res['transition_band'] = float(np.mean([res['per_snr'][s] for s in (-6, -4, -2, 0)]))
    np.savez_compressed(os.path.join(a.out, 'val_pred.npz'), pred=vpred, y=d['yva'], snr=d['sva'])
    np.savez_compressed(os.path.join(a.out, 'val_logits.npz'), logits=vlog.numpy().astype(np.float16))
    save_json(res, os.path.join(a.out, 'val_metrics.json'))
    print(f"[validation] overall {100 * res['overall']:.2f}%  -6..0 dB {100 * res['transition_band']:.2f}%  "
          f"(-20..0 {100 * res['mean_snr_le0']:.2f}%, 0..18 {100 * res['mean_snr_ge0']:.2f}%)  "
          f"epochs {len(log)}  params {n_params:,}")
    if not a.final_test:
        save_json(res, os.path.join(a.out, 'metrics.json'))              # PTQ/QAT/TTA read model args from here
        return res
    tlog = logits_of(model, t['Xte'], t['ste'])
    tpred = tlog.argmax(1).numpy()
    tres = summarize(tpred, d['yte'], d['ste'])
    tres.update(model=a.model, params=n_params, args=vars(a), split='test')
    np.savez_compressed(os.path.join(a.out, 'test_pred.npz'), pred=tpred, y=d['yte'], snr=d['ste'],
                        logits=tlog.numpy().astype(np.float16))
    save_json(tres, os.path.join(a.out, 'metrics.json'))
    print(f"[TEST] overall {100 * tres['overall']:.2f}%  (-20..0 {100 * tres['mean_snr_le0']:.2f}%, "
          f"0..18 {100 * tres['mean_snr_ge0']:.2f}%)")
    return tres


if __name__ == '__main__':
    main()
