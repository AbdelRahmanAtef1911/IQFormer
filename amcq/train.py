"""Train and test one model on RadioML 2016.10A with the released IQFormer split.

Defaults reproduce the released training recipe: AdamW (lr 1e-3, weight decay 0.01), batch 256,
up to 60 epochs, ReduceLROnPlateau on validation loss (factor 0.5, patience 3, min 5e-5),
best checkpoint by validation accuracy, early stopping after 10 epochs without improvement.

Examples
  python amcq/train.py --model iqformer --seed 1 --out runs/iqformer_s1
  python amcq/train.py --model mlp --seed 1 --out runs/mlp_s1
  python amcq/train.py --model iqformer --aug rot --ls 0.1 --seed 1 --out runs/iqf_rot_ls_s1
  python amcq/train.py --model iqformer --init ../save_models/.../weight.pt --epochs 0 --out runs/eval_old
"""
import argparse
import csv
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amcq import setup_repo, default_data                     # noqa: E402


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--repo', default=None, help='IQFormer repository root (default: parent of amcq/)')
    p.add_argument('--data', default=None, help='RML2016.10a_dict.pkl (default: <repo>/dataset/)')
    p.add_argument('--model', default='iqformer')
    p.add_argument('--stft', default='real', choices=['real', 'complex', 'iq_complex', 'mag', 'cplx_logmag'])
    p.add_argument('--iq', default='iq', choices=['iq', 'iqap'])
    p.add_argument('--act', default='gelu', choices=['gelu', 'relu', 'hswish', 'silu', 'relu6'])
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--bs', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=0.01)
    p.add_argument('--opt', default='adamw', choices=['adamw', 'adam', 'sgd', 'radam', 'nadam'])
    p.add_argument('--sched', default='plateau', choices=['plateau', 'cosine', 'none'])
    p.add_argument('--warmup', type=int, default=0, help='linear warm-up epochs (cosine only)')
    p.add_argument('--min-lr', type=float, default=5e-5)
    p.add_argument('--loss', default='ce', choices=['ce', 'focal'])
    p.add_argument('--ls', type=float, default=0.0, help='label smoothing (0 = off)')
    p.add_argument('--focal-gamma', type=float, default=2.0)
    p.add_argument('--aug', default='', help="comma list of rot,flip,rev (empty = none, as released)")
    p.add_argument('--patience', type=int, default=10, help='early stopping (0 = off)')
    p.add_argument('--init', default=None, help='start from a checkpoint (released weight.pt or best.pt)')
    p.add_argument('--subset', type=float, default=1.0, help='fraction of training frames (smoke tests)')
    p.add_argument('--tta', action='store_true', help='also report 4-rotation test-time augmentation')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args(argv)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def make_opt(args, params):
    if args.opt == 'adamw':
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    if args.opt == 'adam':
        return torch.optim.Adam(params, lr=args.lr)
    if args.opt == 'radam':
        return torch.optim.RAdam(params, lr=args.lr, weight_decay=args.wd, decoupled_weight_decay=True)
    if args.opt == 'nadam':
        return torch.optim.NAdam(params, lr=args.lr, weight_decay=args.wd, decoupled_weight_decay=True)
    return torch.optim.SGD(params, lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.wd)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, ls=0.0):
        super().__init__()
        self.gamma, self.ls = gamma, ls

    def forward(self, logits, y):
        ce = F.cross_entropy(logits, y, reduction='none', label_smoothing=self.ls)
        pt = torch.exp(-F.cross_entropy(logits, y, reduction='none'))
        return ((1 - pt) ** self.gamma * ce).mean()


@torch.no_grad()
def evaluate(model, X, y, crit, batch=1024):
    model.eval()
    loss, correct, logits = 0.0, 0, []
    for i in range(0, len(X), batch):
        out = model(X[i:i + batch])
        loss += crit(out, y[i:i + batch]).item() * len(out)
        correct += (out.argmax(1) == y[i:i + batch]).sum().item()
        logits.append(out.float().cpu())
    return loss / len(X), correct / len(X), torch.cat(logits)


def main(argv=None):
    args = get_args(argv)
    repo = setup_repo(args.repo)
    from amcq.data import load_rml2016a, to_device, augment
    from amcq.models import build_model, count_params
    from amcq.metrics import summarize, save_json, predict

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    dev = torch.device(args.device)
    d = load_rml2016a(args.data or default_data(repo))
    if args.subset < 1.0:
        rng = np.random.default_rng(0)
        keep = rng.permutation(len(d['ytr']))[:int(args.subset * len(d['ytr']))]
        for k in ('Xtr', 'ytr', 'str'):
            d[k] = d[k][keep]
    t = to_device(d, dev)

    model = build_model(args.model, stft=args.stft, iq=args.iq, act=args.act).to(dev)
    if args.init:
        if hasattr(model, 'load_legacy'):
            model.load_legacy(args.init, dev)
        else:
            model.load_state_dict(torch.load(args.init, map_location=dev))
    n_params = count_params(model)
    print(f'[model] {args.model}  params {n_params:,}  device {dev}')

    crit = FocalLoss(args.focal_gamma, args.ls) if args.loss == 'focal' else nn.CrossEntropyLoss(label_smoothing=args.ls)
    val_crit = nn.CrossEntropyLoss()                               # model selection always on plain CE
    opt = make_opt(args, model.parameters())
    if args.sched == 'plateau':
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=3, min_lr=args.min_lr)
    elif args.sched == 'cosine':
        f_min = args.min_lr / args.lr
        def lam(e):
            if e < args.warmup:
                return (e + 1) / args.warmup
            prog = (e - args.warmup) / max(1, args.epochs - args.warmup)
            return f_min + (1 - f_min) * 0.5 * (1 + math.cos(math.pi * prog))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    else:
        sched = None
    aug = set(a for a in args.aug.split(',') if a)

    best_acc, bad, log = -1.0, 0, []
    ckpt = os.path.join(args.out, 'best.pt')
    if args.epochs == 0:
        torch.save(model.state_dict(), ckpt)
    n = len(t['ytr'])
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(n, device=dev)
        tot, corr, seen = 0.0, 0, 0
        for i in range(0, n, args.bs):
            idx = perm[i:i + args.bs]
            if len(idx) < 2:                                       # BatchNorm needs >1 frame
                continue
            xb, yb = t['Xtr'][idx], t['ytr'][idx]
            xb = augment(xb, aug)
            out = model(xb)
            loss = crit(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx); corr += (out.argmax(1) == yb).sum().item(); seen += len(idx)
        vl, va, _ = evaluate(model, t['Xva'], t['yva'], val_crit)
        if args.sched == 'plateau':
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
        log.append(row)
        print(f"ep {ep:3d}  train {row['train_loss']:.4f}/{row['train_acc']:.4f}  val {vl:.4f}/{va:.4f}  "
              f"lr {lr:.2e}  {row['sec']}s" + ('  *' if improved else ''))
        if args.patience and bad >= args.patience:
            print(f'early stop at epoch {ep}')
            break
    if log:
        with open(os.path.join(args.out, 'log.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(log[0])); w.writeheader(); w.writerows(log)

    model.load_state_dict(torch.load(ckpt, map_location=dev))
    logits = predict(model, t['Xte'])
    pred = logits.argmax(1).numpy()
    res = summarize(pred, d['yte'], d['ste'])
    res.update(model=args.model, params=n_params, args=vars(args), best_val_acc=best_acc,
               epochs_run=len(log))
    if args.tta:
        res['tta'] = summarize(predict(model, t['Xte'], tta=True).argmax(1).numpy(), d['yte'], d['ste'])['overall']
    np.savez_compressed(os.path.join(args.out, 'test_pred.npz'), pred=pred, y=d['yte'], snr=d['ste'],
                        logits=logits.numpy().astype(np.float16))
    save_json(res, os.path.join(args.out, 'metrics.json'))
    print(f"[test] overall {100 * res['overall']:.2f}%  peak {100 * res['peak']:.2f}%  "
          f"macro-F1 {res['macro_f1']:.4f}  (-20..0 dB {100 * res['mean_snr_le0']:.2f}%, "
          f"0..18 dB {100 * res['mean_snr_ge0']:.2f}%)")
    return res


if __name__ == '__main__':
    main()
