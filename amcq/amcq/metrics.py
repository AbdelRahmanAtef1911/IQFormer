"""Evaluation helpers: accuracy per SNR, macro F1, confusion matrices, paired statistics."""
import json
import numpy as np
import torch
from scipy import stats
from sklearn.metrics import f1_score, confusion_matrix

from .data import CLASSES_2016A


@torch.no_grad()
def predict(model, X, batch=1024, tta=False):
    """Return logits (N, C) as a CPU float tensor. tta: average over the four 90-degree rotations."""
    model.eval()
    outs = []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch]
        if tta:
            i_, q_ = xb[:, 0], xb[:, 1]
            views = [xb, torch.stack([-q_, i_], 1), -xb, torch.stack([q_, -i_], 1)]
            out = sum(torch.softmax(model(v), 1) for v in views) / 4
        else:
            out = model(xb)
        outs.append(out.float().cpu())
    return torch.cat(outs)


def summarize(pred, y, snr, classes=CLASSES_2016A):
    """pred, y, snr: numpy int arrays. Returns a dict of headline metrics and per-SNR accuracy."""
    pred, y, snr = map(np.asarray, (pred, y, snr))
    snrs = sorted(np.unique(snr).tolist())
    per_snr = {int(s): float((pred[snr == s] == y[snr == s]).mean()) for s in snrs}
    low = [per_snr[s] for s in snrs if s <= 0]
    high = [per_snr[s] for s in snrs if s >= 0]
    out = dict(overall=float((pred == y).mean()),
               macro_f1=float(f1_score(y, pred, average='macro')),
               peak=float(max(per_snr.values())),
               mean_snr_le0=float(np.mean(low)) if low else None,      # paper: -20..0 dB
               mean_snr_ge0=float(np.mean(high)) if high else None,    # paper: 0..18 dB
               per_snr=per_snr)
    # per-class accuracy at every SNR
    out['per_class_snr'] = {int(s): {c: float((pred[(snr == s) & (y == k)] == k).mean())
                                     for k, c in enumerate(classes) if ((snr == s) & (y == k)).any()}
                            for s in snrs}
    # largest confusions at SNR >= 0 dB
    m = snr >= 0
    cm = confusion_matrix(y[m], pred[m], labels=list(range(len(classes)))).astype(float)
    cm /= cm.sum(1, keepdims=True).clip(min=1)
    pairs = [(classes[i], classes[j], float(cm[i, j])) for i in range(len(classes)) for j in range(len(classes)) if i != j]
    out['top_confusions_snr_ge0'] = sorted(pairs, key=lambda t: -t[2])[:8]
    return out


def confusion_at(pred, y, snr, s, n=11):
    m = snr == s
    cm = confusion_matrix(y[m], pred[m], labels=list(range(n))).astype(float)
    return cm / cm.sum(1, keepdims=True).clip(min=1)


def paired(pred_a, pred_b, y, snr=None, margin_pp=0.5):
    """Compare two prediction vectors on the SAME frames (e.g. FP32 vs quantized copy of one model).
    Returns accuracy change, flips, exact McNemar p-value."""
    a, b, y = (np.asarray(v) for v in (pred_a, pred_b, y))
    ca, cb = a == y, b == y
    worse = int((ca & ~cb).sum())
    better = int((~ca & cb).sum())
    n = worse + better
    p = float(stats.binomtest(min(worse, better), n, 0.5).pvalue) if n else 1.0
    res = dict(acc_a=float(ca.mean()), acc_b=float(cb.mean()), delta_pp=100 * float(cb.mean() - ca.mean()),
               changed=int((a != b).sum()), worse=worse, better=better, mcnemar_p=p)
    if snr is not None:
        snr = np.asarray(snr)
        res['delta_pp_per_snr'] = {int(s): 100 * float(cb[snr == s].mean() - ca[snr == s].mean())
                                   for s in sorted(np.unique(snr))}
    return res


def tost(deltas_pp, margin_pp=0.5):
    """Two one-sided t-tests for equivalence of paired deltas (one delta per trained model)."""
    d = np.asarray(deltas_pp, float)
    if len(d) < 2:
        return dict(mean=float(d.mean()), p_tost=None)
    p_low = stats.ttest_1samp(d, -margin_pp, alternative='greater').pvalue
    p_high = stats.ttest_1samp(d, margin_pp, alternative='less').pvalue
    return dict(mean=float(d.mean()), sd=float(d.std(ddof=1)), p_tost=float(max(p_low, p_high)),
                equivalent=bool(max(p_low, p_high) < 0.05), margin_pp=margin_pp)


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)
