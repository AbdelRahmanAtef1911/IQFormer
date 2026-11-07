"""D2 - is anything left to learn below -10 dB?  Window statistics vs. IQFormer, on the VALIDATION set.

Computes 10 simple statistics of every window (power, envelope kurtosis, spectral flatness, spectral peak ratio,
spectral asymmetry (one-sided spectrum = AM-SSB cue), carrier ratio, low-frequency power fraction,
power fluctuation, |C40|, C42) and fits small logistic-regression classifiers on the TRAINING split:

  A. per-SNR classifier  (one per SNR level: it is told the SNR, like an oracle) - 11-class accuracy
  B. pooled classifier   (one for all SNRs: blind)                              - 11-class accuracy
  C. AM-SSB vs. rest     (per SNR, ROC AUC)                                     - is AM-SSB separable from noise?

and prints them next to IQFormer's per-SNR validation accuracy (from runs_arch/iqformer_s*/val_pred.npz).
It also prints the mean window power per class, which shows whether the dataset normalises the windows.

  python -m amcq_next.d2_lowsnr_stats
"""
import argparse
import glob
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import amcq_next  # noqa: F401
from amcq import setup_repo, default_data
from amcq.data import load_rml2016a, CLASSES_2016A

NAMES = ['log_power', 'env_kurtosis', 'spec_flatness', 'spec_peak', 'spec_asym', 'carrier', 'lowfreq',
         'power_fluct', 'abs_C40', 'C42']
SSB = CLASSES_2016A.index('AM-SSB')


def features(X):
    z = X[:, 0].astype(np.float64) + 1j * X[:, 1].astype(np.float64)
    p = (np.abs(z) ** 2).mean(1)
    zn = z / np.sqrt(p)[:, None]
    a2 = np.abs(zn) ** 2
    kurt = (a2 ** 2).mean(1)
    S = np.abs(np.fft.fftshift(np.fft.fft(zn, axis=1), axes=1)) ** 2 + 1e-12
    flat = np.exp(np.log(S).mean(1)) / S.mean(1)
    peak = S.max(1) / S.mean(1)
    n = S.shape[1]
    pos, neg = S[:, n // 2 + 1:].sum(1), S[:, :n // 2].sum(1)
    asym = np.abs(pos - neg) / (pos + neg)
    carrier = np.abs(zn.mean(1)) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(n))
    low = S[:, np.abs(f) < 0.1].sum(1) / S.sum(1)
    fluct = a2.std(1)
    m20 = (zn ** 2).mean(1); m21 = a2.mean(1)
    m40 = (zn ** 4).mean(1); m42 = (a2 ** 2).mean(1)
    c40 = np.abs(m40 - 3 * m20 ** 2)
    c42 = (m42 - np.abs(m20) ** 2 - 2 * m21 ** 2).real
    return np.stack([np.log10(p), kurt, flat, peak, asym, carrier, low, fluct, c40, c42], 1)


def fit(Xf, y):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    return clf.fit(Xf, y)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--repo', default=None)
    pa.add_argument('--data', default=None)
    pa.add_argument('--baseline', default='runs_arch/iqformer_s*')
    pa.add_argument('--out', default='results_next/d2_lowsnr_stats.txt')
    a = pa.parse_args()
    repo = setup_repo(a.repo)
    d = load_rml2016a(a.data or default_data(repo))
    Ftr, Fva = features(d['Xtr']), features(d['Xva'])
    ytr, yva, s_tr, s_va = d['ytr'], d['yva'], d['str'], d['sva']
    snrs = sorted(np.unique(s_va).tolist())
    lines = []
    out = lambda s='': (print(s), lines.append(s))                      # noqa: E731

    out('Mean log10 window power by class (training split) - equal rows mean the windows are NOT power-normalised '
        'per class:')
    show = [s for s in (-20, -10, 0, 10, 18) if s in snrs]
    out(f'{"class":8s}' + ''.join(f'{s:>+9d}dB' for s in show))
    for k, c in enumerate(CLASSES_2016A):
        out(f'{c:8s}' + ''.join(f'{Ftr[(ytr == k) & (s_tr == s), 0].mean():11.3f}' for s in show))

    base = {}
    for f in sorted(glob.glob(os.path.join(a.baseline, 'val_pred.npz'))):
        z = np.load(f)
        for s in snrs:
            m = z['snr'] == s
            base.setdefault(s, []).append(100 * (z['pred'][m] == z['y'][m]).mean())
    pooled = fit(Ftr, ytr)
    out('\nVALIDATION accuracy (%) per SNR; AUC = AM-SSB vs. rest from the statistics')
    out(f'{"SNR":>5s} {"IQFormer":>9s} {"stats,SNR known":>16s} {"stats,blind":>12s} {"AM-SSB AUC":>11s}  '
        f'{"AM-SSB AUC power only":>21s}')
    for s in snrs:
        mtr, mva = s_tr == s, s_va == s
        per = fit(Ftr[mtr], ytr[mtr])
        acc_per = 100 * (per.predict(Fva[mva]) == yva[mva]).mean()
        acc_pool = 100 * (pooled.predict(Fva[mva]) == yva[mva]).mean()
        ssb = fit(Ftr[mtr], (ytr[mtr] == SSB).astype(int))
        auc = roc_auc_score(yva[mva] == SSB, ssb.predict_proba(Fva[mva])[:, 1])
        ssbp = fit(Ftr[mtr][:, :1], (ytr[mtr] == SSB).astype(int))
        aucp = roc_auc_score(yva[mva] == SSB, ssbp.predict_proba(Fva[mva][:, :1])[:, 1])
        b = f'{np.mean(base[s]):9.2f}' if s in base else f'{"n/a":>9s}'
        out(f'{s:+5d} {b} {acc_per:16.2f} {acc_pool:12.2f} {auc:11.3f}  {aucp:21.3f}')
    out('\nReading: AUC 0.5 = no information; > 0.7 = a usable cue. If the statistics beat IQFormer at -14..-8 dB, or '
        'AM-SSB is separable there, a gain below -10 dB is physically possible (MSTFFNet-style SNR branch worth trying '
        'first). If not, gains must come from -8..0 dB.')
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    open(a.out, 'w').write('\n'.join(lines) + '\n')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
