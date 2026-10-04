"""RadioML 2016.10A loading, the exact split of the released IQFormer code, GPU STFT features
and I/Q data augmentation.

Split: identical to IQFormer main.py. For every (modulation, SNR) key the 1,000 frames are split
80/20 into train+val / test with random_state=233, then the 80 % part 75/25 into train / val
(random_state=233). Result: 132,000 train, 44,000 val, 44,000 test frames.
"""
import os
import numpy as np
import torch
from scipy.signal import get_window
from sklearn.model_selection import train_test_split

CLASSES_2016A = ['8PSK', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'AM-DSB', 'AM-SSB', 'WBFM']


def _read_pickle(path):
    import pandas as pd                      # the released code reads it this way (handles the Python-2 pickle)
    return pd.read_pickle(path)


def load_rml2016a(path, min_snr=-20, max_snr=18, test_size=0.2, seed=233, cache=True, verbose=True):
    """Return dict of numpy arrays: X* (N,2,128) float32, y* (N,) int64, s* (N,) int64 for tr/va/te."""
    cache_file = f'{path}.split_seed{seed}_snr{min_snr}_{max_snr}.npz'
    if cache and os.path.exists(cache_file):
        d = dict(np.load(cache_file))
        if verbose:
            print(f'[data] loaded cached split {cache_file}')
        return d
    data = _read_pickle(path)
    parts = {k: ([], [], []) for k in ('tr', 'va', 'te')}
    for (label, snr), samples in data.items():
        if snr < min_snr or snr > max_snr or label not in CLASSES_2016A:
            continue
        labels = np.full(len(samples), CLASSES_2016A.index(label))
        snrs = np.full(len(samples), snr)
        X, x, Y, y, S_tr, S_te = train_test_split(samples, labels, snrs, test_size=test_size,
                                                  random_state=seed, stratify=labels)
        tr, va, ytr, yva, s_tr, s_va = train_test_split(X, Y, S_tr, test_size=0.25,
                                                        random_state=seed, stratify=Y)
        for key, (a, b, c) in (('tr', (tr, ytr, s_tr)), ('va', (va, yva, s_va)), ('te', (x, y, S_te))):
            parts[key][0].append(a); parts[key][1].append(b); parts[key][2].append(c)
    d = {}
    for key, (a, b, c) in parts.items():
        d['X' + key] = np.concatenate(a).astype(np.float32)
        d['y' + key] = np.concatenate(b).astype(np.int64)
        d['s' + key] = np.concatenate(c).astype(np.int64)
    if cache:
        np.savez(cache_file, **d)
    if verbose:
        print(f"[data] train {len(d['ytr'])}  val {len(d['yva'])}  test {len(d['yte'])}")
    return d


def to_device(d, device):
    return {k: torch.from_numpy(v).to(device) for k, v in d.items()}


# ------------------------------------------------------------------ STFT on the GPU
_WIN = {}


def _window(device):
    key = str(device)
    if key not in _WIN:
        w = torch.from_numpy(get_window('blackman', 31).astype(np.float32)).to(device)
        _WIN[key] = w
    return _WIN[key]


def stft_bins(sig, nbins=32, nfft=128):
    """Exactly scipy.signal.stft(sig, 1.0, 'blackman', 31, 30, nfft) for a (B,128) real or complex signal
    (two-sided for complex input), restricted to the first `nbins` bins. Returns complex (B, nbins, 128)."""
    w = _window(sig.device)
    pad = torch.nn.functional.pad(sig.unsqueeze(1), (15, 15)).squeeze(1) if not torch.is_complex(sig) else \
        torch.complex(torch.nn.functional.pad(sig.real, (15, 15)), torch.nn.functional.pad(sig.imag, (15, 15)))
    frames = pad.unfold(-1, 31, 1)                                  # (B, 128, 31)
    frames = frames * w
    spec = torch.fft.fft(frames, n=nfft) if torch.is_complex(sig) else torch.fft.rfft(frames, n=nfft)
    spec = spec / w.sum()
    return spec[..., :nbins].transpose(1, 2)                        # (B, nbins, 128)


STFT_CHANNELS = {'real': 1, 'complex': 2, 'iq_complex': 4, 'mag': 1, 'cplx_logmag': 1}


def stft_features(x, mode='real'):
    """x: (B,2,128). 'real' reproduces the released code (real part of the STFT of the I row only)."""
    if mode == 'real':
        return stft_bins(x[:, 0]).real.unsqueeze(1)
    if mode == 'mag':
        return stft_bins(x[:, 0]).abs().unsqueeze(1)
    if mode == 'cplx_logmag':                                       # the earlier 'cplxwide' experiment:
        z = torch.complex(x[:, 0], x[:, 1])                         # STFT of I+jQ, nfft=64, two-sided,
        s = stft_bins(z, nbins=64, nfft=64)                         # fftshift, bins 16:48 (-0.25..+0.234 fs),
        s = torch.fft.fftshift(s, dim=1)[:, 16:48]                  # log(1+|.|)
        return torch.log1p(s.abs()).unsqueeze(1)
    if mode == 'complex':                                           # real + imaginary part of the I-row STFT
        s = stft_bins(x[:, 0])
        return torch.stack([s.real, s.imag], 1)
    if mode == 'iq_complex':                                        # real + imaginary of I-row and Q-row STFTs
        si, sq = stft_bins(x[:, 0]), stft_bins(x[:, 1])
        return torch.stack([si.real, si.imag, sq.real, sq.imag], 1)
    raise ValueError(mode)


def iq_input(x, mode='iq'):
    """Time-domain input channels: 'iq' (I,Q) as released, or 'iqap' (I,Q, amplitude, phase/pi)."""
    if mode == 'iq':
        return x
    if mode == 'iqap':
        a = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
        p = torch.atan2(x[:, 1], x[:, 0]) / np.pi
        return torch.cat([x, a.unsqueeze(1), p.unsqueeze(1)], 1)
    raise ValueError(mode)


# ------------------------------------------------------------------ augmentation (training only)
def augment(x, kinds):
    """kinds: set of 'rot' (random multiple of 90 degrees), 'flip' (random I or Q sign flip),
    'rev' (time reversal). Applied per frame on the GPU. Huang et al., IEEE Access 2020."""
    if not kinds:
        return x
    B = x.shape[0]
    if 'rot' in kinds:
        k = torch.randint(0, 4, (B,), device=x.device)
        i, q = x[:, 0], x[:, 1]
        ri = torch.stack([i, -q, -i, q], 0)                          # 0, 90, 180, 270 degrees
        rq = torch.stack([q, i, -q, -i], 0)
        ar = torch.arange(B, device=x.device)
        x = torch.stack([ri[k, ar], rq[k, ar]], 1)
    if 'flip' in kinds:
        m = torch.randint(0, 3, (B,), device=x.device)               # 0 none, 1 flip I, 2 flip Q
        sign = torch.ones(B, 2, 1, device=x.device)
        sign[m == 1, 0] = -1
        sign[m == 2, 1] = -1
        x = x * sign
    if 'rev' in kinds:
        m = torch.rand(B, device=x.device) < 0.5
        x = torch.where(m[:, None, None], x.flip(-1), x)
    return x
