"""Write a small synthetic file with the same layout as RML2016.10a_dict.pkl (for testing the code only).
  python amcq/tests/make_synthetic.py /tmp/fake_rml.pkl 60
"""
import pickle
import sys

import numpy as np

CLASSES = ['8PSK', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'AM-DSB', 'AM-SSB', 'WBFM']


def frame(k, snr, rng):
    sps, n = 8, 128
    sym = rng.integers(0, 64, n // sps + 1)
    if k in (0, 1, 7):                                  # PSK
        M = {0: 8, 1: 2, 7: 4}[k]
        s = np.exp(1j * 2 * np.pi * (sym % M) / M)
    elif k in (5, 6):                                   # QAM
        M = 4 if k == 5 else 8
        s = (sym % M - (M - 1) / 2) + 1j * ((sym // M) % M - (M - 1) / 2)
    elif k == 4:
        s = (sym % 4 - 1.5).astype(complex)
    elif k in (2, 3):                                   # FSK
        s = np.exp(1j * np.cumsum(np.repeat((sym % 2) * 2 - 1, 1) * (0.5 if k == 2 else 0.3)))
    else:                                               # analog
        t = np.arange(n // sps + 1)
        m = np.sin(2 * np.pi * rng.uniform(0.01, 0.1) * t)
        s = (1 + 0.5 * m) if k == 8 else (m + 1j * np.roll(m, 1) if k == 9 else np.exp(1j * 3 * np.cumsum(m)))
    x = np.repeat(s, sps)[:n]
    x = x / np.sqrt(np.mean(np.abs(x) ** 2) + 1e-9)
    x = x * np.exp(1j * rng.uniform(0, 2 * np.pi))
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2) * 10 ** (-snr / 20)
    x = (x + noise) * 0.01
    return np.stack([x.real, x.imag]).astype(np.float32)


if __name__ == '__main__':
    out, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 60
    rng = np.random.default_rng(0)
    d = {(c, snr): np.stack([frame(k, snr, rng) for _ in range(n)]) for k, c in enumerate(CLASSES)
         for snr in range(-20, 20, 2)}
    with open(out, 'wb') as f:
        pickle.dump(d, f, protocol=2)
    print(f'wrote {out}: {len(d)} keys x {n} frames')
