"""Classical AMC features, computed per window on the GPU (no learned parameters).

cumulant_features(x): 7 phase-invariant higher-order cumulants of the unit-power complex signal z = I + jQ
    |C20|, |C40|, |C41|, Re C42, |C60|, |C61|, Re C63  (Swami & Sadler, IEEE Trans. Commun. 2000).
    Cumulants of order > 2 of Gaussian noise are zero, so they suppress noise by construction, and their values
    separate PSK and QAM orders (e.g. |C40| = 2, 1, 0, 0.68 for BPSK, QPSK, 8PSK, 16QAM at unit power).
    They are computed exactly from moments with the general moment-to-cumulant (set-partition) formula,
    so no hand-written formula can be wrong. A signed log1p keeps the values in a small range.

power_spectra(x): |FFT(z^k)| for k = 2, 4, 8, each divided by its own mean (peak-to-average shape) and log1p'd.
    Raising M-PSK to the M-th power removes the modulation and leaves a spectral line at M times the carrier
    offset (BPSK in z^2, QPSK in z^4, 8PSK in z^8); the line builds up over the whole window (processing gain).
"""
import math
from functools import lru_cache
import torch


def unit_power(x):
    """x: (B,2,N) real I/Q -> complex (B,N) with mean |z|^2 = 1 per window."""
    z = torch.complex(x[:, 0].float(), x[:, 1].float())
    p = (z.real ** 2 + z.imag ** 2).mean(dim=1, keepdim=True)
    return z / torch.sqrt(p + 1e-12)


def _partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in _partitions(rest):
        for i in range(len(part)):
            yield part[:i] + [[first] + part[i]] + part[i + 1:]
        yield [[first]] + part


@lru_cache(maxsize=None)
def _cumulant_terms(p, q):
    """Terms of cum(z,...,z (p-q times), z*,...,z* (q times)) as (coefficient, ((n, c), ...)) over set partitions;
    each block is a moment E[z^(n-c) conj(z)^c]."""
    conj = [0] * (p - q) + [1] * q
    terms = {}
    for part in _partitions(list(range(p))):
        k = len(part)
        coef = (-1) ** (k - 1) * math.factorial(k - 1)
        blocks = tuple(sorted((len(b), sum(conj[i] for i in b)) for b in part))
        terms[blocks] = terms.get(blocks, 0) + coef
    return tuple((c, b) for b, c in terms.items() if c != 0)


def cumulant(z, p, q):
    """Sample cumulant C_pq of complex z (B,N) -> complex (B,)."""
    moments = {}
    total = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
    for coef, blocks in _cumulant_terms(p, q):
        prod = torch.ones_like(total)
        for n, c in blocks:
            if (n, c) not in moments:
                moments[(n, c)] = (z ** (n - c) * z.conj() ** c).mean(dim=1)
            prod = prod * moments[(n, c)]
        total = total + coef * prod
    return total


def raw_cumulants(x):
    """(B,2,N) -> (B,7) real: |C20|, |C40|, |C41|, Re C42, |C60|, |C61|, Re C63 of the unit-power signal."""
    z = unit_power(x)
    c = {pq: cumulant(z, *pq) for pq in ((2, 0), (4, 0), (4, 1), (4, 2), (6, 0), (6, 1), (6, 3))}
    return torch.stack([c[(2, 0)].abs(), c[(4, 0)].abs(), c[(4, 1)].abs(), c[(4, 2)].real,
                        c[(6, 0)].abs(), c[(6, 1)].abs(), c[(6, 3)].real], dim=1)


def cumulant_features(x):
    f = raw_cumulants(x)
    return torch.sign(f) * torch.log1p(f.abs())


def power_spectra(x, powers=(2, 4, 8)):
    """(B,2,N) -> (B,len(powers),N): log1p of |FFT(z^k)| / mean, fftshifted (zero frequency in the middle)."""
    z = unit_power(x)
    out = []
    for k in powers:
        mag = torch.fft.fft(z ** k, dim=1).abs()
        mag = mag / (mag.mean(dim=1, keepdim=True) + 1e-12)
        out.append(torch.log1p(torch.fft.fftshift(mag, dim=1)))
    return torch.stack(out, dim=1)
