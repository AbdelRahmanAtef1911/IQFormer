"""Checks before any training:  python -m amcq_dsp.smoke   (expect every line to say PASS)."""
import math
import torch
import amcq_dsp  # noqa: F401
from amcq import setup_repo
setup_repo(None)
from amcq_dsp.features import raw_cumulants, power_spectra
from amcq_arch.models import build_model          # patched: knows the DSP names
from amcq.quant import quantize_model, calibrate, QConv, QLinear

torch.manual_seed(0)
N = 200_000


def iq(z):
    return torch.stack([z.real, z.imag], 0).unsqueeze(0).float()


def const(points):
    pts = torch.tensor(points, dtype=torch.complex64)
    pts = pts / torch.sqrt((pts.abs() ** 2).mean())
    return pts[torch.randint(len(pts), (N,))]


qam16 = [complex(a, b) for a in (-3, -1, 1, 3) for b in (-3, -1, 1, 3)]
signals = {'BPSK': const([1, -1]), 'QPSK': const([1, 1j, -1, -1j]),
           '8PSK': const([complex(math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)) for k in range(8)]),
           '16QAM': const(qam16),
           'noise': torch.complex(torch.randn(N), torch.randn(N)) / math.sqrt(2)}
# theory (unit power): |C40|, C42, |C63|
theory = {'BPSK': (2, -2, 16), 'QPSK': (1, -1, 4), '8PSK': (0, -1, 4), '16QAM': (0.68, -0.68, 2.08), 'noise': (0, 0, 0)}
ok = True
for name, z in signals.items():
    c = raw_cumulants(iq(z))[0]
    got = (c[1].item(), c[3].item(), abs(c[6].item()))
    good = all(abs(g - t) < 0.06 * max(1, abs(t)) for g, t in zip(got, theory[name]))
    ok &= good
    print(f'cumulants {name:6s} |C40| {got[0]:+.3f} C42 {got[1]:+.3f} |C63| {got[2]:.3f}  theory {theory[name]}  '
          + ('PASS' if good else 'FAIL'))
z = signals['QPSK'][:4096]
a = raw_cumulants(iq(z)); b = raw_cumulants(iq(z * complex(math.cos(0.7), math.sin(0.7))))
good = torch.allclose(a, b, atol=1e-4); ok &= good
print('cumulants are carrier-phase invariant', 'PASS' if good else 'FAIL')

n = torch.arange(128)
off = torch.exp(1j * 2 * math.pi * 0.05 * n)
# an M-PSK signal shows a spectral line in z^k exactly when k is a multiple of M (k = 2, 4, 8)
for name, first in (('BPSK', 0), ('QPSK', 1), ('8PSK', 2)):
    s = power_spectra(iq(signals[name][:128] * off))[0]
    peaks = s.amax(dim=1)
    good = all((peaks[i] > 2.5) == (i >= first) for i in range(3)) and all(peaks[i] < 2 for i in range(first))
    ok &= good
    print(f'power-law spectra {name}: peak heights x^2,x^4,x^8 = {[round(v, 2) for v in peaks.tolist()]}  '
          + ('PASS' if good else 'FAIL'))

for name in ('iqf_cum', 'iqf_pls', 'iqf_dsp', 'iqf_ms_dsp'):
    ref = build_model('iqformer').eval()
    mod = build_model(name).eval()
    mod.m.load_state_dict(ref.m.state_dict())
    for bs in (1, 4):
        x = torch.randn(bs, 2, 128) * 0.01
        with torch.no_grad():
            same = torch.allclose(ref(x), mod(x), atol=1e-5)
        assert same, (name, bs)
    mod.train()
    loss = torch.nn.functional.cross_entropy(mod(torch.randn(8, 2, 128) * 0.01), torch.randint(0, 11, (8,)))
    loss.backward()
    assert mod.dsp.film.weight.grad is not None and mod.dsp.film.weight.grad.abs().sum() > 0
    q = quantize_model(mod.eval(), 'all=W4A4')
    n_obs = calibrate(q, torch.randn(16, 2, 128) * 0.01, 'auto', 99.9, batch=16)
    covered = sum(isinstance(m, (QConv, QLinear)) for m in q.dsp.modules())
    assert covered >= 2 and q(torch.randn(1, 2, 128) * 0.01).shape == (1, 11)
    extra = sum(p.numel() for p in mod.parameters()) - sum(p.numel() for p in ref.parameters())
    print(f'{name:11s} identical to IQFormer at init (batch 1 and 4), gradients reach FiLM, '
          f'{covered} DSP layers quantized at W4A4, +{extra:,} parameters  PASS')
print('ALL PASS' if ok else 'SOME CHECKS FAILED')
