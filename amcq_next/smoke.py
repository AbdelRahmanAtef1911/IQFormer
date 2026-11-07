"""Checks before any long run (CPU is enough, no dataset needed):  python -m amcq_next.smoke

1. 'iqfx' equals the toolkit's IQFormer (same parameters, same outputs with the same weights).
2. Every variant builds, runs with batch 1 and 4, and reports its size; the oracle refuses to run without SNR.
3. The complex STFT has the expected shape; a phase rotation leaves the window's power unchanged.
4. Training-only parts (SNR loss, InfoNCE) reach their layers through the gradient.
5. The new layers are wrapped by the 4-bit quantizer (W4A4 coverage), so the winner can go through QAT.
"""
import math
import torch
import amcq_next  # noqa: F401
from amcq import setup_repo
from amcq_arch.models import build_model
from amcq_next.models import complex_stft4, snr_bin
from amcq_next.train import phase_rotate, info_nce

VARIANTS = ['iqfx', 'iqfx_snr', 'iqfx_oracle', 'iqfx_c2d', 'iqfx_slim', 'iqfx_ctr', 'iqfx_snr_c2d', 'iqfx_slim_snr']


def check(name, ok, detail=''):
    print(f'{name:58s} {"PASS" if ok else "FAIL"}  {detail}')
    return ok


def main():
    setup_repo(None)
    torch.manual_seed(0)
    good = True
    x4, x1 = torch.randn(4, 2, 128) * 0.01, torch.randn(1, 2, 128) * 0.01
    s4 = torch.tensor([-20, -6, 0, 18])

    ref = build_model('iqformer').eval()
    base = build_model('iqfx').eval()
    base.m.load_state_dict(ref.m.state_dict())
    n_ref = sum(p.numel() for p in ref.parameters())
    n_b = sum(p.numel() for p in base.parameters())
    with torch.no_grad():
        diff = (ref(x4) - base(x4)).abs().max().item()
    good &= check('iqfx == toolkit IQFormer (parameters, outputs)', n_ref == n_b and diff < 1e-5,
                  f'{n_b:,} params, max diff {diff:.1e}')

    for v in VARIANTS:
        m = build_model(v).eval()
        n = sum(p.numel() for p in m.parameters())
        try:
            with torch.no_grad():
                o4 = m(x4, snr=s4) if 'oracle' in v else m(x4)
                o1 = m(x1, snr=s4[:1]) if 'oracle' in v else m(x1)
            ok = o4.shape == (4, 11) and o1.shape == (1, 11)
        except Exception as e:                                     # noqa: BLE001
            ok, o4 = False, e
        good &= check(f'{v}: batch 4 and 1', ok, f'{n:,} params ({n - n_ref:+,} vs IQFormer)')
    try:
        build_model('iqfx_oracle')(x4)
        good &= check('oracle refuses to run without SNR', False)
    except RuntimeError:
        good &= check('oracle refuses to run without SNR', True)

    s = complex_stft4(x4)
    good &= check('complex STFT shape (B,4,32,128)', tuple(s.shape) == (4, 4, 32, 128))
    xr = phase_rotate(x4)
    pw = ((x4 ** 2).sum(1) - (xr ** 2).sum(1)).abs().max().item()
    good &= check('phase rotation keeps |z|^2 per sample', pw < 1e-9, f'max diff {pw:.1e}')
    good &= check('SNR bins: -20,-6,0,18 dB -> 0,2,4,7', snr_bin(s4).tolist() == [0, 2, 4, 7])

    m = build_model('iqfx_snr_ctr').train()
    out, aux = m(torch.randn(8, 2, 128) * 0.01, return_aux=True)
    loss = torch.nn.functional.cross_entropy(aux['snr_logits'], snr_bin(torch.full((8,), -4))) \
        + info_nce(aux['z_iq'], aux['z_tf'], 0.07)
    loss.backward()
    g1 = m.snr_branch.mlp[-1].weight.grad.abs().sum().item()
    g2 = m.proj_tf[-1].weight.grad.abs().sum().item()
    good &= check('SNR loss and InfoNCE reach their layers', g1 > 0 and g2 > 0, f'grad {g1:.2e}, {g2:.2e}')
    good &= check('InfoNCE of identical views < of random views',
                  info_nce(aux['z_iq'], aux['z_iq'], 0.07).item() < info_nce(aux['z_iq'], aux['z_tf'], 0.07).item())

    try:
        from amcq.quant import quantize_model, QConv, QLinear
        for v in ('iqfx_snr', 'iqfx_c2d'):
            mm = build_model(v).eval()
            q = quantize_model(mm, 'all=W4A4')
            names = [n for n, mod in q.named_modules() if isinstance(mod, (QConv, QLinear))]
            new = [n for n in names if 'snr_branch' in n or 'cls_head' in n or 'patch_embedSTFT' in n]
            good &= check(f'{v}: new layers quantized at W4A4', len(new) > 0, f'{len(new)} new layers wrapped')
    except Exception as e:                                         # noqa: BLE001
        good &= check('quantizer coverage', False, repr(e))
    print('ALL PASS' if good else 'SOME CHECKS FAILED - do not start the long runs')


if __name__ == '__main__':
    main()
