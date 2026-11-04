"""Run from repository root: python amcq/arch_smoke.py"""
import os
import sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amcq import setup_repo
setup_repo(None)
from amcq_arch.models import build_model
from amcq.quant import quantize_model, calibrate, QConv, QLinear

for name in ('iqf_ms', 'iqf_ms_gate', 'iqf_ms_snr'):
    original = build_model('iqformer').eval()
    edited = build_model(name).eval()
    edited.m.load_state_dict(original.m.state_dict())
    for batch in (1, 4):
        x = torch.randn(batch, 2, 128)
        with torch.no_grad():
            a = original(x)
            b, snr = edited(x, return_snr=True)
        assert b.shape == (batch, 11)
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), (name, batch, (a-b).abs().max().item())
        assert (snr is None) == (name != 'iqf_ms_snr')
    x = torch.randn(4, 2, 128)
    mod, snr = edited.train()(x, return_snr=True)
    loss = torch.nn.functional.cross_entropy(mod, torch.randint(0, 11, (4,)))
    if snr is not None:
        loss += 0.1 * torch.nn.functional.cross_entropy(snr, torch.randint(0, 5, (4,)))
    loss.backward()
    assert edited.ms_adapter.proj.weight.grad is not None
    q = quantize_model(edited.eval(), 'all=W4A4')
    count = calibrate(q, torch.randn(4, 2, 128), 'auto', 99.9, batch=4)
    assert count > 0 and any(isinstance(m, (QConv, QLinear)) for m in q.ms_adapter.modules())
    assert q(torch.randn(1, 2, 128)).shape == (1, 11)
    print(name, 'PASS', 'params', sum(p.numel() for p in edited.parameters()), 'activation observers', count)
