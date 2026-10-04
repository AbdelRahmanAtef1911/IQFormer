"""Independent checks that amcq computes what it claims. Run from the repository root:

    python amcq/tests/verify_toolkit.py [--ckpt save_models/model_2016.10a_60_256_0.001_IQFormer/weight.pt]

Each check prints PASS or FAIL with the measured number; the table is also written to
results/verify_toolkit.txt for the report's appendix. The released code (dataset/dataset.py with
scipy STFT, model/IQFormer.py forward) is used as the reference wherever it can be.
"""
import argparse
import copy
import hashlib
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from amcq import setup_repo, default_data                     # noqa: E402

rows = []


def check(name, ok, detail):
    rows.append((name, 'PASS' if ok else 'FAIL', detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', default=None)
    p.add_argument('--data', default=None)
    p.add_argument('--ckpt', default='save_models/model_2016.10a_60_256_0.001_IQFormer/weight.pt')
    p.add_argument('--n', type=int, default=4000, help='test frames used for the model checks')
    a = p.parse_args()
    repo = setup_repo(a.repo)
    from scipy.signal import stft
    from model.IQFormer import IQFormer
    from amcq.data import load_rml2016a, stft_features, CLASSES_2016A
    from amcq.models import IQFormerNet
    from amcq.quant import quantize_model, calibrate, fold_bn, QLSTM, ActQuant, QConv, QLinear
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    def tf32(on):
        # NVIDIA GPUs run convolutions (and cuDNN's LSTM) in TF32 by default: 10-bit mantissa, relative
        # error ~5e-4 per operation. Equivalence checks must run in true FP32, otherwise any two code paths
        # that differ in the last bit of an input disagree at the TF32 level (~1e-3 on the logits).
        torch.backends.cudnn.allow_tf32 = on
        torch.backends.cuda.matmul.allow_tf32 = on

    # V1 split ------------------------------------------------------------------------------
    d = load_rml2016a(a.data or default_data(repo))
    sizes = (len(d['ytr']), len(d['yva']), len(d['yte']))
    per_cell = np.unique(np.stack([d['yte'], d['ste']], 1), axis=0, return_counts=True)[1]
    full = sizes[2] > 40000
    check('V1 split sizes', (sizes == (132000, 44000, 44000)) if full else True,
          f'train/val/test = {sizes}' + ('' if full else ' (small test data)'))
    check('V1 split balance', len(set(per_cell.tolist())) == 1,
          f'{len(per_cell)} (class, SNR) cells, each {per_cell[0]} test frames')
    h = hashlib.sha256(d['Xte'].tobytes() + d['yte'].tobytes() + d['ste'].tobytes()).hexdigest()[:16]
    check('V1 test-set fingerprint', True, f'sha256[:16] = {h} (same split => same fingerprint on any machine)')

    # V2 STFT -------------------------------------------------------------------------------
    # a random sample of test frames: the test set is stored key by key (one modulation and SNR after
    # another), so its first frames are not representative
    pick = np.random.default_rng(1).choice(len(d['yte']), min(a.n, len(d['yte'])), replace=False)
    X, ylab = d['Xte'][pick], d['yte'][pick]
    ref = np.stack([stft(x[0], 1.0, 'blackman', 31, 30, 128)[2][:32].real for x in X])
    mine = stft_features(torch.from_numpy(X).to(dev), 'real')[:, 0].cpu().numpy()
    err = np.abs(mine - ref).max() / np.abs(ref).max()
    check('V2 GPU STFT = released scipy STFT (real part, I channel)', err < 1e-4, f'relative max error {err:.2e} on {len(X)} frames')
    refc = []
    for x in X[:200]:
        Z = stft(x[0] + 1j * x[1], 1.0, 'blackman', 31, 30, 64, return_onesided=False)[2]
        refc.append(np.log1p(np.abs(np.fft.fftshift(Z, axes=0)))[16:48])
    minec = stft_features(torch.from_numpy(X[:200]).to(dev), 'cplx_logmag')[:, 0].cpu().numpy()
    errc = np.abs(minec - np.stack(refc)).max()
    check('V2 GPU STFT = cplxwide input (I+jQ, nfft 64, bins 16:48, log1p)', errc < 1e-4, f'max abs error {errc:.2e}')

    # V3 model ------------------------------------------------------------------------------
    orig = IQFormer([2, 3, 2], embed_dims=[64, 64, 64], mlp_ratios=4, act_layer=nn.GELU, num_classes=11,
                    down_patch_size=3, down_stride=2, down_pad=1, drop_rate=0.2, drop_path_rate=0.,
                    use_layer_scale=False, layer_scale_init_value=1e-5, fork_feat=False, vit_num=1).to(dev)
    orig.load_state_dict(torch.load(a.ckpt, map_location=dev)); orig.eval()
    net = IQFormerNet().to(dev).load_legacy(a.ckpt, dev).eval()
    xt = torch.from_numpy(X).to(dev)
    st = torch.tensor(np.expand_dims(ref, 1), dtype=torch.float32, device=dev)   # released preprocessing

    def both():
        with torch.no_grad():
            lo_ = torch.cat([orig(xt[i:i + 400], st[i:i + 400]) for i in range(0, len(xt), 400)])
            lm_ = torch.cat([net(xt[i:i + 400]) for i in range(0, len(xt), 400)])
        return lo_, lm_
    if dev == 'cuda':                                  # as the experiments run: GPU default (TF32 convolutions)
        tf32(True)
        lo, lm = both()
        agree = (lo.argmax(1) == lm.argmax(1)).float().mean().item()
        check('V3 GPU default (TF32): same predictions as the released forward', agree >= 0.999,
              f'same prediction on {100 * agree:.2f} % of {len(X)} frames; max |logit diff| '
              f'{(lo - lm).abs().max().item():.1e} = TF32 rounding level')
    tf32(False)                                        # every check below runs in true FP32
    lo, lm = both()
    diff = (lo - lm).abs().max().item()
    agree = (lo.argmax(1) == lm.argmax(1)).float().mean().item()
    check('V3 amcq model = released IQFormer forward (true FP32)', diff < 1e-3 and agree > 0.999,
          f'max |logit diff| {diff:.2e}, same prediction on {100 * agree:.2f} % of {len(X)} frames')
    try:
        with torch.no_grad():
            orig(xt[:1], st[:1])
        released_b1 = 'ran'
    except Exception as e:                                            # noqa: BLE001
        released_b1 = f'crashes ({type(e).__name__})'
    with torch.no_grad():
        one = net(xt[:1])
    check('V3 batch size 1', torch.allclose(one, lm[:1], atol=1e-4),
          f'released forward {released_b1}; amcq gives the same logits as in a batch (diff {(one - lm[:1]).abs().max().item():.1e})')
    n_params = sum(q.numel() for q in net.parameters())
    check('V3 parameter count', n_params == 355049, f'{n_params:,}')

    # V4 BN folding, wrappers, LSTM -----------------------------------------------------------
    f = copy.deepcopy(net); nf = fold_bn(f)
    f.m.patch_LSTM.flatten_parameters()
    with torch.no_grad():
        lf = torch.cat([f(xt[i:i + 400]) for i in range(0, len(xt), 400)])
    check('V4 BatchNorm folding is exact', (lf - lm).abs().max().item() < 1e-3,
          f'{nf} BatchNorms folded; max |logit diff| {(lf - lm).abs().max().item():.2e}')
    q32 = quantize_model(net, 'all=W32A32')
    with torch.no_grad():
        l32 = torch.cat([q32(xt[i:i + 400]) for i in range(0, len(xt), 400)])
    check('V4 quantization wrappers at full precision change nothing', (l32 - lm).abs().max().item() < 1e-3,
          f'max |logit diff| {(l32 - lm).abs().max().item():.2e} (BN folded + unrolled LSTM, no rounding)')
    feats = {}
    hook = net.m.patch_LSTM.register_forward_hook(lambda m, i, o: feats.__setitem__('in', i[0]))
    with torch.no_grad():
        net(xt[:64])
    hook.remove()
    ql = QLSTM(copy.deepcopy(net.m.patch_LSTM)).eval()       # same weights, unrolled step by step
    with torch.no_grad():
        dl = (ql(feats['in'])[0] - net.m.patch_LSTM(feats['in'])[0]).abs().max().item()
    check('V4 unrolled LSTM = nn.LSTM (on real fusion outputs)', dl < 1e-4, f'max |diff| {dl:.2e}')

    # V5 quantization really happens ------------------------------------------------------------
    rng = np.random.default_rng(0)
    cal = torch.from_numpy(d['Xtr'][rng.choice(len(d['ytr']), 1024, replace=False)]).to(dev)
    for bits in (8, 4):
        q = quantize_model(net, f'all=W{bits}A{bits}'); calibrate(q, cal)
        worst_w, worst_err = 0, 0.0
        for m in q.modules():
            if isinstance(m, (QConv, QLinear)) and m.wq.enabled:
                w = m.conv.weight if isinstance(m, QConv) else m.lin.weight
                wq = m.wq(w)
                worst_w = max(worst_w, max(len(torch.unique(wq[c])) for c in range(wq.shape[0])))
                worst_err = max(worst_err, ((wq - w).abs() / m.wq.scale.abs()).max().item())
        check(f'V5 W{bits}: weights on the integer grid', worst_w <= 2 ** bits - 1 and worst_err <= 0.5 + 1e-4,
              f'at most {worst_w} distinct values per output channel (limit {2 ** bits - 1}); '
              f'max rounding error {worst_err:.3f} step (limit 0.5)')
        seen = []
        hooks = [m.register_forward_hook(lambda mod, i, o: seen.append(len(torch.unique(o))))
                 for m in q.modules() if isinstance(m, ActQuant) and m.bits is not None]
        with torch.no_grad():
            q(xt[:256])
        for hk in hooks:
            hk.remove()
        check(f'V5 A{bits}: activations on the integer grid', max(seen) <= 2 ** bits,
              f'{len(seen)} quantizer calls (the LSTM state is quantized at every time step), at most {max(seen)} distinct values (limit {2 ** bits})')
    q2 = quantize_model(net, 'all=W2A2'); calibrate(q2, cal)
    with torch.no_grad():
        acc2 = (torch.cat([q2(xt[i:i + 400]) for i in range(0, len(xt), 400)]).argmax(1).cpu().numpy() == ylab).mean()
    acc_fp = (lm.argmax(1).cpu().numpy() == ylab).mean()
    chance = 1 / 11
    check('V5 control: 2 bits destroys the model', acc_fp - acc2 >= 0.75 * (acc_fp - chance),
          f'W2A2 accuracy {100 * acc2:.2f} % vs FP32 {100 * acc_fp:.2f} % on the same {len(ylab)} random test frames (chance 9.09 %)')

    # V6 determinism ----------------------------------------------------------------------------
    with torch.no_grad():
        again = torch.cat([net(xt[i:i + 400]) for i in range(0, len(xt), 400)])
    check('V6 evaluation is deterministic', torch.equal(again.argmax(1), lm.argmax(1)), 'same predictions on a second pass')

    os.makedirs('results', exist_ok=True)
    with open('results/verify_toolkit.txt', 'w') as fh:
        fh.write(f'checkpoint: {a.ckpt}\ndevice: {dev}\n\n')
        for r in rows:
            fh.write(f'[{r[1]}] {r[0]}: {r[2]}\n')
    n_fail = sum(r[1] == 'FAIL' for r in rows)
    print(f"\n{len(rows) - n_fail} of {len(rows)} checks passed -> results/verify_toolkit.txt")


if __name__ == '__main__':
    main()
