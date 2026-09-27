"""Validates that quantization is actually happening, and sweeps bit width.

Three checks:
  1. quantized weights take at most 2^b distinct values per channel
  2. calibration produced real activation scales
  3. control: lowering bit width must degrade accuracy
"""
import copy, torch, numpy as np, pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from brevitas.nn import QuantConv1d, QuantConv2d, QuantLinear
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat, Int8ActPerTensorFloat
from brevitas.graph.calibrate import calibration_mode
from dataset.dataset import RMLgeneral, RMLtest
from model.IQFormer import IQFormer

C = ['8PSK','BPSK','CPFSK','GFSK','PAM4','QAM16','QAM64','QPSK','AM-DSB','AM-SSB','WBFM']
CK = {'r1':'save_models/model_2016.10a_60_256_0.001_IQFormer/weight.pt',
      'r2':'save_models/model_2016.10a_60_256_0.001_IQFormer_r2/weight.pt',
      'r3':'save_models/model_2016.10a_60_256_0.001_IQFormer_r3/weight.pt'}
dev = 'cuda:0'

data = pd.read_pickle('./dataset/RML2016.10a_dict.pkl')
tr, te = [[], [], []], [[], [], []]
for (lb, s), sm in data.items():
    if lb not in C: continue
    l = np.full(len(sm), C.index(lb)); n = np.full(len(sm), s)
    X, x, Y, y, S_tr, S_te = train_test_split(sm, l, n, test_size=0.2, random_state=233, stratify=l)
    t_, v_, tl, vl, S_t, S_v = train_test_split(X, Y, S_tr, test_size=0.25, random_state=233, stratify=Y)
    tr[0].extend(t_); tr[1].extend(tl); tr[2].extend(S_t)
    te[0].extend(x);  te[1].extend(y);  te[2].extend(S_te)
train_ds = RMLgeneral(np.array(tr[0]), np.array(tr[1]), np.array(tr[2]))
loader = DataLoader(RMLtest(np.array(te[0]), np.array(te[1]), np.array(te[2])), batch_size=400)
rng = np.random.default_rng(0)
calib = DataLoader(Subset(train_ds, rng.choice(len(train_ds), 1024, replace=False)), batch_size=256)

def build(ck):
    m = IQFormer([2,3,2], embed_dims=[64,64,64], mlp_ratios=4, act_layer=nn.GELU,
                 num_classes=11, down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0.2, drop_path_rate=0., use_layer_scale=False,
                 layer_scale_init_value=1e-5, fork_feat=False, vit_num=1).to(dev)
    m.load_state_dict(torch.load(ck, map_location=dev)); return m.eval()

def make_quants(bits):
    class WQ(Int8WeightPerChannelFloat): bit_width = bits
    class AQ(Int8ActPerTensorFloat):     bit_width = bits
    return WQ, AQ

def swap(module, WQ, AQ, act=True):
    kw = dict(weight_quant=WQ)
    if act: kw['input_quant'] = AQ
    for name, ch in module.named_children():
        if isinstance(ch, nn.Conv1d):
            q = QuantConv1d(ch.in_channels, ch.out_channels, ch.kernel_size, stride=ch.stride,
                            padding=ch.padding, dilation=ch.dilation, groups=ch.groups,
                            bias=ch.bias is not None, **kw)
        elif isinstance(ch, nn.Conv2d):
            q = QuantConv2d(ch.in_channels, ch.out_channels, ch.kernel_size, stride=ch.stride,
                            padding=ch.padding, dilation=ch.dilation, groups=ch.groups,
                            bias=ch.bias is not None, **kw)
        elif isinstance(ch, nn.Linear):
            q = QuantLinear(ch.in_features, ch.out_features, bias=ch.bias is not None, **kw)
        else:
            swap(ch, WQ, AQ, act); continue
        q.load_state_dict(ch.state_dict(), strict=False)
        setattr(module, name, q)

@torch.no_grad()
def calibrate(m):
    with calibration_mode(m):
        for iq, st, _, _ in calib: m(iq.to(dev), st.to(dev))
    return m

@torch.no_grad()
def acc(m):
    P, T = [], []
    for iq, st, sn, lb in loader:
        P.append(m(iq.to(dev), st.to(dev)).argmax(1).cpu().numpy()); T.append(lb.numpy())
    P, T = np.concatenate(P), np.concatenate(T)
    return (P == T).mean()

# ---------- CHECK 1 & 2 on r1 at 8 bits ----------
print("=" * 62)
print("CHECK 1 & 2 — is quantization actually applied?")
print("=" * 62)
fp = build(CK['r1'])
WQ, AQ = make_quants(8)
q8 = copy.deepcopy(fp); swap(q8, WQ, AQ); q8 = q8.to(dev).eval(); calibrate(q8)

fp_layers = [(n, m) for n, m in fp.named_modules() if isinstance(m, nn.Conv1d)]
q_layers  = [(n, m) for n, m in q8.named_modules() if isinstance(m, QuantConv1d)]
n0, m0 = fp_layers[2]; n1, m1 = q_layers[2]
w_fp = m0.weight.detach().flatten()
w_q  = m1.quant_weight().value.detach().flatten()
print(f"layer inspected            : {n1}")
print(f"distinct values, FP32      : {len(torch.unique(w_fp)):>6}  of {w_fp.numel()} weights")
print(f"distinct values, INT8      : {len(torch.unique(w_q)):>6}  (must be <= 256 per channel)")
print(f"max |w_int8 - w_fp32|      : {(w_q - w_fp).abs().max().item():.6e}")
print(f"weights actually changed   : {'YES' if not torch.allclose(w_q, w_fp) else 'NO  <-- PROBLEM'}")

scales = [m.input_quant.scale().item() for _, m in q_layers[:6]
          if hasattr(m, 'input_quant') and m.input_quant is not None]
print(f"\nactivation scales (first 6): {[f'{s:.4g}' for s in scales]}")
print(f"calibration produced scales: {'YES' if scales and all(s > 0 for s in scales) else 'NO  <-- PROBLEM'}")

# ---------- CHECK 3: bit-width control sweep ----------
print("\n" + "=" * 62)
print("CHECK 3 — control: does lowering bit width degrade accuracy?")
print("=" * 62)
rows = []
for tag, ck in CK.items():
    fp = build(ck); a_fp = acc(fp)
    r = {'tag': tag, 'fp32': a_fp}
    for b in [8, 6, 4, 2]:
        WQ, AQ = make_quants(b)
        m = copy.deepcopy(fp); swap(m, WQ, AQ); m = m.to(dev).eval(); calibrate(m)
        r[f'w{b}a{b}'] = acc(m)
    rows.append(r)
    print(f"{tag}: FP32 {a_fp*100:.2f}%  " +
          "  ".join(f"W{b}A{b} {r[f'w{b}a{b}']*100:.2f}%" for b in [8, 6, 4, 2]))

d = pd.DataFrame(rows); d.to_csv('bitwidth_sweep.csv', index=False)
print(f"\n{'precision':<12}{'accuracy':>10}{'std':>8}{'vs FP32':>12}")
print(f"{'FP32':<12}{d.fp32.mean()*100:>9.2f}%{d.fp32.std(ddof=1)*100:>8.2f}{'—':>12}")
for b in [8, 6, 4, 2]:
    c = f'w{b}a{b}'; dd = (d[c] - d.fp32) * 100
    print(f"{'W'+str(b)+'A'+str(b):<12}{d[c].mean()*100:>9.2f}%{d[c].std(ddof=1)*100:>8.2f}"
          f"{dd.mean():>+11.2f}")

drop2 = (d.w2a2.mean() - d.fp32.mean()) * 100
print(f"\nVERDICT: quantization machinery is "
      f"{'WORKING — low precision degrades as expected' if drop2 < -1 else 'SUSPECT — 2-bit shows no degradation'}")
