"""Block-wise quantization sensitivity (proposal 3.3 item 2).

Quantizes ONE block group at a time at 8/6/4 bits, leaving the rest in FP32,
to rank the architecture's components by how much damage each causes.

CONV and ATTENTION groups: weights (per-channel) + activations (per-tensor, calibrated).
LSTM group: weight-only fake quantization applied directly to the weight tensors,
because nn.LSTM is not reachable by module substitution.
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
from model.IQFormer import IQFormer_Encoder, EfficientAdditiveAttnetion

C = ['8PSK','BPSK','CPFSK','GFSK','PAM4','QAM16','QAM64','QPSK','AM-DSB','AM-SSB','WBFM']
CK = {'r1':'save_models/model_2016.10a_60_256_0.001_IQFormer/weight.pt',
      'r2':'save_models/model_2016.10a_60_256_0.001_IQFormer_r2/weight.pt',
      'r3':'save_models/model_2016.10a_60_256_0.001_IQFormer_r3/weight.pt'}
dev = 'cuda:0'
BITS = [8, 6, 4]
GROUPS = ['CONV', 'LSTM', 'ATTENTION', 'HEAD', 'ALL']

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

def group_of(path, model):
    """Assign a module path to an architectural group."""
    if path.startswith('patch_LSTM'): return 'LSTM'
    if path in ('head',) or path.startswith('head'): return 'HEAD'
    if '.attn.' in path or path.endswith('.attn'): return 'ATTENTION'
    return 'CONV'

def fake_quant_(t, bits):
    """Per-row symmetric fake quantization, in place."""
    qmax = 2 ** (bits - 1) - 1
    s = (t.detach().abs().amax(dim=-1, keepdim=True) / qmax).clamp(min=1e-12)
    t.data.copy_((t.detach() / s).round().clamp(-qmax - 1, qmax) * s)

def apply_group(model, group, bits):
    """Quantize only the named group; everything else stays FP32."""
    class WQ(Int8WeightPerChannelFloat): bit_width = bits
    class AQ(Int8ActPerTensorFloat):     bit_width = bits

    if group in ('LSTM', 'ALL'):
        for name, p in model.patch_LSTM.named_parameters():
            if 'weight' in name: fake_quant_(p, bits)

    def rec(mod, prefix=''):
        for name, ch in list(mod.named_children()):
            path = f'{prefix}{name}'
            if isinstance(ch, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                g = group_of(path, model)
                if group != 'ALL' and g != group:
                    continue
                if group == 'LSTM':
                    continue
                kw = dict(weight_quant=WQ, input_quant=AQ)
                if isinstance(ch, nn.Conv1d):
                    q = QuantConv1d(ch.in_channels, ch.out_channels, ch.kernel_size, stride=ch.stride,
                                    padding=ch.padding, dilation=ch.dilation, groups=ch.groups,
                                    bias=ch.bias is not None, **kw)
                elif isinstance(ch, nn.Conv2d):
                    q = QuantConv2d(ch.in_channels, ch.out_channels, ch.kernel_size, stride=ch.stride,
                                    padding=ch.padding, dilation=ch.dilation, groups=ch.groups,
                                    bias=ch.bias is not None, **kw)
                else:
                    q = QuantLinear(ch.in_features, ch.out_features, bias=ch.bias is not None, **kw)
                q.load_state_dict(ch.state_dict(), strict=False)
                setattr(mod, name, q)
            else:
                rec(ch, path + '.')
    rec(model)
    return model

@torch.no_grad()
def calibrate(m):
    has_q = any(isinstance(x, (QuantConv1d, QuantConv2d, QuantLinear)) for x in m.modules())
    if not has_q: return
    with calibration_mode(m):
        for iq, st, _, _ in calib: m(iq.to(dev), st.to(dev))

@torch.no_grad()
def acc(m):
    P, T = [], []
    for iq, st, sn, lb in loader:
        P.append(m(iq.to(dev), st.to(dev)).argmax(1).cpu().numpy()); T.append(lb.numpy())
    P, T = np.concatenate(P), np.concatenate(T)
    return (P == T).mean()

rows = []
for tag, ck in CK.items():
    fp = build(ck); a0 = acc(fp)
    n_q = {g: 0 for g in GROUPS}
    for g in GROUPS:
        for b in BITS:
            m = copy.deepcopy(fp)
            apply_group(m, g, b)
            m = m.to(dev).eval(); calibrate(m)
            a = acc(m)
            rows.append(dict(run=tag, group=g, bits=b, acc_fp32=a0, acc=a,
                             delta_pp=(a - a0) * 100))
            print(f"{tag}  {g:<9} {b}-bit  {a*100:6.2f}%  ({(a-a0)*100:+6.2f} pp)")

d = pd.DataFrame(rows); d.to_csv('block_sensitivity.csv', index=False)

print("\n" + "=" * 70)
print("BLOCK SENSITIVITY — accuracy change when only that group is quantized")
print("=" * 70)
piv = d.pivot_table(index='group', columns='bits', values='delta_pp', aggfunc='mean')
piv = piv.reindex(GROUPS)
sdv = d.pivot_table(index='group', columns='bits', values='delta_pp', aggfunc='std').reindex(GROUPS)
print(f"{'group':<11}" + "".join(f"{b}-bit".rjust(18) for b in BITS))
for g in GROUPS:
    line = f"{g:<11}"
    for b in BITS:
        line += f"{piv.loc[g,b]:>+11.2f} ±{sdv.loc[g,b]:>5.2f}"
    print(line)

print(f"\nmeasurement floor (2 sigma) = 0.62 pp")
r4 = piv[4].drop('ALL').sort_values()
print(f"\nranking by damage at 4 bits (most fragile first):")
for g, v in r4.items():
    print(f"  {g:<11}{v:+7.2f} pp   {'** MOST FRAGILE **' if g == r4.index[0] else ''}")
