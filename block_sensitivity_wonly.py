"""Control for block_sensitivity.py: WEIGHT-ONLY quantization for every group,
so CONV/ATTENTION/HEAD receive the same treatment the LSTM necessarily gets.
Also reports each group's share of parameters, since a larger group accumulates
more rounding error independently of block type."""
import copy, torch, numpy as np, pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset.dataset import RMLtest
from model.IQFormer import IQFormer

C = ['8PSK','BPSK','CPFSK','GFSK','PAM4','QAM16','QAM64','QPSK','AM-DSB','AM-SSB','WBFM']
CK = {'r1':'save_models/model_2016.10a_60_256_0.001_IQFormer/weight.pt',
      'r2':'save_models/model_2016.10a_60_256_0.001_IQFormer_r2/weight.pt',
      'r3':'save_models/model_2016.10a_60_256_0.001_IQFormer_r3/weight.pt'}
dev = 'cuda:0'; BITS = [8, 6, 4, 3]; GROUPS = ['CONV', 'LSTM', 'ATTENTION', 'HEAD']

data = pd.read_pickle('./dataset/RML2016.10a_dict.pkl')
te = [[], [], []]
for (lb, s), sm in data.items():
    if lb not in C: continue
    l = np.full(len(sm), C.index(lb)); n = np.full(len(sm), s)
    _, x, _, y, _, ss = train_test_split(sm, l, n, test_size=0.2, random_state=233, stratify=l)
    te[0].extend(x); te[1].extend(y); te[2].extend(ss)
loader = DataLoader(RMLtest(np.array(te[0]), np.array(te[1]), np.array(te[2])), batch_size=400)

def build(ck):
    m = IQFormer([2,3,2], embed_dims=[64,64,64], mlp_ratios=4, act_layer=nn.GELU,
                 num_classes=11, down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0.2, drop_path_rate=0., use_layer_scale=False,
                 layer_scale_init_value=1e-5, fork_feat=False, vit_num=1).to(dev)
    m.load_state_dict(torch.load(ck, map_location=dev)); return m.eval()

def group_of(path):
    if path.startswith('patch_LSTM'): return 'LSTM'
    if path.startswith('head'):       return 'HEAD'
    if '.attn.' in path:              return 'ATTENTION'
    return 'CONV'

def fq_(t, bits, dim=0):
    """Per-output-channel symmetric fake quantization, in place."""
    qmax = 2 ** (bits - 1) - 1
    flat = t.detach().reshape(t.shape[0], -1)
    s = (flat.abs().amax(dim=1) / qmax).clamp(min=1e-12)
    s = s.reshape([-1] + [1] * (t.dim() - 1))
    t.data.copy_((t.detach() / s).round().clamp(-qmax - 1, qmax) * s)

@torch.no_grad()
def acc(m):
    P, T = [], []
    for iq, st, sn, lb in loader:
        P.append(m(iq.to(dev), st.to(dev)).argmax(1).cpu().numpy()); T.append(lb.numpy())
    P, T = np.concatenate(P), np.concatenate(T)
    return (P == T).mean()

ref = build(CK['r1'])
share = {g: 0 for g in GROUPS}
for n_, p in ref.named_parameters():
    if 'weight' in n_ and p.dim() >= 2:
        share[group_of(n_)] += p.numel()
tot = sum(share.values())
print("quantizable weight parameters by group:")
for g in GROUPS:
    print(f"  {g:<11}{share[g]:>8,}  ({share[g]/tot*100:5.1f}%)")

rows = []
for tag, ck in CK.items():
    fp = build(ck); a0 = acc(fp)
    for g in GROUPS:
        for b in BITS:
            m = copy.deepcopy(fp)
            for n_, p in m.named_parameters():
                if 'weight' in n_ and p.dim() >= 2 and group_of(n_) == g:
                    fq_(p, b)
            a = acc(m.eval())
            rows.append(dict(run=tag, group=g, bits=b, delta_pp=(a - a0) * 100,
                             n_params=share[g]))
            print(f"{tag}  {g:<11}{b}-bit  {a*100:6.2f}%  ({(a-a0)*100:+7.2f} pp)")

d = pd.DataFrame(rows); d.to_csv('block_sensitivity_weightonly.csv', index=False)
piv = d.pivot_table(index='group', columns='bits', values='delta_pp', aggfunc='mean').reindex(GROUPS)
sdv = d.pivot_table(index='group', columns='bits', values='delta_pp', aggfunc='std').reindex(GROUPS)

print("\n" + "=" * 76)
print("WEIGHT-ONLY SENSITIVITY — identical treatment for every group")
print("=" * 76)
print(f"{'group':<11}{'params':>9}" + "".join(f"{b}-bit".rjust(16) for b in BITS))
for g in GROUPS:
    line = f"{g:<11}{share[g]/tot*100:>8.1f}%"
    for b in BITS:
        line += f"{piv.loc[g,b]:>+10.2f}±{sdv.loc[g,b]:>4.2f}"
    print(line)

print("\ndamage per 1% of parameters quantized (4-bit), normalising for group size:")
for g in GROUPS:
    pct = share[g] / tot * 100
    print(f"  {g:<11}{piv.loc[g,4]/pct:>+8.3f} pp per 1% of params")
