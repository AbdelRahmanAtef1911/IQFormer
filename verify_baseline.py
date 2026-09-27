"""Closes proposal gaps: block-level params/MACs (3.2), macro F1, model size,
and the confusion pairs named in 3.3."""
import os, numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from dataset.dataset import RMLtest
from model.IQFormer import IQFormer
from model.IQFormer import ConvEncoder_IQ, IQFormer_Encoder, Fusion, Embedding

C = ['8PSK','BPSK','CPFSK','GFSK','PAM4','QAM16','QAM64','QPSK','AM-DSB','AM-SSB','WBFM']
CK = {'r1':'save_models/model_2016.10a_60_256_0.001_IQFormer/weight.pt',
      'r2':'save_models/model_2016.10a_60_256_0.001_IQFormer_r2/weight.pt',
      'r3':'save_models/model_2016.10a_60_256_0.001_IQFormer_r3/weight.pt'}
dev = 'cuda:0'

def build(ck=None):
    m = IQFormer([2,3,2], embed_dims=[64,64,64], mlp_ratios=4, act_layer=nn.GELU,
                 num_classes=11, down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0.2, drop_path_rate=0., use_layer_scale=False,
                 layer_scale_init_value=1e-5, fork_feat=False, vit_num=1).to(dev)
    if ck: m.load_state_dict(torch.load(ck, map_location=dev))
    return m.eval()

# ---------- 3.2 block table ----------
model = build()
macs = {}
def hook(name):
    def f(mod, inp, out):
        if isinstance(mod, nn.Conv1d):
            n = mod.out_channels * out.shape[-1] * (mod.in_channels//mod.groups) * mod.kernel_size[0]
        elif isinstance(mod, nn.Conv2d):
            n = mod.out_channels * out.shape[-2]*out.shape[-1] * (mod.in_channels//mod.groups) * mod.kernel_size[0]*mod.kernel_size[1]
        elif isinstance(mod, nn.Linear):
            n = mod.in_features * mod.out_features * int(np.prod(out.shape[1:-1]))
        elif isinstance(mod, nn.LSTM):
            L = inp[0].shape[1]; D = 2 if mod.bidirectional else 1
            n = 0
            for layer in range(mod.num_layers):
                i_sz = mod.input_size if layer == 0 else mod.hidden_size*D
                n += 4*mod.hidden_size*(i_sz + mod.hidden_size)*L*D
        else: return
        macs[name] = macs.get(name, 0) + n
    return f

hs = [m.register_forward_hook(hook(n)) for n, m in model.named_modules()
      if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.LSTM))]
with torch.no_grad():
    model(torch.randn(2,2,128, device=dev), torch.randn(2,1,32,128, device=dev))
for h in hs: h.remove()

def group_of(path):
    if path.startswith(('BN','patch_embed')): return 'Input BN + I/Q and STFT stems'
    if path.startswith('fusion'):             return 'Fusion (two 1x1 convolutions)'
    if path.startswith('patch_LSTM'):         return 'Bidirectional LSTM, 2 layers'
    if path.startswith('network'):
        parts = path.split('.')
        blk = model.network[int(parts[1])]
        sub = blk[int(parts[2])] if parts[2].isdigit() else None
        if isinstance(sub, ConvEncoder_IQ):    return '4 convolutional encoder blocks'
        if isinstance(sub, IQFormer_Encoder):  return '3 IQFormer encoder blocks'
        return 'Downsampling embedding'
    return 'Output BN + classifier'

gp, gm = {}, {}
for n, p in model.named_parameters():
    owner = n.rsplit('.', 1)[0]
    gp[group_of(owner)] = gp.get(group_of(owner), 0) + p.numel()
for n, v in macs.items(): gm[group_of(n)] = gm.get(group_of(n), 0) + v

TOTP = sum(x.numel() for x in model.parameters()); TOTM = sum(gm.values())
PROP = {'Input BN + I/Q and STFT stems': (350, 0.04),
        'Fusion (two 1x1 convolutions)': (1664, 0.20),
        'Bidirectional LSTM, 2 layers': (41984, 5.2),
        '4 convolutional encoder blocks': (134144, 16.9),
        '3 IQFormer encoder blocks': (176064, 22.1),
        'Output BN + classifier': (843, 0.01)}

print("=== Proposal Table 3.2 verification ===")
print(f"{'Block':<34}{'params':>9}{'prop':>9}{'MACs(M)':>9}{'prop':>8}")
for k in PROP:
    pp, pm = PROP[k]
    print(f"{k:<34}{gp.get(k,0):>9,}{pp:>9,}{gm.get(k,0)/1e6:>9.2f}{pm:>8.2f}")
for k in gp:
    if k not in PROP: print(f"{k:<34}{gp[k]:>9,}{'-':>9}{gm.get(k,0)/1e6:>9.2f}{'-':>8}")
print(f"{'TOTAL':<34}{TOTP:>9,}{355049:>9,}{TOTM/1e6:>9.2f}{44.4:>8.2f}")

# ---------- 3.3 macro F1, model size, confusion pairs ----------
data = pd.read_pickle('./dataset/RML2016.10a_dict.pkl')
te = [[],[],[]]
for (lb,s), sm in data.items():
    if lb not in C: continue
    l = np.full(len(sm), C.index(lb)); nn_ = np.full(len(sm), s)
    _, x, _, y, _, ss = train_test_split(sm, l, nn_, test_size=0.2, random_state=233, stratify=l)
    te[0].extend(x); te[1].extend(y); te[2].extend(ss)
loader = DataLoader(RMLtest(np.array(te[0]),np.array(te[1]),np.array(te[2])), batch_size=400)

size_mb = os.path.getsize(CK['r1'])/1e6
f1s, pairs = [], []
for tag, ck in CK.items():
    m = build(ck); P,T,S = [],[],[]
    with torch.no_grad():
        for iq, st, sn, lb in loader:
            P.append(m(iq.to(dev), st.to(dev)).argmax(1).cpu().numpy())
            T.append(lb.numpy()); S.append(np.asarray(sn))
    P,T,S = np.concatenate(P), np.concatenate(T), np.concatenate(S)
    f1s.append(f1_score(T, P, average='macro'))
    hi = S >= 0
    cm = confusion_matrix(T[hi], P[hi], labels=range(11), normalize='true')
    g = lambda a,b: cm[C.index(a), C.index(b)]*100
    pairs.append([g('QAM16','QAM64'), g('QAM64','QAM16'), g('WBFM','AM-DSB'), g('AM-DSB','WBFM'),
                  g('8PSK','QPSK'), g('QPSK','8PSK')])

f1s = np.array(f1s); pr = np.array(pairs)
print(f"\n=== Proposal 3.3 metrics ===")
print(f"macro F1 (mean of 3)   : {f1s.mean():.4f} +/- {f1s.std(ddof=1):.4f}")
print(f"stored model size FP32 : {size_mb:.2f} MB")
print(f"projected INT8 size    : {size_mb/4:.2f} MB  (proposal states ~0.36 MB)")
print(f"\nconfusion pairs at SNR >= 0 dB (% of true class sent to the other):")
for i, n in enumerate(['16QAM -> 64QAM','64QAM -> 16QAM','WBFM -> AM-DSB','AM-DSB -> WBFM',
                       '8PSK -> QPSK','QPSK -> 8PSK']):
    print(f"  {n:<16}{pr[:,i].mean():>6.2f} +/- {pr[:,i].std(ddof=1):.2f} pp")

pd.DataFrame({'metric':['macro_f1','f1_std','size_mb_fp32'],
              'value':[f1s.mean(), f1s.std(ddof=1), size_mb]}).to_csv('fp32_metrics.csv', index=False)
