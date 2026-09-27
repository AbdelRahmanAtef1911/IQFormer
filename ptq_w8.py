"""Weight-only INT8 post-training quantization (per-channel), paired against
each model's own FP32 accuracy so that seed variation cancels."""
import copy, torch, numpy as np, pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from brevitas.nn import QuantConv1d, QuantConv2d, QuantLinear
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat as WQ
from dataset.dataset import RMLtest
from model.IQFormer import IQFormer

C = ['8PSK','BPSK','CPFSK','GFSK','PAM4','QAM16','QAM64','QPSK','AM-DSB','AM-SSB','WBFM']
CK = {'r1':'save_models/model_2016.10a_60_256_0.001_IQFormer/weight.pt',
      'r2':'save_models/model_2016.10a_60_256_0.001_IQFormer_r2/weight.pt',
      'r3':'save_models/model_2016.10a_60_256_0.001_IQFormer_r3/weight.pt'}
dev = 'cuda:0'

def swap(module):
    """Replace every Conv1d/Conv2d/Linear with an INT8 weight-quantized version."""
    for name, ch in module.named_children():
        if isinstance(ch, nn.Conv1d):
            q = QuantConv1d(ch.in_channels, ch.out_channels, ch.kernel_size,
                            stride=ch.stride, padding=ch.padding, dilation=ch.dilation,
                            groups=ch.groups, bias=ch.bias is not None, weight_quant=WQ)
        elif isinstance(ch, nn.Conv2d):
            q = QuantConv2d(ch.in_channels, ch.out_channels, ch.kernel_size,
                            stride=ch.stride, padding=ch.padding, dilation=ch.dilation,
                            groups=ch.groups, bias=ch.bias is not None, weight_quant=WQ)
        elif isinstance(ch, nn.Linear):
            q = QuantLinear(ch.in_features, ch.out_features,
                            bias=ch.bias is not None, weight_quant=WQ)
        else:
            swap(ch); continue
        q.load_state_dict(ch.state_dict(), strict=False)
        setattr(module, name, q)

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

@torch.no_grad()
def evaluate(m):
    P, T, S = [], [], []
    for iq, st, sn, lb in loader:
        P.append(m(iq.to(dev), st.to(dev)).argmax(1).cpu().numpy())
        T.append(lb.numpy()); S.append(np.asarray(sn))
    P, T, S = np.concatenate(P), np.concatenate(T), np.concatenate(S)
    acc = pd.DataFrame([{'SNR': s, 'accuracy': (P[S==s]==T[S==s]).mean()}
                        for s in sorted(np.unique(S))])
    from sklearn.metrics import f1_score, confusion_matrix
    hi = S >= 0
    cm = confusion_matrix(T[hi], P[hi], labels=range(11), normalize='true')
    return acc, (P==T).mean(), f1_score(T, P, average='macro'), cm[10, 8]*100

rows = []
for tag, ck in CK.items():
    fp = build(ck)
    n_before = sum(isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)) for m in fp.modules())
    q = copy.deepcopy(fp); swap(q); q = q.to(dev).eval()
    n_after = sum(isinstance(m, (QuantConv1d, QuantConv2d, QuantLinear)) for m in q.modules())
    af, accf, f1f, wbf = evaluate(fp)
    aq, accq, f1q, wbq = evaluate(q)
    aq.to_csv(f'w8_per_snr_{tag}.csv', index=False)
    rows.append(dict(tag=tag, accf=accf, accq=accq, f1f=f1f, f1q=f1q, wbf=wbf, wbq=wbq))
    print(f"{tag}: quantized {n_after}/{n_before} modules | "
          f"FP32 {accf*100:.2f}%  W8 {accq*100:.2f}%  delta {(accq-accf)*100:+.2f} pp")

d = pd.DataFrame(rows)
def s(a, b, label, unit='pp'):
    dd = (d[b] - d[a]) * (100 if unit == 'pp' else 1)
    print(f"{label:<26}{(d[a].mean()*100 if unit=='pp' else d[a].mean()):>8.2f}"
          f"{(d[b].mean()*100 if unit=='pp' else d[b].mean()):>9.2f}"
          f"{dd.mean():>+9.3f} +/- {dd.std(ddof=1):.3f}")

print(f"\n{'metric':<26}{'FP32':>8}{'W8':>9}{'change':>9}")
s('accf', 'accq', 'overall accuracy (%)')
s('f1f', 'f1q', 'macro F1', unit='raw')
s('wbf', 'wbq', 'WBFM -> AM-DSB (%)', unit='raw')
dm = ((d.accq - d.accf) * 100)
print(f"\nmeasurement floor (2 sigma) = 0.62 pp")
print("verdict:", "SIGNIFICANT loss" if abs(dm.mean()) > 0.62 else "below the noise floor")
d.to_csv('w8_summary.csv', index=False)
