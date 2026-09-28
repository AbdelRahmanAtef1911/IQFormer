"""W8A8 post-training quantization (proposal 3.3 item 1).

Per-channel INT8 weights + per-tensor INT8 activations, calibrated on training
data. Compared paired against each model's own FP32 accuracy, and against the
weight-only result, so the activation contribution is isolated.
"""
import copy, torch, numpy as np, pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from brevitas.nn import QuantConv1d, QuantConv2d, QuantLinear
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat as WQ
from brevitas.quant.scaled_int import Int8ActPerTensorFloat as AQ
from dataset.dataset import RMLgeneral, RMLtest
from model.IQFormer import IQFormer

try:
    from brevitas.graph.calibrate import calibration_mode
except ImportError as e:
    raise SystemExit(f"calibration_mode import failed: {e}\n"
                     "Run: python -c 'import brevitas.graph.calibrate as m; print(dir(m))'")

C = ['8PSK','BPSK','CPFSK','GFSK','PAM4','QAM16','QAM64','QPSK','AM-DSB','AM-SSB','WBFM']
CK = {'r1':'save_models/model_2016.10a_60_256_0.001_IQFormer/weight.pt',
      'r2':'save_models/model_2016.10a_60_256_0.001_IQFormer_r2/weight.pt',
      'r3':'save_models/model_2016.10a_60_256_0.001_IQFormer_r3/weight.pt'}
dev = 'cuda:0'
N_CALIB = 1024          # training frames used to estimate activation ranges

# ---------------- data: rebuild train and test splits exactly as main.py ----------------
data = pd.read_pickle('./dataset/RML2016.10a_dict.pkl')
tr, te = [[], [], []], [[], [], []]
for (lb, s), sm in data.items():
    if lb not in C: continue
    l = np.full(len(sm), C.index(lb)); n = np.full(len(sm), s)
    X, x, Y, y, S_tr, S_te = train_test_split(sm, l, n, test_size=0.2,
                                              random_state=233, stratify=l)
    trn, val, trl, vl, S_t, S_v = train_test_split(X, Y, S_tr, test_size=0.25,
                                                   random_state=233, stratify=Y)
    tr[0].extend(trn); tr[1].extend(trl); tr[2].extend(S_t)
    te[0].extend(x);   te[1].extend(y);   te[2].extend(S_te)

train_ds = RMLgeneral(np.array(tr[0]), np.array(tr[1]), np.array(tr[2]))
test_ds  = RMLtest(np.array(te[0]), np.array(te[1]), np.array(te[2]))
print(f"train {len(train_ds)}  test {len(test_ds)}")

rng = np.random.default_rng(0)
calib_idx = rng.choice(len(train_ds), N_CALIB, replace=False)
calib_loader = DataLoader(Subset(train_ds, calib_idx), batch_size=256, shuffle=False)
loader = DataLoader(test_ds, batch_size=400, shuffle=False)

# ---------------- model helpers ----------------
def build(ck):
    m = IQFormer([2,3,2], embed_dims=[64,64,64], mlp_ratios=4, act_layer=nn.GELU,
                 num_classes=11, down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0.2, drop_path_rate=0., use_layer_scale=False,
                 layer_scale_init_value=1e-5, fork_feat=False, vit_num=1).to(dev)
    m.load_state_dict(torch.load(ck, map_location=dev)); return m.eval()

def swap(module, act=False):
    """Replace Conv1d/Conv2d/Linear with quantized versions.
    act=False -> weights only.  act=True -> weights + per-tensor input activations."""
    kw = dict(weight_quant=WQ)
    if act: kw['input_quant'] = AQ
    for name, ch in module.named_children():
        if isinstance(ch, nn.Conv1d):
            q = QuantConv1d(ch.in_channels, ch.out_channels, ch.kernel_size,
                            stride=ch.stride, padding=ch.padding, dilation=ch.dilation,
                            groups=ch.groups, bias=ch.bias is not None, **kw)
        elif isinstance(ch, nn.Conv2d):
            q = QuantConv2d(ch.in_channels, ch.out_channels, ch.kernel_size,
                            stride=ch.stride, padding=ch.padding, dilation=ch.dilation,
                            groups=ch.groups, bias=ch.bias is not None, **kw)
        elif isinstance(ch, nn.Linear):
            q = QuantLinear(ch.in_features, ch.out_features,
                            bias=ch.bias is not None, **kw)
        else:
            swap(ch, act); continue
        q.load_state_dict(ch.state_dict(), strict=False)
        setattr(module, name, q)

@torch.no_grad()
def calibrate(m):
    """One pass over training frames to estimate activation ranges."""
    with calibration_mode(m):
        for iq, st, _, _ in calib_loader:
            m(iq.to(dev), st.to(dev))
    return m

@torch.no_grad()
def evaluate(m):
    P, T, S = [], [], []
    for iq, st, sn, lb in loader:
        P.append(m(iq.to(dev), st.to(dev)).argmax(1).cpu().numpy())
        T.append(lb.numpy()); S.append(np.asarray(sn))
    P, T, S = np.concatenate(P), np.concatenate(T), np.concatenate(S)
    per = pd.DataFrame([{'SNR': s, 'accuracy': (P[S==s]==T[S==s]).mean()}
                        for s in sorted(np.unique(S))])
    hi = S >= 0
    cm = confusion_matrix(T[hi], P[hi], labels=range(11), normalize='true')
    band = per[(per.SNR >= -10) & (per.SNR <= 2)].accuracy.mean()
    return per, (P==T).mean(), f1_score(T, P, average='macro'), cm[10,8]*100, band

# ---------------- run ----------------
rows = []
for tag, ck in CK.items():
    fp = build(ck)

    qw = copy.deepcopy(fp); swap(qw, act=False); qw = qw.to(dev).eval()

    qa = copy.deepcopy(fp); swap(qa, act=True);  qa = qa.to(dev).eval()
    calibrate(qa)
    n_act = sum(1 for m in qa.modules()
                if isinstance(m, (QuantConv1d, QuantConv2d, QuantLinear)))

    pf, af, f1f, wf, bf = evaluate(fp)
    pw, aw, f1w, ww, bw = evaluate(qw)
    pa, aa, f1a, wa, ba = evaluate(qa)
    pa.to_csv(f'w8a8_per_snr_{tag}.csv', index=False)

    rows.append(dict(tag=tag, fp=af, w8=aw, w8a8=aa, f1f=f1f, f1w=f1w, f1a=f1a,
                     wf=wf, ww=ww, wa=wa, bf=bf, bw=bw, ba=ba))
    print(f"{tag}: {n_act} quantized modules | FP32 {af*100:.2f}%  "
          f"W8 {aw*100:.2f}%  W8A8 {aa*100:.2f}%  "
          f"| W8A8-FP32 {(aa-af)*100:+.2f} pp")

d = pd.DataFrame(rows)
d.to_csv('w8a8_summary.csv', index=False)

def line(label, a, b, c, scale=100, fmt='.2f'):
    va, vb, vc = d[a].mean()*scale, d[b].mean()*scale, d[c].mean()*scale
    dt = ((d[c]-d[a])*scale)
    print(f"{label:<24}{va:>9{fmt}}{vb:>9{fmt}}{vc:>9{fmt}}"
          f"{dt.mean():>+10.3f} +/- {dt.std(ddof=1):.3f}")

print(f"\n{'metric':<24}{'FP32':>9}{'W8':>9}{'W8A8':>9}{'W8A8 - FP32':>18}")
line('overall accuracy (%)', 'fp', 'w8', 'w8a8')
line('macro F1', 'f1f', 'f1w', 'f1a', scale=1, fmt='.4f')
line('transition band (%)',  'bf', 'bw', 'ba')
line('WBFM -> AM-DSB (%)',   'wf', 'ww', 'wa', scale=1)

dt = ((d.w8a8 - d.fp) * 100)
da = ((d.w8a8 - d.w8) * 100)
print(f"\nmeasurement floor (2 sigma)      = 0.62 pp")
print(f"total INT8 cost (W8A8 - FP32)    = {dt.mean():+.3f} +/- {dt.std(ddof=1):.3f} pp")
print(f"  of which weights               = {((d.w8-d.fp)*100).mean():+.3f} pp")
print(f"  of which activations           = {da.mean():+.3f} +/- {da.std(ddof=1):.3f} pp")
print("\nverdict:", "SIGNIFICANT loss" if abs(dt.mean()) > 0.62 else "below the noise floor")
