"""Reproduces Fig. 6 (per-class accuracy vs SNR) and Fig. 7(a) (confusion
matrices at -4 dB and 2 dB) for RadioML 2016.10a, averaged over three runs."""
import numpy as np, pandas as pd, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from dataset.dataset import RMLtest
from model.IQFormer import IQFormer

C = ['8PSK','BPSK','CPFSK','GFSK','PAM4','QAM16','QAM64','QPSK','AM-DSB','AM-SSB','WBFM']
TAGS = ['r1','r2','r3']
CK = {'r1':'save_models/model_2016.10a_60_256_0.001_IQFormer/weight.pt',
      'r2':'save_models/model_2016.10a_60_256_0.001_IQFormer_r2/weight.pt',
      'r3':'save_models/model_2016.10a_60_256_0.001_IQFormer_r3/weight.pt'}
dev = 'cuda:0'

# ---- Fig 6: per-class accuracy vs SNR (paper style) ----
pc = np.mean([pd.read_csv(f'perclass_{t}.csv')[C].values for t in TAGS], axis=0)
snr = pd.read_csv('perclass_r1.csv').SNR.values
mk = ['x','o','s','^','v','D','<','>','p','*','+']
fig, ax = plt.subplots(figsize=(6.5, 5))
for i, c in enumerate(C):
    ax.plot(snr, pc[:, i], marker=mk[i], ms=5, lw=1.2, label=c)
ax.set_xlabel('SNR(dB)'); ax.set_ylabel('Accuracy')
ax.set_xlim(-20, 20); ax.set_ylim(0, 1); ax.grid(alpha=0.25)
ax.set_title('RML2016.10a — reproduction (mean of 3 runs)')
ax.legend(fontsize=8, loc='lower right', ncol=1)
fig.tight_layout(); fig.savefig('fig6_repro_perclass.pdf', dpi=300)
fig.savefig('fig6_repro_perclass.png', dpi=150)

# ---- Fig 7(a): confusion matrices at -4 dB and 2 dB ----
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

cms = {-4: [], 2: []}
for ck in CK.values():
    m = build(ck); P, T, S = [], [], []
    with torch.no_grad():
        for iq, st, sn, lb in loader:
            P.append(m(iq.to(dev), st.to(dev)).argmax(1).cpu().numpy())
            T.append(lb.numpy()); S.append(np.asarray(sn))
    P, T, S = np.concatenate(P), np.concatenate(T), np.concatenate(S)
    for s in cms:
        k = S == s
        cms[s].append(confusion_matrix(T[k], P[k], labels=range(11), normalize='true'))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
for ax, s in zip(axes, [-4, 2]):
    cm = np.mean(cms[s], axis=0)
    ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
    for i in range(11):
        for j in range(11):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f'{cm[i,j]:.2f}', ha='center', va='center', fontsize=6.5,
                        color='white' if cm[i, j] > 0.5 else 'black')
    ax.set_xticks(range(11)); ax.set_xticklabels(C, rotation=90, fontsize=7)
    ax.set_yticks(range(11)); ax.set_yticklabels(C, fontsize=7)
    ax.set_title(f'RML2016.10a  SNR = {s} dB')
fig.tight_layout(); fig.savefig('fig7_repro_confusion.pdf', dpi=300)
fig.savefig('fig7_repro_confusion.png', dpi=150)

print('wrote fig6_repro_perclass.pdf/.png and fig7_repro_confusion.pdf/.png')
for s in cms:
    cm = np.mean(cms[s], axis=0)
    print(f"\nSNR {s:+d} dB:")
    print(f"  WBFM recall        : {cm[10,10]*100:5.1f}%   -> AM-DSB: {cm[10,8]*100:5.1f}%")
    print(f"  AM-DSB recall      : {cm[8,8]*100:5.1f}%   -> WBFM  : {cm[8,10]*100:5.1f}%")
    print(f"  QAM16 recall       : {cm[5,5]*100:5.1f}%   -> QAM64 : {cm[5,6]*100:5.1f}%")
    print(f"  QAM64 recall       : {cm[6,6]*100:5.1f}%   -> QAM16 : {cm[6,5]*100:5.1f}%")
