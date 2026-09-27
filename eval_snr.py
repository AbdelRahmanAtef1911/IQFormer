import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset.dataset import RMLtest
from model.IQFormer import IQFormer

classes = ['8PSK','BPSK','CPFSK','GFSK','PAM4','QAM16','QAM64','QPSK','AM-DSB','AM-SSB','WBFM']
data = pd.read_pickle('./dataset/RML2016.10a_dict.pkl')
te = [[],[],[]]
for (label, SNR), samples in data.items():
    if label not in classes: continue
    labels = np.full(len(samples), classes.index(label))
    snrs = np.full(len(samples), SNR)
    X, x, Y, y, _, SNR_te = train_test_split(samples, labels, snrs, test_size=0.2,
                                             random_state=233, stratify=labels)
    te[0].extend(x); te[1].extend(y); te[2].extend(SNR_te)

ds = RMLtest(np.array(te[0]), np.array(te[1]), np.array(te[2]))
loader = DataLoader(ds, batch_size=400, shuffle=False)

dev = 'cuda:0'
model = IQFormer([2,3,2], embed_dims=[64,64,64], mlp_ratios=4, act_layer=nn.GELU,
                 num_classes=11, down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0.2, drop_path_rate=0., use_layer_scale=False,
                 layer_scale_init_value=1e-5, fork_feat=False, vit_num=1).to(dev)
model.load_state_dict(torch.load('save_models/model_2016.10a_60_256_0.001_IQFormer/weight.pt',
                                 map_location=dev))
model.eval()

P, T, S = [], [], []
with torch.no_grad():
    for iq, stft, snr, lab in loader:
        out = model(iq.to(dev), stft.to(dev))
        P.append(out.argmax(1).cpu().numpy()); T.append(lab.numpy()); S.append(np.asarray(snr))
P, T, S = np.concatenate(P), np.concatenate(T), np.concatenate(S)

rows = [{'SNR': s, 'accuracy': (P[S==s]==T[S==s]).mean(), 'n': (S==s).sum()}
        for s in sorted(np.unique(S))]
df = pd.DataFrame(rows)
df.to_csv('fp32_per_snr.csv', index=False)
print(df.to_string(index=False))
print(f"\noverall: {(P==T).mean():.6f}")
