import argparse, time, numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset.dataset import RMLtest
from model.IQFormer import IQFormer

CLASSES = ['8PSK','BPSK','CPFSK','GFSK','PAM4','QAM16','QAM64','QPSK','AM-DSB','AM-SSB','WBFM']

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='save_models/model_2016.10a_60_256_0.001_IQFormer/weight.pt')
ap.add_argument('--tag', default='r1')
a = ap.parse_args()

data = pd.read_pickle('./dataset/RML2016.10a_dict.pkl')
te = [[], [], []]
for (label, SNR), samples in data.items():
    if label not in CLASSES: continue
    lab = np.full(len(samples), CLASSES.index(label))
    snr = np.full(len(samples), SNR)
    _, x, _, y, _, s = train_test_split(samples, lab, snr, test_size=0.2,
                                        random_state=233, stratify=lab)
    te[0].extend(x); te[1].extend(y); te[2].extend(s)
loader = DataLoader(RMLtest(np.array(te[0]), np.array(te[1]), np.array(te[2])),
                    batch_size=400, shuffle=False)

dev = 'cuda:0'
model = IQFormer([2,3,2], embed_dims=[64,64,64], mlp_ratios=4, act_layer=nn.GELU,
                 num_classes=11, down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0.2, drop_path_rate=0., use_layer_scale=False,
                 layer_scale_init_value=1e-5, fork_feat=False, vit_num=1).to(dev)
model.load_state_dict(torch.load(a.ckpt, map_location=dev))
model.eval()

P, T, S, n, t = [], [], [], 0, 0.0
with torch.no_grad():
    for iq, stft, snr, lab in loader:
        iq, stft = iq.to(dev), stft.to(dev)
        torch.cuda.synchronize(); t0 = time.time()
        out = model(iq, stft)
        torch.cuda.synchronize(); t += time.time() - t0; n += len(lab)
        P.append(out.argmax(1).cpu().numpy()); T.append(lab.numpy()); S.append(np.asarray(snr))
P, T, S = np.concatenate(P), np.concatenate(T), np.concatenate(S)
snrs = sorted(np.unique(S))

# per-class recall, same layout as the authors' Test_modA_SNR.csv
rows = []
for s in snrs:
    m = S == s
    rows.append({'SNR': s, **{c: (P[m & (T==i)] == i).mean() for i, c in enumerate(CLASSES)}})
pc = pd.DataFrame(rows)
pc.to_csv(f'perclass_{a.tag}.csv', index=False)

acc = pd.DataFrame([{'SNR': s, 'accuracy': (P[S==s]==T[S==s]).mean()} for s in snrs])
acc.to_csv(f'per_snr_{a.tag}.csv', index=False)

lo = acc[acc.SNR <= 0].accuracy.mean()
hi = acc[acc.SNR >= 0].accuracy.mean()
print(f"\n--- Table II row, {a.tag} ---")
print(f"{'metric':<20}{'paper':>10}{'yours':>10}{'delta':>9}")
for k, p, v in [('Highest Accuracy', 93.91, acc.accuracy.max()*100),
                ('SNR -20-0',        40.33, lo*100),
                ('SNR 0-18',         93.15, hi*100),
                ('OverAll',          64.19, acc.accuracy.mean()*100)]:
    print(f"{k:<20}{p:>9.2f}%{v:>9.2f}%{v-p:>+8.2f}")
print(f"{'Params':<20}{'0.35M':>10}{sum(x.numel() for x in model.parameters())/1e6:>9.2f}M")
print(f"{'Inference (ms/smp)':<20}{0.7114:>10.4f}{t/n*1000:>10.4f}   (hardware-dependent)")
