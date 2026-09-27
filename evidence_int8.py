"""Evidence that INT8 quantization does not degrade IQFormer accuracy."""
import copy, torch, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from scipy import stats
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
MARGIN = 0.5   # equivalence margin in percentage points, pre-specified

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

def swap(module, bits=8):
    class WQ(Int8WeightPerChannelFloat): bit_width = bits
    class AQ(Int8ActPerTensorFloat):     bit_width = bits
    def rec(mod):
        for name, ch in mod.named_children():
            kw = dict(weight_quant=WQ, input_quant=AQ)
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
                rec(ch); continue
            q.load_state_dict(ch.state_dict(), strict=False)
            setattr(mod, name, q)
    rec(module)

@torch.no_grad()
def calibrate(m):
    with calibration_mode(m):
        for iq, st, _, _ in calib: m(iq.to(dev), st.to(dev))

@torch.no_grad()
def predict(m):
    P, T, S = [], [], []
    for iq, st, sn, lb in loader:
        P.append(m(iq.to(dev), st.to(dev)).argmax(1).cpu().numpy())
        T.append(lb.numpy()); S.append(np.asarray(sn))
    return np.concatenate(P), np.concatenate(T), np.concatenate(S)

per_snr_rows, summary_rows = [], []
w_fp_sample = w_q_sample = None
for tag, ck in CK.items():
    fp = build(ck)
    q = copy.deepcopy(fp); swap(q, 8); q = q.to(dev).eval(); calibrate(q)
    if w_fp_sample is None:
        f_l = [m for _, m in fp.named_modules() if isinstance(m, nn.Conv1d)][2]
        q_l = [m for _, m in q.named_modules()  if isinstance(m, QuantConv1d)][2]
        w_fp_sample = f_l.weight.detach().cpu().numpy().ravel()
        w_q_sample  = q_l.quant_weight().value.detach().cpu().numpy().ravel()
    Pf, T, S = predict(fp)
    Pq, _, _ = predict(q)
    changed = Pf != Pq
    f_ok, q_ok = (Pf == T), (Pq == T)
    b = int((f_ok & ~q_ok).sum()); c = int((~f_ok & q_ok).sum())
    mp = stats.binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    summary_rows.append(dict(run=tag, n_test=len(T), acc_fp32=f_ok.mean(), acc_int8=q_ok.mean(),
        delta_pp=(q_ok.mean() - f_ok.mean()) * 100, n_pred_changed=int(changed.sum()),
        pct_pred_changed=changed.mean() * 100, n_fp32right_int8wrong=b,
        n_fp32wrong_int8right=c, mcnemar_p=mp))
    for s in sorted(np.unique(S)):
        k = S == s
        per_snr_rows.append(dict(run=tag, SNR=int(s), acc_fp32=f_ok[k].mean(),
            acc_int8=q_ok[k].mean(), delta_pp=(q_ok[k].mean() - f_ok[k].mean()) * 100,
            n_changed=int(changed[k].sum()), n=int(k.sum())))
    print(f"{tag}: FP32 {f_ok.mean()*100:.3f}%  INT8 {q_ok.mean()*100:.3f}%  "
          f"| {int(changed.sum())}/{len(T)} predictions changed "
          f"({b} worse, {c} better, McNemar p={mp:.3f})")

S_df = pd.DataFrame(summary_rows); S_df.to_csv('int8_evidence_summary.csv', index=False)
P_df = pd.DataFrame(per_snr_rows); P_df.to_csv('int8_evidence_per_snr.csv', index=False)
pd.DataFrame({'w_fp32': w_fp_sample, 'w_int8': w_q_sample}).to_csv('int8_weight_grid.csv', index=False)

d = S_df.delta_pp.values
n = len(d); m, sd = d.mean(), d.std(ddof=1); se = sd / np.sqrt(n)
p_nhst = 2 * stats.t.sf(abs(m / se), n - 1)
p_tost = max(stats.t.sf((m + MARGIN) / se, n - 1), stats.t.cdf((m - MARGIN) / se, n - 1))
ci = stats.t.interval(0.90, n - 1, loc=m, scale=se)
print("\n" + "=" * 68); print("STATISTICAL EVIDENCE"); print("=" * 68)
print(f"paired deltas (INT8 - FP32) : {np.round(d,4)} pp")
print(f"mean                        : {m:+.4f} pp   (sd {sd:.4f}, se {se:.4f})")
print(f"90% confidence interval     : [{ci[0]:+.4f}, {ci[1]:+.4f}] pp\n")
print(f"Difference test  H0: delta = 0      -> p = {p_nhst:.4f} "
      f"({'no significant difference' if p_nhst > .05 else 'SIGNIFICANT difference'})")
print(f"Equivalence test (TOST), margin +/-{MARGIN} pp -> p = {p_tost:.5f} "
      f"({'EQUIVALENT to FP32' if p_tost < .05 else 'equivalence NOT established'})\n")
print(f"total predictions changed   : {S_df.n_pred_changed.sum()} of {S_df.n_test.sum()}")
print(f"  FP32 right -> INT8 wrong  : {S_df.n_fp32right_int8wrong.sum()}")
print(f"  FP32 wrong -> INT8 right  : {S_df.n_fp32wrong_int8right.sum()}")
print(f"McNemar p-values            : {np.round(S_df.mcnemar_p.values,3)} "
      f"({'symmetric = noise' if (S_df.mcnemar_p > .05).all() else 'asymmetric = real harm'})")
with open('int8_evidence_stats.txt', 'w') as f:
    f.write(f"paired deltas (pp): {list(np.round(d,4))}\nmean {m:+.4f} pp, sd {sd:.4f}, se {se:.4f}, n={n}\n")
    f.write(f"90% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}] pp\ndifference test p = {p_nhst:.4f}\n")
    f.write(f"TOST equivalence, margin +/-{MARGIN} pp, p = {p_tost:.5f}\n")
    f.write(f"predictions changed: {S_df.n_pred_changed.sum()} / {S_df.n_test.sum()}\n")
    f.write(f"worse {S_df.n_fp32right_int8wrong.sum()}, better {S_df.n_fp32wrong_int8right.sum()}\n")
    f.write(f"McNemar p: {list(np.round(S_df.mcnemar_p.values,4))}\n")

RED, BLU, GRY = '#c0392b', '#2c6fbb', '#888888'
sw = pd.read_csv('bitwidth_sweep.csv')
cols = ['fp32', 'w8a8', 'w6a6', 'w4a4', 'w2a2']
mu = np.array([sw[c].mean() for c in cols]) * 100
sg = np.array([sw[c].std(ddof=1) for c in cols]) * 100
fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.errorbar(range(5), mu, yerr=sg, marker='o', ms=7, lw=1.8, capsize=5, color=RED)
ax.axhline(mu[0], ls='--', lw=1, color=GRY)
ax.axhspan(mu[0]-0.62, mu[0]+0.62, color=BLU, alpha=0.15, label='FP32 ±2σ (measurement floor)')
ax.axhline(100/11, ls=':', lw=1.2, color='k', label='chance level (9.09%)')
for i, v in enumerate(mu):
    ax.annotate(f'{v:.2f}%', (i, v), textcoords='offset points', xytext=(0,12), ha='center', fontsize=9)
ax.set_xticks(range(5)); ax.set_xticklabels(['FP32','W8A8','W6A6','W4A4','W2A2'])
ax.set_ylabel('Test accuracy (%)'); ax.set_xlabel('Weight and activation precision')
ax.set_ylim(0, 75); ax.grid(alpha=0.25); ax.legend(fontsize=8, loc='lower left')
ax.set_title('Accuracy vs quantization precision (mean ± std, n = 3)')
fig.tight_layout(); fig.savefig('fig_bitwidth_sweep.pdf', dpi=300); fig.savefig('fig_bitwidth_sweep.png', dpi=150)

g = P_df.groupby('SNR'); snr = np.array(sorted(P_df.SNR.unique()))
a_f = g.acc_fp32.mean().values*100; a_q = g.acc_int8.mean().values*100
dl = g.delta_pp.mean().values; ds = g.delta_pp.std(ddof=1).values
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.8, 6.2), sharex=True, gridspec_kw={'height_ratios':[2.3,1]})
a1.plot(snr, a_f, '-o', ms=5, lw=1.6, color='k', label='FP32')
a1.plot(snr, a_q, '--s', ms=5, lw=1.6, color=RED, label='INT8 (W8A8)')
a1.set_ylabel('Accuracy (%)'); a1.set_ylim(0,100); a1.grid(alpha=0.25)
a1.legend(fontsize=9); a1.set_title('FP32 vs INT8 accuracy at every SNR (mean of 3 runs)')
a2.axhspan(-0.62, 0.62, color=BLU, alpha=0.18, label='±2σ measurement floor')
a2.axhline(0, color='k', lw=0.8)
a2.errorbar(snr, dl, yerr=ds, marker='o', ms=4, lw=1.2, capsize=3, color=RED)
a2.set_xlabel('SNR (dB)'); a2.set_ylabel('INT8 − FP32 (pp)')
a2.set_xticks(snr[::2]); a2.grid(alpha=0.25); a2.legend(fontsize=8, loc='lower right')
fig.tight_layout(); fig.savefig('fig_int8_per_snr.pdf', dpi=300); fig.savefig('fig_int8_per_snr.png', dpi=150)

fig, (b1, b2) = plt.subplots(1, 2, figsize=(10, 4))
b1.hist(w_fp_sample, bins=120, color='k', alpha=0.75)
b1.set_title(f'FP32 weights\n{len(np.unique(w_fp_sample))} distinct values')
b1.set_xlabel('weight value'); b1.set_ylabel('count')
b2.hist(w_q_sample, bins=120, color=RED, alpha=0.8)
b2.set_title(f'INT8 weights\n{len(np.unique(w_q_sample))} distinct values (256-level per-channel grid)')
b2.set_xlabel('weight value')
for a in (b1, b2): a.grid(alpha=0.25)
fig.suptitle('Proof that quantization was applied — one convolutional layer', y=1.02)
fig.tight_layout(); fig.savefig('fig_weight_grid.pdf', dpi=300, bbox_inches='tight')
fig.savefig('fig_weight_grid.png', dpi=150, bbox_inches='tight')

nc = g.n_changed.mean().values; nn_ = g.n.mean().values
fig, ax = plt.subplots(figsize=(6.8, 3.8))
ax.bar(snr, nc/nn_*100, width=1.5, color=RED, alpha=0.85)
ax.set_xlabel('SNR (dB)'); ax.set_ylabel('% of predictions changed')
ax.set_xticks(snr[::2]); ax.grid(alpha=0.25, axis='y')
ax.set_title('Fraction of predictions altered by INT8 quantization (mean of 3 runs)')
fig.tight_layout(); fig.savefig('fig_prediction_changes.pdf', dpi=300); fig.savefig('fig_prediction_changes.png', dpi=150)

print("\nwrote: int8_evidence_summary.csv, int8_evidence_per_snr.csv,")
print("       int8_weight_grid.csv, int8_evidence_stats.txt")
print("       fig_bitwidth_sweep, fig_int8_per_snr, fig_weight_grid, fig_prediction_changes (.pdf/.png)")
