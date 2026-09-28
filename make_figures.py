import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASSES = ['8PSK','BPSK','CPFSK','GFSK','PAM4','QAM16','QAM64','QPSK','AM-DSB','AM-SSB','WBFM']
TAGS = ['r1','r2','r3']

paper = pd.read_csv('Test_modA_SNR.csv')
paper_acc = paper[CLASSES].mean(axis=1).values
snr = paper['SNR'].values

runs = np.array([pd.read_csv(f'per_snr_{t}.csv').accuracy.values for t in TAGS])
mean, std = runs.mean(0), runs.std(0, ddof=1)

# Fig 1: accuracy vs SNR
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(snr, paper_acc*100, 'k--o', ms=5, lw=1.5, label='Published (Shao et al.)')
ax.plot(snr, mean*100, '-s', ms=5, lw=1.5, color='#c0392b', label='Reproduction (mean, n=3)')
ax.fill_between(snr, (mean-std)*100, (mean+std)*100, color='#c0392b', alpha=0.2, label='±1 std')
ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Accuracy (%)')
ax.set_xticks(snr[::2]); ax.set_ylim(0, 100); ax.grid(alpha=0.3)
ax.legend(loc='upper left', fontsize=9)
ax.set_title('IQFormer on RadioML 2016.10A: reproduction vs published')
fig.tight_layout(); fig.savefig('fig_acc_vs_snr.pdf', dpi=300); fig.savefig('fig_acc_vs_snr.png', dpi=150)

# Fig 2: per-class recall difference heatmap
pc = np.mean([pd.read_csv(f'perclass_{t}.csv')[CLASSES].values for t in TAGS], axis=0)
diff = (pc - paper[CLASSES].values) * 100
fig, ax = plt.subplots(figsize=(8, 5))
v = np.abs(diff).max()
im = ax.imshow(diff.T, cmap='RdBu_r', vmin=-v, vmax=v, aspect='auto')
ax.set_xticks(range(len(snr))); ax.set_xticklabels(snr, rotation=45, fontsize=8)
ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES, fontsize=9)
ax.set_xlabel('SNR (dB)')
ax.set_title('Per-class recall: reproduction − published (pp)')
fig.colorbar(im, ax=ax, label='percentage points')
fig.tight_layout(); fig.savefig('fig_perclass_diff.pdf', dpi=300); fig.savefig('fig_perclass_diff.png', dpi=150)

print('wrote fig_acc_vs_snr.pdf/.png and fig_perclass_diff.pdf/.png')
print(f'max |per-class delta|: {np.abs(diff).max():.1f} pp')
print(f'mean |per-class delta|: {np.abs(diff).mean():.2f} pp')
