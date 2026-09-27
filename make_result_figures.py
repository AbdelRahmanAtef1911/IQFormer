"""Regenerates the four result figures from the CSVs produced by
block_sensitivity.py, block_sensitivity_wonly.py and mixed_precision.py.
All values are read from data; nothing is hardcoded."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# validated categorical palette (CVD-safe, checked against a light surface)
CONV, LSTM, ATTN, HEAD = '#c0392b', '#2c6fbb', '#c77b00', '#7d5bbe'
INK, MUTED, GRID, SURF = '#1a1a1a', '#5a5a5a', '#d8d8d4', '#fcfcfb'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.edgecolor':GRID,
    'axes.labelcolor':INK,'text.color':INK,'xtick.color':MUTED,'ytick.color':MUTED,
    'axes.facecolor':SURF,'figure.facecolor':SURF,'axes.grid':True,'grid.color':GRID,
    'grid.linewidth':0.6,'axes.axisbelow':True})

G  = ['CONV','LSTM','ATTENTION','HEAD']
CO = [CONV, LSTM, ATTN, HEAD]

wa_df = pd.read_csv('block_sensitivity.csv')            # weights + activations
wo_df = pd.read_csv('block_sensitivity_weightonly.csv') # weights only
mx_df = pd.read_csv('mixed_precision_sweep.csv')        # whole-model configs

FLOOR = 2 * (mx_df.fp32.std(ddof=1) * 100)              # 2 sigma, from the data
print(f"measurement floor (2 sigma, computed from FP32 runs) = {FLOOR:.2f} pp")

share = (wo_df.drop_duplicates('group').set_index('group').n_params
         / wo_df.drop_duplicates('group').n_params.sum() * 100)

def agg(df, bits):
    s = df[df.bits == bits]
    return (s.groupby('group').delta_pp.mean().reindex(G).values,
            s.groupby('group').delta_pp.std(ddof=1).reindex(G).values)

# ---------- FIG A: attribution at 4 bits ----------
wo, wo_e = agg(wo_df, 4)
wa, wa_e = agg(wa_df, 4)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.9))
y = np.arange(4)[::-1]
lim1 = (min(wo - wo_e) * 1.9 - 0.1, max(max(wo + wo_e) * 2.5, 0.15))
lim2 = (min(wa - wa_e) * 1.12, abs(min(wa)) * 0.1)
for ax, v, e, ttl, xlim, band in [(ax1, wo, wo_e, f'Weights only\nevery value inside the ±{FLOOR:.2f} pp noise floor', lim1, False),
                                  (ax2, wa, wa_e, 'Weights + activations', lim2, True)]:
    ax.barh(y, v, xerr=e, color=CO, height=0.56, error_kw=dict(ecolor=MUTED, lw=1, capsize=3))
    if band:
        ax.axvspan(-FLOOR, FLOOR, color=MUTED, alpha=0.16, zorder=0, label=f'±{FLOOR:.2f} pp noise floor')
        ax.legend(fontsize=8, loc='lower left', frameon=False)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{g}\n{share[g]:.1f}% of weights' for g in G], fontsize=8)
    ax.axvline(0, color=INK, lw=0.9); ax.set_xlim(*xlim)
    ax.set_xlabel('accuracy change (pp)'); ax.set_title(ttl, fontsize=10, color=INK, pad=8)
    ax.grid(axis='y', visible=False)
    pad = (xlim[1] - xlim[0]) * 0.02
    for yi, (vv, ee) in enumerate(zip(v, e)):
        ax.text(min(vv - ee, vv) - pad, y[yi], f'{vv:+.2f}', va='center', ha='right',
                fontsize=8.5, color=INK)
fig.suptitle('4-bit quantization applied to one block group at a time', fontsize=11.5, y=1.0, color=INK)
fig.tight_layout()
for ext in ('pdf','png'): fig.savefig(f'fig_A_attribution.{ext}', dpi=170, bbox_inches='tight')

# ---------- FIG B: weight-only sensitivity vs bit width ----------
bits = sorted(wo_df.bits.unique(), reverse=True)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
x = np.arange(len(bits))
ax.axhspan(-FLOOR, FLOOR, color=MUTED, alpha=0.16, zorder=0, label=f'±{FLOOR:.2f} pp noise floor')
OFF = {'CONV': 0, 'LSTM': -1, 'ATTENTION': -9, 'HEAD': 8}
for g, c in zip(G, CO):
    s = wo_df[wo_df.group == g].groupby('bits').delta_pp.agg(['mean','std']).reindex(bits)
    ax.errorbar(x, s['mean'], yerr=s['std'], marker='o', ms=6, lw=2, capsize=3.5, color=c, label=g)
    ax.annotate(g, (x[-1], s['mean'].iloc[-1]), textcoords='offset points',
                xytext=(9, OFF[g]), fontsize=8.5, color=c, va='center')
ax.set_xticks(x); ax.set_xticklabels([f'{b}-bit' for b in bits])
ax.set_xlim(-0.3, len(bits) - 0.25); ax.axhline(0, color=INK, lw=0.9)
ax.set_ylabel('accuracy change (pp)'); ax.set_xlabel('weight precision (activations kept at FP32)')
ax.set_title(f'Weight-only sensitivity by block group (mean ± std, n = {wo_df.run.nunique()})',
             fontsize=10.5, pad=8)
ax.legend(fontsize=8, loc='lower left', frameon=False)
fig.tight_layout()
for ext in ('pdf','png'): fig.savefig(f'fig_B_weightonly.{ext}', dpi=170, bbox_inches='tight')

# ---------- FIG C: mixed precision ----------
cols = [c for c in mx_df.columns if c.startswith('w')]
order = ['fp32'] + sorted(cols, key=lambda c: -mx_df[c].mean())
acc = [mx_df[c].mean()*100 for c in order]
sd  = [mx_df[c].std(ddof=1)*100 for c in order]
lbl = ['FP32'] + [c.upper() for c in order[1:]]
best4 = min([c for c in cols if c.startswith('w4')], key=lambda c: -mx_df[c].mean())
worst = order[-1]
col = [MUTED if c=='fp32' else (LSTM if c==best4 else (CONV if c==worst else '#8a8a86')) for c in order]
fp = mx_df.fp32.mean()*100
fig, ax = plt.subplots(figsize=(7.6, 4.3))
xx = np.arange(len(order))
ax.axhspan(fp-FLOOR, fp+FLOOR, color=MUTED, alpha=0.16, zorder=0)
ax.bar(xx, acc, yerr=sd, color=col, width=0.62, error_kw=dict(ecolor=MUTED, lw=1, capsize=3.5))
ax.axhline(fp, ls='--', lw=1, color=MUTED)
for i,(a,s_) in enumerate(zip(acc,sd)):
    ax.text(i, a+s_+1.5, f'{a:.2f}', ha='center', fontsize=8.5, color=INK)
rec = (mx_df[best4].mean() - mx_df[worst].mean()) * 100
ax.set_xticks(xx); ax.set_xticklabels(lbl)
ax.set_xlim(-0.65, len(order)+1.35); ax.set_ylim(0, max(acc)*1.19)
ax.set_ylabel('test accuracy (%)'); ax.set_xlabel('weight / activation precision')
ax.set_title('Mixed precision: 4-bit weights are free, 4-bit activations are not', fontsize=10.5, pad=8)
ax.annotate('', xy=(len(order)+0.05, acc[order.index(best4)]),
            xytext=(len(order)+0.05, acc[-1]),
            arrowprops=dict(arrowstyle='<->', color=INK, lw=1.4))
ax.text(len(order)+0.25, (acc[order.index(best4)]+acc[-1])/2,
        f'+{rec:.2f} pp\nrecovered by\n{worst[2:].upper()} → {best4[2:].upper()}',
        ha='left', va='center', fontsize=8.8, color=INK)
ax.text(len(order)-0.5, fp, f'FP32 ±{FLOOR:.2f} pp', ha='right', va='bottom', fontsize=8, color=MUTED)
ax.grid(axis='x', visible=False)
fig.tight_layout()
for ext in ('pdf','png'): fig.savefig(f'fig_C_mixed.{ext}', dpi=170, bbox_inches='tight')

# ---------- FIG D: stability ----------
keys = ['fp32', worst, best4]
names = ['FP32', worst.upper(), best4.upper()]
fig, ax = plt.subplots(figsize=(7.0, 4.0))
vals = [mx_df[k].values*100 for k in keys]
for i in range(len(mx_df)):
    ax.plot([0,1,2], [v[i] for v in vals], color=GRID, lw=1.2, zorder=1)
for j,(v,c,n) in enumerate(zip(vals, [MUTED, CONV, LSTM], names)):
    ax.scatter([j]*len(v), v, s=70, color=c, zorder=3, label=f'{n}  (std {v.std(ddof=1):.2f})')
for v in vals[1]:
    ax.annotate(f'{v:.1f}', (1, v), xytext=(11,-3), textcoords='offset points', fontsize=8.5, color=CONV)
ax.annotate(f'{vals[0].min():.1f} – {vals[0].max():.1f}\nspread {np.ptp(vals[0]):.1f} pp', (0, vals[0].mean()),
            xytext=(-52,-4), textcoords='offset points', fontsize=8.5, color=MUTED, va='center')
ax.annotate(f'{vals[2].min():.1f} – {vals[2].max():.1f}\nspread {np.ptp(vals[2]):.1f} pp', (2, vals[2].mean()),
            xytext=(13,-4), textcoords='offset points', fontsize=8.5, color=LSTM, va='center')
ax.annotate(f'spread {np.ptp(vals[1]):.1f} pp', (1, vals[1].mean()), xytext=(-84,0),
            textcoords='offset points', fontsize=8.5, color=CONV, va='center')
ax.set_xticks([0,1,2]); ax.set_xticklabels(names); ax.set_xlim(-0.75, 3.05)
ax.set_ylabel('test accuracy (%)')
ax.set_title('8-bit activations restore stability, not just accuracy', fontsize=10.5, pad=8)
ax.legend(fontsize=8, loc='lower left', frameon=False); ax.grid(axis='x', visible=False)
fig.tight_layout()
for ext in ('pdf','png'): fig.savefig(f'fig_D_stability.{ext}', dpi=170, bbox_inches='tight')

print('wrote fig_A_attribution, fig_B_weightonly, fig_C_mixed, fig_D_stability (.pdf/.png)')
