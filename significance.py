"""Per-class reproduction deltas vs measured seed variance.

The binomial error model is inadequate here: it assumes one model resampled,
whereas the comparison is between models trained from different random
initializations. Seed-to-seed variation dominates and is measured directly
from the three reproduction runs.
"""
import numpy as np, pandas as pd

C = ['8PSK','BPSK','CPFSK','GFSK','PAM4','QAM16','QAM64','QPSK','AM-DSB','AM-SSB','WBFM']
TAGS = ['r1','r2','r3']; K = len(TAGS); N = 200

paper = pd.read_csv('Test_modA_SNR.csv')
runs = np.array([pd.read_csv(f'perclass_{t}.csv')[C].values for t in TAGS])
ref, mine, seed_sd = paper[C].values, runs.mean(0), runs.std(0, ddof=1)

diff = (mine - ref) * 100
binom = np.sqrt(ref * (1 - ref) / N) * 100      # within-model sampling
seed = seed_sd * 100                             # between-model (measured)
se_one = np.sqrt(binom**2 + seed**2)             # paper: one run
se_comb = np.sqrt(se_one**2 + se_one**2 / K)     # vs our mean of K
z = np.divide(diff, se_comb, out=np.zeros_like(diff), where=se_comb > 1e-9)

valid = (ref * N >= 5) & (se_comb > 1e-9)        # normal approx needs np>=5
zv, cells = z[valid], valid.sum()
exp_max = np.sqrt(2 * np.log(cells))

print(f"cells compared            : {cells} of {z.size}  "
      f"({(~valid).sum()} excluded: expected count < 5)")
print(f"median binomial SE        : {np.median(binom):.2f} pp")
print(f"median measured seed SE   : {np.median(seed):.2f} pp   <- dominant term")
print(f"mean |delta|              : {np.abs(diff).mean():.2f} pp")
print(f"max  |z|                  : {np.abs(zv).max():.2f} sigma")
print(f"expected max |z| by chance: {exp_max:.2f} sigma")
print(f"beyond 2 sigma            : {(np.abs(zv)>2).sum()} / {cells}"
      f"   (chance: {0.0455*cells:.1f})")
print(f"beyond 3 sigma            : {(np.abs(zv)>3).sum()} / {cells}"
      f"   (chance: {0.0027*cells:.1f})")
print(f"\nverdict: {'CONSISTENT with seed variation' if np.abs(zv).max() < exp_max else 'EXCEEDS chance - investigate'}")

print("\ntop deviations (valid cells only):")
zm = np.where(valid, np.abs(z), -1)
idx = np.dstack(np.unravel_index(np.argsort(-zm, axis=None)[:8], z.shape))[0]
print(f"{'SNR':>5} {'class':<8} {'paper':>7} {'ours':>7} {'delta':>7} {'binom':>6} {'seed':>6} {'z':>6}")
for r, c in idx:
    print(f"{paper.SNR[r]:>5} {C[c]:<8} {ref[r,c]*100:>6.1f}% {mine[r,c]*100:>6.1f}% "
          f"{diff[r,c]:>+6.1f} {binom[r,c]:>5.2f} {seed[r,c]:>5.2f} {z[r,c]:>+6.2f}")

print("\nexcluded cells (expected count < 5, normal approximation invalid):")
er, ec = np.where(~valid)
for r, c in list(zip(er, ec))[:10]:
    print(f"  SNR {paper.SNR[r]:>4}  {C[c]:<8} paper {ref[r,c]*100:>5.1f}% "
          f"(~{ref[r,c]*N:.0f}/200 samples)  ours {mine[r,c]*100:>5.1f}%")

pd.DataFrame(z, columns=C).assign(SNR=paper.SNR).to_csv('perclass_zscores.csv', index=False)
