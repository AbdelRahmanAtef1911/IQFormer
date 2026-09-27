"""Is the reproduction's per-class deviation larger than sampling noise? """
import numpy as np, pandas as pd

C = ['8PSK','BPSK','CPFSK','GFSK','PAM4','QAM16','QAM64','QPSK','AM-DSB','AM-SSB','WBFM']
TAGS, N_RUNS = ['r1','r2','r3'], 3

paper = pd.read_csv('Test_modA_SNR.csv')
mine = np.mean([pd.read_csv(f'perclass_{t}.csv')[C].values for t in TAGS], axis=0)
ref = paper[C].values
diff = (mine - ref) * 100

n = 200                                     # 2200 per SNR / 11 classes
se1 = np.sqrt(ref * (1 - ref) / n) * 100    # paper: single run
se_comb = np.sqrt(se1**2 + (se1**2)/N_RUNS) # vs our mean of N_RUNS
z = np.divide(diff, se_comb, out=np.zeros_like(diff), where=se_comb > 0)

cells = diff.size
expected_max = np.sqrt(2 * np.log(cells))

print(f"cells compared          : {cells}")
print(f"samples per cell        : {n}")
print(f"mean |delta|            : {np.abs(diff).mean():.2f} pp")
print(f"max  |delta|            : {np.abs(diff).max():.1f} pp")
print(f"max  |z|                : {np.abs(z).max():.2f} sigma")
print(f"expected max |z| by chance: {expected_max:.2f} sigma")
print(f"cells beyond 2 sigma    : {(np.abs(z) > 2).sum()} / {cells}"
      f"   (expected by chance: {0.0455*cells:.1f})")
print(f"cells beyond 3 sigma    : {(np.abs(z) > 3).sum()} / {cells}"
      f"   (expected by chance: {0.0027*cells:.1f})")
verdict = "CONSISTENT with sampling noise" if np.abs(z).max() < expected_max \
          else "EXCEEDS chance - investigate"
print(f"\nverdict: {verdict}")

print("\ntop deviations:")
idx = np.dstack(np.unravel_index(np.argsort(-np.abs(z), axis=None)[:8], z.shape))[0]
print(f"{'SNR':>5} {'class':<8} {'paper':>7} {'ours':>7} {'delta':>7} {'SE':>6} {'z':>6}")
for r, c in idx:
    print(f"{paper.SNR[r]:>5} {C[c]:<8} {ref[r,c]*100:>6.1f}% {mine[r,c]*100:>6.1f}% "
          f"{diff[r,c]:>+6.1f} {se_comb[r,c]:>5.2f} {z[r,c]:>+6.2f}")

pd.DataFrame(z, columns=C).assign(SNR=paper.SNR).to_csv('perclass_zscores.csv', index=False)
