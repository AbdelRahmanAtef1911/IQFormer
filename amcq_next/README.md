# amcq_next — runbook for the next round of experiments

Everything runs from `~/amc/IQFormer` after `source ~/amc/.venv/bin/activate`. The add-on sits beside `amcq/`,
`amcq_arch/` and `amcq_dsp/` and changes none of them. **Every training run is validation-only.** The test set is
opened once, in Step 8.

After each step, send me the lines listed under **Send me**. I check them and tell you the next step (and, after
Step 3, which training recipe the later steps use).

| Step | What | When | GPU time |
| --- | --- | --- | --- |
| 0 | install + smoke test | now | 2 min |
| 1 | D1 ensemble ceiling + D2 low-SNR statistics | now | 15 min |
| 2 | D3 oracle-SNR ceiling (2 runs) + E1a cosine-100 recipe (5 runs) | night 1 | about 6 h |
| 3 | E1b cosine-100 + label smoothing (5 runs) | night 2 | about 5 h |
| 4 | E2a/E2b blind SNR branch (2 x 5 runs) | night 3 | 5-10 h |
| 5 | E3 phase augmentation + E5 InfoNCE (2 x 5 runs) | night 4 | 5-10 h |
| 6 | E4 complex STFT (5 runs) | night 5 | 5-6 h |
| 7 | F1 slim model with distillation (5 runs) | night 6 | 6-8 h |
| 8 | combine winners, one test run, then 4-bit QAT | after I confirm | |

Never run two trainings on the GPU at the same time. The short scripts (Steps 1, the compare commands) are fine
while a training runs.

## Step 0 — install and check (2 minutes)

```bash
cd ~/amc/IQFormer && source ~/amc/.venv/bin/activate
cp /mnt/c/Users/abdoa/Downloads/IQFormer_next.zip .        # adjust the path
unzip -o IQFormer_next.zip
python -m amcq_next.smoke 2>&1 | grep -v Warning
```

**Send me:** the smoke output. It must end with `ALL PASS`.

## Step 1 — diagnostics D1 and D2 (about 15 minutes, no training)

```bash
mkdir -p results_next
python -m amcq_next.d1_ensemble --runs runs_arch/iqformer_s1 runs_arch/iqformer_s2 runs_arch/iqformer_s3 runs_arch/iqformer_s4 runs_arch/iqformer_s5 2>&1 | grep -v Warning
python -m amcq_next.d2_lowsnr_stats 2>&1 | grep -v Warning
```

- **D1** averages the 5 baselines. If the ensemble beats one model by at least 1 pp, distillation (Step 7) has room.
- **D2** asks whether anything below -10 dB can still be learned. Simple window statistics (power, kurtosis, spectral
  flatness and asymmetry, cumulants) are fed to a small classifier. It also prints whether AM-SSB can be told
  apart from noise (AUC). A gain below -10 dB is only physically possible if these show some information there.

**Send me:** both printouts (or `results_next/d1_ensemble.csv` and `results_next/d2_lowsnr_stats.txt`).

## Step 2 — night 1: D3 oracle ceiling + E1a recipe

```bash
export PYTHONUNBUFFERED=1
nohup bash -c '
for s in 1 2; do
  python -m amcq_next.train --model iqfx_oracle --seed $s --out runs_next/d3_oracle_s$s
done
for s in 1 2 3 4 5; do
  python -m amcq_next.train --model iqformer --sched cosine --epochs 100 --warmup 5 --patience 0 --seed $s --out runs_next/e1_cos100_s$s
done' > night1.log 2>&1 &
```

- **D3 (diagnostic only)** feeds the *true* SNR bin to the classifier. It measures how much SNR knowledge is worth
  in our code (MC-IQFormer measured +3.8 pp). It is never run on test and never reported as a result.
- **E1a** is the same IQFormer trained 100 epochs with a cosine schedule and no early stop, instead of plateau
  with at most 60 epochs.

Check progress any time: `grep "\[validation\]" night1.log` (7 lines when done) and `tail -n 3 night1.log`.

**Send me (morning):**
```bash
grep "\[validation\]" night1.log
python -m amcq_next.compare --roots runs_arch runs_next --variants d3_oracle e1_cos100 2>&1 | grep -v Warning
```

## Step 3 — night 2: E1b recipe with label smoothing

```bash
nohup bash -c '
for s in 1 2 3 4 5; do
  python -m amcq_next.train --model iqformer --sched cosine --epochs 100 --warmup 5 --patience 0 --ls 0.1 --seed $s --out runs_next/e1_cos100ls_s$s
done' > night2.log 2>&1 &
```

**Send me (morning):**
```bash
grep "\[validation\]" night2.log
python -m amcq_next.compare --roots runs_arch runs_next --variants e1_cos100 e1_cos100ls 2>&1 | grep -v Warning
```

I then tell you the **RECIPE** for every later step: either `--sched cosine --epochs 100 --warmup 5 --patience 0`
(with `--ls 0.1` if it won), or nothing (the released recipe). I also tell you the **BASE** group the later
experiments are compared against (`e1_cos100`, `e1_cos100ls` or `iqformer`). In the commands below, set them first:

```bash
RECIPE="--sched cosine --epochs 100 --warmup 5 --patience 0"     # example: use what I send you
BASE=e1_cos100                                                     # example
```

## Step 4 — night 3: E2 blind SNR branch (MSTFFNet-style, concatenated)

```bash
nohup bash -c "
for s in 1 2 3 4 5; do
  python -m amcq_next.train --model iqfx_snr --snr-loss 0.05 $RECIPE --seed \$s --out runs_next/e2a_snr_s\$s
  python -m amcq_next.train --model iqfx_snr --snr-loss 0    $RECIPE --seed \$s --out runs_next/e2b_snrnoloss_s\$s
done" > night3.log 2>&1 &
```

- **E2a**: the SNR head is trained with the true SNR bin as a training-only target.
- **E2b**: the same branch without that target. It tells whether a gain comes from the SNR labels or just from the
  extra path.

**Send me:** `grep "\[validation\]" night3.log` and
`python -m amcq_next.compare --roots runs_arch runs_next --baseline $BASE --variants e2a_snr e2b_snrnoloss --detail e2a_snr`

## Step 5 — night 4: E3 phase augmentation + E5 InfoNCE

```bash
nohup bash -c "
for s in 1 2 3 4 5; do
  python -m amcq_next.train --model iqformer --aug phase $RECIPE --seed \$s --out runs_next/e3_phase_s\$s
  python -m amcq_next.train --model iqfx_ctr --ctr-weight 0.1 $RECIPE --seed \$s --out runs_next/e5_ctr_s\$s
done" > night4.log 2>&1 &
```

Then, in the morning (no training, about 10 minutes):

```bash
python -m amcq_next.tta --runs runs_next/e3_phase_s1 runs_next/e3_phase_s2 runs_next/e3_phase_s3 runs_next/e3_phase_s4 runs_next/e3_phase_s5 --methods guard mean 2>&1 | grep -v Warning
```

**Send me:** `grep "\[validation\]" night4.log`, the compare output for `e3_phase e5_ctr` against `$BASE`, and the
TTA table.

## Step 6 — night 5: E4 complex STFT with a 2-D encoder

```bash
nohup bash -c "
for s in 1 2 3 4 5; do
  python -m amcq_next.train --model iqfx_c2d $RECIPE --seed \$s --out runs_next/e4_c2d_s\$s
done" > night5.log 2>&1 &
```

**Send me:** `grep "\[validation\]" night5.log` and the compare output for `e4_c2d --detail e4_c2d`.

## Step 7 — night 6: F1 efficiency track (half-width IQFormer, about 91 K parameters)

Teachers are the 5 baseline models (or the E1 winner's runs if I say so). Distillation makes each epoch about 3x
slower.

```bash
T="runs_arch/iqformer_s1 runs_arch/iqformer_s2 runs_arch/iqformer_s3 runs_arch/iqformer_s4 runs_arch/iqformer_s5"
nohup bash -c "
for s in 1 2 3 4 5; do
  python -m amcq_next.train --model iqfx_slim --aug phase --teachers $T $RECIPE --seed \$s --out runs_next/f1_slim_kd_s\$s
done" > night6.log 2>&1 &
```

**Send me:** `grep "\[validation\]" night6.log` and the compare output for `f1_slim_kd` against `iqformer`.

## Step 8 — combine, test once, quantize

After Steps 2-7 I send you one command set: the combined model (for example `iqfx_snr_c2d` with phase
augmentation), its 5 training runs, then **one** held-out test with and without guarded TTA:

```bash
python -m amcq_next.tta --runs <the 5 chosen runs> --methods guard --final-test
```

then QAT to W4A4 with `python -m amcq_next.qat ...` (same arguments as `amcq_arch.qat`), exactly as in
Part 15 of the guide.

## Keeping the record

Every run folder has `log.csv`, `val_metrics.json`, `val_pred.npz` and `val_logits.npz` (for ensembles). Push at the
end of each step (logs and predictions, not the `best.pt` weights):

```bash
git add amcq_next results_next runs_next ':!runs_next/*/best.pt' night*.log
git commit -m "next round: <step>"
git push origin phase3
```

## What is in the package

| File | Purpose |
| --- | --- |
| `models.py` | `iqfx_*` models: `snr` (blind SNR branch, concatenated), `oracle` (diagnostic), `c2d` (complex 4-channel STFT, 2-D encoder), `slim` (half width), `ctr` (InfoNCE heads); options combine, e.g. `iqfx_snr_c2d` |
| `train.py` | validation-only training; `--aug phase`, `--snr-loss`, `--ctr-weight`, `--teachers` |
| `d1_ensemble.py`, `d2_lowsnr_stats.py` | diagnostics D1, D2 |
| `compare.py`, `tta.py`, `final_test.py`, `ptq.py`, `qat.py` | Welch comparison on validation; guarded TTA; one test evaluation; quantization of the chosen model |
| `smoke.py` | checks before the long runs |
