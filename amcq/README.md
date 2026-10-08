# amcq — experiments for the preliminary and final reports

Builds on what is already done (verified FP32 reproduction, Brevitas INT8/W4A8 study) and adds what
the next two reports need:

| Report needs (syllabus) | Step |
|---|---|
| Approach + preliminary results: quantization of the **whole** model, LSTM included | 3, 4 |
| Discussion and error analysis: which block breaks, and why 4-bit collapses | 5, 5b, 6 |
| "Different models such as MLP, CNN, and transformer" (10 pts) | 7 |
| "Propose new model / system / architecture" (20 pts): the proposed low-precision IQFormer | 9, 10 |
| Comparison with other works, ablation study (final report) | 7, 9, 10, 12 |
| Real-time argument (batch size 1) | 11 |

## What is new compared with the Brevitas scripts

* **The LSTM is quantized too.** `nn.LSTM` cannot be swapped by Brevitas; here it is unrolled with the
  same weights (`QLSTM`), so its weights, inputs and recurrent state are quantized. Before, 11.8 % of
  the model stayed FP32.
* **Integer-style scheme.** BatchNorm that follows a convolution is folded into it (exact), as on an FPGA.
  `--act-scheme symmetric --no-fold --calib-method percentile --pct 99.999` reproduces the earlier
  Brevitas setup, so old and new numbers can be compared (step 3).
* **STFT on the GPU.** Same numbers as scipy (checked in step 1), much faster than the per-sample CPU STFT.
* **Batch size 1 works** (`squeeze(dim=2)`), so latency can be measured (step 11).
* **QAT** with learned step sizes (LSQ) and optional **distillation** from the FP32 runs.

The authors' code is not modified: `amcq` imports `model/IQFormer.py` and the baselines from the repository.

## A to Z on Windows 11 + WSL2 (Ubuntu)

### 0. Put the code in place (once)

Open **Ubuntu** from the Start menu, then:

```bash
cd ~/amc/IQFormer
git checkout main && git pull            # your fork, tag v1-block-sensitivity
git checkout -b phase3                   # keep main = the verified reproduction
cp /mnt/c/Users/<YOUR_WINDOWS_USER>/Downloads/amcq.zip .    # where the browser saved it
unzip amcq.zip                           # creates ~/amc/IQFormer/amcq/
source ~/amc/.venv/bin/activate
```

Nothing new to install: the scripts use torch, numpy, scipy, scikit-learn, pandas, matplotlib, timm,
einops (already in the venv). Brevitas is not needed.

### 1. Check the environment

```bash
bash amcq/run.sh step1
```

Passes when it prints `ALL CHECKS PASSED`: GPU matmul ran, dataset found, and the GPU STFT matches
scipy (relative error below 1e-4).

### 2. Import the three verified FP32 models (2 min)

```bash
bash amcq/run.sh step2
```

Evaluates `save_models/model_2016.10a_60_256_0.001_IQFormer{,_r2,_r3}/weight.pt` with the new code and
stores them as `runs/iqformer_s1..s3`. **Overall must be 63.44 / 63.91 / 64.02 % (±0.02).** If not,
stop: the split or the preprocessing differs. (Different checkpoint paths: `CK1=... CK2=... CK3=... bash amcq/run.sh step2`.)

### 2b. Prove the toolkit is correct (5 min)

```bash
bash amcq/run.sh verify
```

18 checks (17 on a CPU) against the released code (split, STFT, model outputs, BatchNorm folding, unrolled LSTM,
weights and activations really on the integer grid, 2-bit control, determinism). Every line must say
PASS; the table is saved to `results/verify_toolkit.txt` for the report's appendix.

### 3. Cross-check the new quantizer against the Brevitas numbers (~10 min)

```bash
bash amcq/run.sh step3
```

Expect W8A8 ≈ 63.76, W6A6 ≈ 63.39, W4A4 ≈ 38 % (the 4-bit value varies a lot between models). Small
differences are expected because the LSTM is now quantized as well. This step is what lets you put
old and new results in one table.

### 4. Whole model, LSTM included (RQ1, H1) (~20 min)

```bash
bash amcq/run.sh step4
```

Runs the whole-model preset with 3 calibration sets. Output: `results/step4_whole_calib_summary.csv`
(per precision: mean over models and calibration sets, change against FP32, largest calibration range),
`results/step4_whole_calib_detail.csv`, and per calibration set `results/step4_whole_calib<c>/ptq_runs.csv`
(per model, per-SNR deltas, McNemar).

### 5. One block family at a time (RQ2, H2) (several hours: run it in the background)

```bash
nohup bash amcq/run.sh step5 > step5.log 2>&1 &
tail -f step5.log                  # Ctrl+C stops watching, not the job
grep -c " delta " step5.log        # progress: finished evaluations, out of 504
```

3 calibration sets × 3 models × 56 settings, like step 4. For a quick first look with one calibration
set: `CALIBS=0 bash amcq/run.sh step5`. Output: `results/step5_blocks_calib_summary.csv` and
`results/step5_fine_calib_summary.csv`.

Same design as before (W+A, W only, A only) but now the LSTM also gets activation quantization, so the
H2 comparison is fair. `--preset fine` splits the CONV family into stems / fusion / conv encoders /
local representation / feed-forward to find which part carries the 4-bit activation damage.

### 5b. Which activations must stay at 8 bits (RQ3) (about as long as step 4)

```bash
bash amcq/run.sh step5b
```

4-bit weights everywhere; activations go from 4 to 8 bits one group at a time, in the order of Step 5's
loss per share of compute (fusion layer, stems + classifier, attention, feed-forward, conv encoders,
LSTM), plus the literature rule alone (first and last layer at 8 bits). 3 calibration sets. Also writes
`results/model_cost.csv` (weights, MACs and activation values per group) and
`results/model_cost_specs.csv` (share of MACs still at 4-bit activations for each spec), from
`amcq/cost.py`, which needs no data. Output: `results/step5b_a8path_calib_summary.csv`.

### 5c. Step 5b with clipped activation scales (after step 6) (about as long as step 4)

```bash
bash amcq/run.sh step5c
```

Same path as 5b, but activation scales are set at the 99.9th percentile instead of min-max (the fix
step 6 found). Output: `results/step5c_a8path_p999_calib_summary.csv`.

### 5d. Clip only the 4-bit activations (about as long as step 4)

```bash
bash amcq/run.sh step5d
```

Same path again with `--calib-method auto`: 4-bit activations clipped at the 99.9th percentile, 8-bit
ones min-max (step 5c showed clipping helps at 4 bits but costs ~0.35 pp at 8 bits). Self-check: its
`all=W4A4` row must equal step 5c and its `all=W4A8` row step 5b. Output:
`results/step5d_a8path_auto_calib_summary.csv`.

### 6. Why 4 bits collapse (error analysis) (about 1 h)

```bash
bash amcq/run.sh step6
```

Output in `results/step6/`:

* `activation_ranges_<run>.csv`: per layer, max vs 99.9th/99th percentile and the number of the 16 levels
  the bulk of the values uses.
* `channel_ranges_<run>.csv`: per layer, how unequal the channel ranges are under one shared scale.
* `fusion_branches.csv`: range of the IQ and STFT halves of the fusion layer's input, and how many of the
  16 levels each half gets.
* `step6_summary.csv`: accuracy tests over 3 models x 3 calibration sets, covering:
  * min-max vs percentile clipping;
  * fusion layer with one scale per branch or per channel;
  * whole model with one scale per fusion branch (deployable) or per channel (upper bound).

If clipping recovers the loss, outliers are the cause. If separate scales recover it, a shared scale over
unequal channels is the cause. The script checks itself: its fusion min-max row must equal Step 5's
`FUSION=W32A4` exactly (it prints PASS).

### 7. Different models (rubric) (~5–7 h, run overnight)

```bash
nohup bash amcq/run.sh step7 > step7.log 2>&1 &
```

Trains the IQFormer paper's baselines from the repository: FEA-T (transformer), MCLDNN (CNN + LSTM),
PET-CGDNN (CNN + GRU) and AMC-Net (CNN + attention). Each is trained 3 times with the same split and
recipe, then quantized at W8A8 / W6A6 / W4A8 / W4A4. This gives the model comparison **and** shows
whether IQFormer is more or less robust to quantization than the other models. PET-CGDNN's GRU is not
quantized (only LSTMs are unrolled), so it stays in floating point. To add an MLP and a VGG-style CNN:
`MODELS7="mlp cnn feat mcldnn petcgdnn amcnet" bash amcq/run.sh step7`.

### 7b. Retrain MCLDNN (about 1 h)

```bash
nohup bash amcq/run.sh step7b > step7b.log 2>&1 &
```

In step 7 MCLDNN stayed at chance (9.09 %). The likely cause: with the raw, very small input values and no
normalization layer it never left the starting plateau, and early stopping ended it after 10 epochs
(check with `head -12 runs/mcldnn_s1/log.csv`). Step 7b retrains it
with `--input-norm` (every frame scaled to unit RMS, like a receiver's automatic gain control) and
`--patience 20`, then quantizes it as in step 7. `MODELS7B="mcldnn petcgdnn"` also redoes PET-CGDNN.

### 7c. Fair 4-bit comparison (about 1 h)

```bash
nohup bash amcq/run.sh step7c > step7c.log 2>&1 &
```

Quantizes every comparison model again with the rule that worked best for IQFormer (`--calib-method auto`,
step 5d) and 3 calibration sets, so their 4-bit numbers can be compared with IQFormer's 58.94 % fairly.
Output: `results/step7c_<model>_calib_summary.csv`.

### 8. Two more IQFormer seeds (~1 h)

```bash
bash amcq/run.sh step8
```

Checks that `train.py` reproduces the baseline (both inside 63.79 ± ~0.6 %) and gives 5 baseline seeds.

### 9. Proposed model — accuracy arms (~10 h, overnight)

```bash
nohup bash amcq/run.sh step9 > step9.log 2>&1 &
```

5 seeds each: rotation augmentation, rotation+flip augmentation, ReLU instead of GELU, hard-swish
instead of GELU. One change per arm, so each effect is measured alone (ablation).

### 10. Proposed model — quantization arms (~6–9 h)

```bash
nohup bash amcq/run.sh step10 > step10.log 2>&1 &
```

LSQ quantization-aware training at W4A4, with and without distillation from the three FP32 models, and a
mixed version (stems and classifier at 8 bits). Re-run on the best FP32 arm of step 9 by replacing
`runs/iqformer_s$s` in `run.sh` with that arm's run folders.

### 11. Latency (5 min)

```bash
bash amcq/run.sh step11
```

### 12. Tables and figures

```bash
bash amcq/run.sh step12
```

`results/analysis/summary_models.md` (mean ± std per model/variant), accuracy-vs-SNR, per-class and
confusion-matrix figures (cf. IQFormer paper Figs. 6–7), PTQ bar charts.

## Running one script directly

```bash
python amcq/train.py --help
python amcq/train.py --model iqformer --aug rot --seed 1 --out runs/iqf_rot_s1
python amcq/ptq.py --runs runs/iqformer_s1 runs/iqformer_s2 runs/iqformer_s3 --preset custom --specs "CONV=W4A8,ATTN=W4A6,LSTM=W8A8,HEAD=W8A8" --out results/my_mix
python amcq/qat.py --run runs/iqformer_s1 --spec all=W4A4 --teacher runs/iqformer_s1 runs/iqformer_s2 runs/iqformer_s3 --out runs/qatkd_s1
python amcq/train.py --model iqformer --stft cplx_logmag --seed 1 --out runs/iqf_cplxwide_s1   # the earlier 'cplxwide' input
```

Precision specs: `all=W8A8`, `CONV=W4A8,LSTM=W8A8,...`; families `CONV ATTN LSTM HEAD`, fine groups
`STEM FUSION CONVENC LOCAL FFN`; `A32`/`W32` = leave in floating point.

## Quick smoke test (2 minutes, before a long run)

```bash
EPOCHS=1 SUBSET=0.05 bash amcq/run.sh step8     # writes runs/iqformer_s4/5 — delete them afterwards
rm -rf runs/iqformer_s4 runs/iqformer_s5
```

## Files

| File | Purpose |
|---|---|
| `amcq/data.py` | released split (seed 233), GPU STFT (all input variants), I/Q augmentation |
| `amcq/models.py` | IQFormer wrapper (+ options), MLP, CNN, repository baselines |
| `amcq/quant.py` | fake-quant layers, `QLSTM`, BN folding, calibration, LSQ, per-branch / per-channel activation scales |
| `amcq/metrics.py` | per-SNR accuracy, macro F1, confusion, McNemar, TOST |
| `train.py`, `ptq.py`, `qat.py`, `act_stats.py`, `latency.py`, `analyze.py` | the experiments |
| `cost.py` | weights, MACs and activation values per block group; cost of a precision spec |
| `calib_summary.py` | combines results from several calibration sets (mean, range) |
| `run.sh` | steps 1–12 above (with 2b verify, 3b, 3c, 5b) |
| `check_env.py` | step 1 |
| `tests/make_synthetic.py` | fake dataset for testing the code without the real data |

Every result folder keeps the exact command-line arguments (`metrics.json → args`), so each number in
the report can be traced to the run that produced it. Commit `runs/*/metrics.json` and `results/` CSVs
(not the `.pt` files) to the `phase3` branch.
