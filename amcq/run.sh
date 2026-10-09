#!/usr/bin/env bash
# Phase 3 / final-report experiments, one step at a time.
#   cd ~/amc/IQFormer && source ~/amc/.venv/bin/activate
#   bash amcq/run.sh step2          (then check the output before the next step)
# Long steps: run them in the background and watch the log, e.g.
#   nohup bash amcq/run.sh step7 > logs_step7.txt 2>&1 &     tail -f logs_step7.txt
set -euo pipefail
export PYTHONUNBUFFERED=1                     # print lines immediately, also into nohup logs
cd "$(dirname "$0")/.."                       # repository root (~/amc/IQFormer)

EPOCHS=${EPOCHS:-60}                          # QUICK smoke test: EPOCHS=1 SUBSET=0.05
SUBSET=${SUBSET:-1.0}
QAT_EPOCHS=${QAT_EPOCHS:-15}
CK1=${CK1:-save_models/model_2016.10a_60_256_0.001_IQFormer/weight.pt}
CK2=${CK2:-save_models/model_2016.10a_60_256_0.001_IQFormer_r2/weight.pt}
CK3=${CK3:-save_models/model_2016.10a_60_256_0.001_IQFormer_r3/weight.pt}
BASE=${BASE:-"runs/iqformer_s1 runs/iqformer_s2 runs/iqformer_s3"}
T="python amcq/train.py --epochs $EPOCHS --subset $SUBSET"
# mixed precision for QAT: stems, fusion layer and classifier at 8 bits (0.5 % of MACs; Steps 5, 5d: the cheapest groups with the largest loss per MAC)
MIXED=${MIXED:-"all=W4A4,STEM=W8A8,FUSION=W8A8,HEAD=W8A8"}
# comparison models for steps 7 and 11: the IQFormer paper's baselines from the repository
# (FEA-T = transformer; MCLDNN, PET-CGDNN, AMC-Net = convolutional hybrids). Add "mlp cnn" to include those.
MODELS7=${MODELS7:-"feat mcldnn petcgdnn amcnet"}

step1() {   # environment, GPU, dataset, STFT equivalence
  python amcq/check_env.py
}

verify() {  # independent proof that amcq computes what it claims (18 checks on a GPU, PASS/FAIL table)
  python amcq/tests/verify_toolkit.py --ckpt "$CK1"
}

step2() {   # import the three verified FP32 checkpoints as run folders (evaluation only)
  python amcq/train.py --init "$CK1" --epochs 0 --out runs/iqformer_s1
  python amcq/train.py --init "$CK2" --epochs 0 --out runs/iqformer_s2
  python amcq/train.py --init "$CK3" --epochs 0 --out runs/iqformer_s3
  echo ">>> CHECK: overall must be 63.44 / 63.91 / 64.02 % (the verified reproduction)."
}

step3() {   # the new quantizer, set up like the earlier Brevitas runs, must agree with them
  python amcq/ptq.py --runs $BASE --preset custom --specs all=W8A8 all=W6A6 all=W4A4 \
         --act-scheme symmetric --no-fold --calib-method percentile --pct 99.999 --out results/step3_brevitas_like
  echo ">>> CHECK against the Brevitas table: W8A8 ~63.76, W6A6 ~63.39, W4A4 ~38 (+-8) %."
  echo "    Small differences are expected: here the LSTM is ALSO quantized (it was skipped before)."
}

step3b() {  # why is W4A4 lower than with Brevitas? LSTM quantized now, and 640 bare parameters rounded now.
            # 2 x 2 design: LSTM quantized or float x bare parameters rounded or float.
  local B="--act-scheme symmetric --no-fold --calib-method percentile --pct 99.999"
  python amcq/ptq.py --runs $BASE --preset custom --specs all=W4A4 "all=W4A4,LSTM=W32A32" $B \
         --out results/step3b_bare_rounded
  python amcq/ptq.py --runs $BASE --preset custom --specs all=W4A4 "all=W4A4,LSTM=W32A32" $B --no-bare \
         --out results/step3b_bare_float
  echo ">>> 'all=W4A4,LSTM=W32A32' with --no-bare is the Brevitas setting: it should give ~38 %."
}

step3c() {  # is 4-bit PTQ ill-conditioned? same models, same settings, 5 different calibration sets
  local B="--act-scheme symmetric --no-fold --calib-method percentile --pct 99.999 --no-bare"
  for c in 0 1 2 3 4; do
    python amcq/ptq.py --runs $BASE --preset custom --specs all=W8A8 "all=W4A4,LSTM=W32A32" $B \
           --calib-seed $c --out results/step3c_calib$c
  done
  python amcq/calib_summary.py 'results/step3c_calib*'
  echo ">>> range_pp = how much the accuracy moves when only the 1,024 calibration frames change."
}

step4() {   # RQ1 + H1: whole model, LSTM included, integer-style scheme (BN folded, min-max calibration),
            # each with 3 calibration sets, because Step 3c showed 4-bit results move ~5 pp with the calibration set
  for c in 0 1 2; do
    python amcq/ptq.py --runs $BASE --preset whole --calib-seed $c --out results/step4_whole_calib$c
  done
  python amcq/calib_summary.py 'results/step4_whole_calib*'
}

step5() {   # RQ2 + H2: one block family at a time (W+A, W only, A only), then the CONV family split.
            # Same 3 calibration sets as step 4 (4-bit activations move with the calibration set).
            # Quick first look with one set:  CALIBS=0 bash amcq/run.sh step5
  local CALIBS=${CALIBS:-"0 1 2"}
  for c in $CALIBS; do
    python amcq/ptq.py --runs $BASE --preset blocks --calib-seed $c --out results/step5_blocks_calib$c
    python amcq/ptq.py --runs $BASE --preset fine   --calib-seed $c --out results/step5_fine_calib$c
  done
  python amcq/calib_summary.py 'results/step5_blocks_calib*'
  python amcq/calib_summary.py 'results/step5_fine_calib*'
  echo ">>> paste results/step5_blocks_calib_summary.csv and results/step5_fine_calib_summary.csv"
}

step5b() {  # RQ3: which activations must stay at 8 bits? 4-bit weights everywhere; activations raised to 8 bits
            # group by group in the order of Step 5's loss per share of compute, plus the literature rule
            # (first + last layer at 8 bits). Same 3 calibration sets.
  local CALIBS=${CALIBS:-"0 1 2"}
  python amcq/cost.py --out results/model_cost.csv
  for c in $CALIBS; do
    python amcq/ptq.py --runs $BASE --preset a8path --calib-seed $c --out results/step5b_a8path_calib$c
  done
  python amcq/calib_summary.py 'results/step5b_a8path_calib*'
  echo ">>> paste results/step5b_a8path_calib_summary.csv and results/model_cost_specs.csv"
}

step5c() {  # Step 5b again with the fix Step 6 found: activation scales clipped at the 99.9th percentile
  local CALIBS=${CALIBS:-"0 1 2"}
  for c in $CALIBS; do
    python amcq/ptq.py --runs $BASE --preset a8path --calib-method percentile --pct 99.9 --calib-seed $c \
           --out results/step5c_a8path_p999_calib$c
  done
  python amcq/calib_summary.py 'results/step5c_a8path_p999_calib*'
  echo ">>> paste results/step5c_a8path_p999_calib_summary.csv"
}

step5d() {  # Step 5b again with the calibration rule Steps 5c/6 point to: clip (99.9th percentile) only the 4-bit
            # activations, min-max for the 8-bit ones. Self-check: all=W4A4 must equal step 5c, all=W4A8 step 5b.
  local CALIBS=${CALIBS:-"0 1 2"}
  for c in $CALIBS; do
    python amcq/ptq.py --runs $BASE --preset a8path --calib-method auto --pct 99.9 --calib-seed $c \
           --out results/step5d_a8path_auto_calib$c
  done
  python amcq/calib_summary.py 'results/step5d_a8path_auto_calib*'
  echo ">>> paste results/step5d_a8path_auto_calib_summary.csv"
}

step6() {   # error analysis of the 4-bit collapse: ranges per layer / channel / fusion branch, clipping,
            # one scale per fusion branch, one scale per channel. Every model x 3 calibration sets.
  python amcq/act_stats.py --runs $BASE --calib-seeds ${CALIBS:-0 1 2} --out results/step6
  echo ">>> paste results/step6/step6_summary.csv and results/step6/fusion_branches.csv"
}

step7() {   # rubric: 'different models such as MLP, CNN, and transformer' (+ the IQFormer paper's baselines)
  for m in $MODELS7; do
    for s in 1 2 3; do $T --model $m --seed $s --out runs/${m}_s$s; done
  done
  for m in $MODELS7; do
    python amcq/ptq.py --runs runs/${m}_s1 runs/${m}_s2 runs/${m}_s3 --preset custom \
           --specs all=W8A8 all=W6A6 all=W4A8 all=W4A4 --out results/step7_ptq_$m
  done
}

step7b() {  # MCLDNN did not train in step 7 (stuck at chance, 9.09 %). Retrain it with per-frame power
            # normalization (--input-norm) and longer early-stopping patience, then quantize as in step 7.
            # MODELS7B="mcldnn petcgdnn" also redoes PET-CGDNN (one of its runs was weak, 51.25 %).
  local M=${MODELS7B:-mcldnn}
  for m in $M; do
    for s in 1 2 3; do $T --model $m --seed $s --input-norm --patience 20 --out runs/${m}_norm_s$s; done
    python amcq/ptq.py --runs runs/${m}_norm_s1 runs/${m}_norm_s2 runs/${m}_norm_s3 --preset custom \
           --specs all=W8A8 all=W6A6 all=W4A8 all=W4A4 --out results/step7b_ptq_${m}_norm
  done
  echo ">>> paste: grep \"\\[test\\]\" of the log, and cat results/step7b_ptq_*/ptq_summary.csv"
}

step7c() {  # fair 4-bit comparison: every comparison model with the calibration rule that worked best for
            # IQFormer (auto: clip 4-bit activations at the 99.9th percentile, min-max for wider ones), 3 calibration sets
  local CALIBS=${CALIBS:-"0 1 2"}
  for m in ${MODELS7C:-feat mcldnn_norm petcgdnn amcnet}; do
    for c in $CALIBS; do
      python amcq/ptq.py --runs runs/${m}_s1 runs/${m}_s2 runs/${m}_s3 --preset custom \
             --specs all=W8A8 all=W6A6 all=W4A8 all=W4A4 --calib-method auto --pct 99.9 --calib-seed $c \
             --out results/step7c_${m}_calib$c
    done
    python amcq/calib_summary.py "results/step7c_${m}_calib*"
  done
  echo '>>> paste: for f in results/step7c_*_calib_summary.csv; do echo $f; cat $f; done'
}

step8() {   # two more IQFormer seeds with this pipeline: checks train.py reproduces the baseline,
            # and gives 5 baseline seeds for the ablation (the literature varies by ~1 pp between papers)
  for s in 4 5; do $T --model iqformer --seed $s --out runs/iqformer_s$s; done
  echo ">>> CHECK: both should fall inside 63.79 +- ~0.6 %."
}

step9() {   # proposed model, accuracy part: one change per variant, 5 seeds each (about 20 trainings, ~10 h).
            # ARMS9="name:train options|..." picks the variants; SEEDS9 the seeds. Runs go to runs/iqf_<name>_s<seed>.
            # Already finished runs are skipped, so the step can be stopped and restarted.
  local ARMS=${ARMS9:-"rot:--aug rot|rotflip:--aug rot,flip|relu:--act relu|hswish:--act hswish"}
  local SEEDS=${SEEDS9:-"1 2 3 4 5"}
  local IFS_OLD=$IFS
  for s in $SEEDS; do
    IFS='|'; for arm in $ARMS; do IFS=$IFS_OLD
      local name=${arm%%:*} opts=${arm#*:}
      if [ -f runs/iqf_${name}_s$s/metrics.json ]; then echo "skip runs/iqf_${name}_s$s (done)"; continue; fi
      $T --model iqformer $opts --seed $s --out runs/iqf_${name}_s$s
    done; IFS=$IFS_OLD
  done
  step9s
}

step9s() {  # summary of step 9: each variant vs the five baseline seeds (Welch t-test), overall and in -6..0 dB
  python amcq/seeds_summary.py --base 'runs/iqformer_s[1-5]' --variants 'runs/iqf_*_s[0-9]' --out results/step9_summary.csv
  echo ">>> paste the table above (results/step9_summary.csv)"
}

step9b() {  # does a training change also make the model easier to quantize? PTQ of every step-9 variant and of
            # the 5 baseline seeds with the bit-width-aware rule (calibration set 0; ~1 h)
  python amcq/ptq.py --runs runs/iqformer_s1 runs/iqformer_s2 runs/iqformer_s3 runs/iqformer_s4 runs/iqformer_s5 \
         --preset custom --specs all=W8A8 all=W6A6 all=W4A8 all=W4A4 --calib-method auto --pct 99.9 --save-pred \
         --out results/step9b_ptq_iqformer                     # --save-pred: per-class analysis for the final report
  for d in runs/iqf_*_s1; do
    local name=$(basename ${d%_s1})
    python amcq/ptq.py --runs runs/${name}_s[0-9] --preset custom --specs all=W8A8 all=W6A6 all=W4A8 all=W4A4 \
           --calib-method auto --pct 99.9 --out results/step9b_ptq_${name}
  done
  echo '>>> paste: for f in results/step9b_ptq_*/ptq_summary.csv; do echo $f; cat $f; done'
}

step9c() {  # ReLU vs GELU, for the report: PTQ with calibration sets 1 and 2 (set 0 = step 9b), combined over the
            # 3 sets, and the activation ranges of the ReLU models (are their outliers smaller?) (~1 h)
  for m in iqformer iqf_relu; do
    rm -rf results/step9c_ptq_${m}_calib0 && cp -r results/step9b_ptq_${m} results/step9c_ptq_${m}_calib0
    for c in 1 2; do
      python amcq/ptq.py --runs runs/${m}_s1 runs/${m}_s2 runs/${m}_s3 runs/${m}_s4 runs/${m}_s5 --preset custom \
             --specs all=W8A8 all=W6A6 all=W4A8 all=W4A4 --calib-method auto --pct 99.9 --calib-seed $c \
             --out results/step9c_ptq_${m}_calib$c
    done
    python amcq/calib_summary.py "results/step9c_ptq_${m}_calib*"
  done
  python amcq/act_stats.py --runs runs/iqf_relu_s1 runs/iqf_relu_s2 runs/iqf_relu_s3 --calib-seeds 0 --out results/step6_relu
  echo '>>> paste: cat results/step9c_ptq_*_calib_summary.csv; cat results/step6_relu/step6_summary.csv'
}

step10() {  # proposed model, quantization part: LSQ quantization-aware training (QAT), ranges initialized with the
            # bit-width-aware rule (clip 4-bit activations at the 99.9th percentile). Configurations (QAT_CONFIGS):
            #   w4a4       W4A4, plain QAT
            #   kd_w4a4    W4A4 + distillation from the FP32 models in TEACH
            #   kd_mixed   W4A4 with 8-bit stems, fusion and classifier ($MIXED, 0.5 % of MACs) + distillation
            #   kd_w4a8    W4A8 + distillation (optional: can W4A8 reach full precision?)
            #   kd_fp32    control: the same fine-tuning + distillation with NO quantization (how much is distillation alone?)
            # QAT_RUNS: the FP32 models to start from (default: the 3 original IQFormer models).
            # Output: runs/qat_<config>_<run name>, e.g. runs/qat_kd_w4a4_iqformer_s1. Finished runs are skipped.
  local RUNS=${QAT_RUNS:-$BASE}
  local TEACH=${TEACH:-"runs/iqformer_s1 runs/iqformer_s2 runs/iqformer_s3 runs/iqformer_s4 runs/iqformer_s5"}
  for cfg in ${QAT_CONFIGS:-w4a4 kd_w4a4 kd_mixed}; do
    for r in $RUNS; do
      local out=runs/qat_${cfg}_$(basename $r)
      if [ -f $out/metrics.json ]; then echo "skip $out (done)"; continue; fi
      case $cfg in
        w4a4)     python amcq/qat.py --run $r --spec all=W4A4 --epochs $QAT_EPOCHS --out $out ;;
        kd_w4a4)  python amcq/qat.py --run $r --spec all=W4A4 --epochs $QAT_EPOCHS --teacher $TEACH --out $out ;;
        kd_mixed) python amcq/qat.py --run $r --spec "$MIXED" --epochs $QAT_EPOCHS --teacher $TEACH --out $out ;;
        kd_w4a8)  python amcq/qat.py --run $r --spec all=W4A8 --epochs $QAT_EPOCHS --teacher $TEACH --out $out ;;
        kd_fp32)  python amcq/qat.py --run $r --spec FP32 --epochs $QAT_EPOCHS --teacher $TEACH --out $out ;;
        *) echo "unknown QAT config $cfg"; exit 1 ;;
      esac
    done
  done
  step10s
}

step10s() { # summary of step 10
  python amcq/qat_summary.py 'runs/qat_*' --out results/step10_summary.csv
  echo ">>> paste the table above (results/step10_summary.csv)"
}

step11() {  # real-time argument: latency at batch size 1 (crashes in the released code)
  python amcq/latency.py --run runs/iqformer_s1 --out results/step11_latency_iqformer.json
  for m in $MODELS7; do python amcq/latency.py --model $m --out results/step11_latency_$m.json; done
}

step12() {  # tables and figures for the report
  python amcq/analyze.py --runs 'runs/*' --ptq results/step4_whole_calib0 results/step5_blocks_calib0 results/step5b_a8path_calib0 --detail runs/iqformer_s1 \
         --out results/analysis
}

"${1:-step1}"
