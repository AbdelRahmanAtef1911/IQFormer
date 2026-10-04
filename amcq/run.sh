#!/usr/bin/env bash
# Phase 3 / final-report experiments, one step at a time.
#   cd ~/amc/IQFormer && source ~/amc/.venv/bin/activate
#   bash amcq/run.sh step2          (then check the output before the next step)
# Long steps: run them in the background and watch the log, e.g.
#   nohup bash amcq/run.sh step7 > logs_step7.txt 2>&1 &     tail -f logs_step7.txt
set -euo pipefail
cd "$(dirname "$0")/.."                       # repository root (~/amc/IQFormer)

EPOCHS=${EPOCHS:-60}                          # QUICK smoke test: EPOCHS=1 SUBSET=0.05
SUBSET=${SUBSET:-1.0}
QAT_EPOCHS=${QAT_EPOCHS:-15}
CK1=${CK1:-save_models/model_2016.10a_60_256_0.001_IQFormer/weight.pt}
CK2=${CK2:-save_models/model_2016.10a_60_256_0.001_IQFormer_r2/weight.pt}
CK3=${CK3:-save_models/model_2016.10a_60_256_0.001_IQFormer_r3/weight.pt}
BASE=${BASE:-"runs/iqformer_s1 runs/iqformer_s2 runs/iqformer_s3"}
T="python amcq/train.py --epochs $EPOCHS --subset $SUBSET"
# mixed precision for QAT: stems, fusion layer and classifier at 8 bits (0.5 % of MACs; Step 5). Revisit after 5b/6.
MIXED=${MIXED:-"all=W4A4,STEM=W8A8,FUSION=W8A8,HEAD=W8A8"}

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

step6() {   # error analysis of the 4-bit collapse: ranges per layer / channel / fusion branch, clipping,
            # one scale per fusion branch, one scale per channel. Every model x 3 calibration sets.
  python amcq/act_stats.py --runs $BASE --calib-seeds ${CALIBS:-0 1 2} --out results/step6
  echo ">>> paste results/step6/step6_summary.csv and results/step6/fusion_branches.csv"
}

step7() {   # rubric: 'different models such as MLP, CNN, and transformer' (+ the IQFormer paper's baselines)
  for m in mlp cnn feat mcldnn petcgdnn amcnet; do
    for s in 1 2 3; do $T --model $m --seed $s --out runs/${m}_s$s; done
  done
  for m in mlp cnn feat mcldnn petcgdnn amcnet; do
    python amcq/ptq.py --runs runs/${m}_s1 runs/${m}_s2 runs/${m}_s3 --preset custom \
           --specs all=W8A8 all=W6A6 all=W4A8 all=W4A4 --out results/step7_ptq_$m
  done
}

step8() {   # two more IQFormer seeds with this pipeline: checks train.py reproduces the baseline,
            # and gives 5 baseline seeds for the ablation (the literature varies by ~1 pp between papers)
  for s in 4 5; do $T --model iqformer --seed $s --out runs/iqformer_s$s; done
  echo ">>> CHECK: both should fall inside 63.79 +- ~0.6 %."
}

step9() {   # proposed model, accuracy part (one change per arm, 5 seeds each)
  for s in 1 2 3 4 5; do
    $T --model iqformer --aug rot      --seed $s --out runs/iqf_rot_s$s
    $T --model iqformer --aug rot,flip --seed $s --out runs/iqf_rotflip_s$s
    $T --model iqformer --act relu     --seed $s --out runs/iqf_relu_s$s
    $T --model iqformer --act hswish   --seed $s --out runs/iqf_hswish_s$s
  done
}

step10() {  # proposed model, quantization part: LSQ QAT, with and without distillation
  for s in 1 2 3; do
    python amcq/qat.py --run runs/iqformer_s$s --spec all=W4A4 --epochs $QAT_EPOCHS --out runs/qat_w4a4_s$s
    python amcq/qat.py --run runs/iqformer_s$s --spec all=W4A4 --epochs $QAT_EPOCHS --teacher $BASE --out runs/qatkd_w4a4_s$s
    python amcq/qat.py --run runs/iqformer_s$s --spec "$MIXED" --epochs $QAT_EPOCHS \
           --teacher $BASE --out runs/qatkd_mixed_s$s
  done
}

step11() {  # real-time argument: latency at batch size 1 (crashes in the released code)
  python amcq/latency.py --run runs/iqformer_s1 --out results/step11_latency_iqformer.json
  for m in mlp cnn feat; do python amcq/latency.py --model $m --out results/step11_latency_$m.json; done
}

step12() {  # tables and figures for the report
  python amcq/analyze.py --runs 'runs/*' --ptq results/step4_whole_calib0 results/step5_blocks_calib0 results/step5b_a8path_calib0 --detail runs/iqformer_s1 \
         --out results/analysis
}

"${1:-step1}"
