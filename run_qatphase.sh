T="runs_next/e3b_phase_cos100_s1 runs_next/e3b_phase_cos100_s2 runs_next/e3b_phase_cos100_s3 runs_next/e3b_phase_cos100_s4 runs_next/e3b_phase_cos100_s5"
for s in 1 2 3; do
  python amcq/qat_phase.py --run runs_next/e3b_phase_cos100_s$s --spec all=W4A4 --epochs 15 --aug phase --teacher $T --out runs_qatphase/w4a4kd_phase_s$s
  python amcq/qat_phase.py --run runs_next/e3b_phase_cos100_s$s --spec "all=W4A4,STEM=W8A8,FUSION=W8A8,HEAD=W8A8" --epochs 15 --aug phase --teacher $T --out runs_qatphase/mixedkd_phase_s$s
done
echo ALL DONE
