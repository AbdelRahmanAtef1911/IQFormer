"""Compare run groups on the VALIDATION set (Welch's t-test over seeds), from the saved val_pred.npz files.

  python -m amcq_next.compare --roots runs_arch runs_next
  python -m amcq_next.compare --roots runs_arch runs_next --baseline e1_cos100 --detail e2a_snr
Run folders are <name>_s<seed>. Same script as amcq_dsp.compare_val (primary metric -6..0 dB, guardrail overall).
"""
import sys
from amcq_dsp.compare_val import main

if __name__ == '__main__':
    if not any(x.startswith('--out') for x in sys.argv):
        sys.argv += ['--out', 'results_next/compare_val.csv']
    import os
    os.makedirs('results_next', exist_ok=True)
    main()
