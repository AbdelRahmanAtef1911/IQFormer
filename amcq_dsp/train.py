"""Run amcq_arch.train with the DSP models registered. Same arguments as amcq_arch.train."""
import sys
import amcq_dsp  # noqa: F401  (registers iqf_cum, iqf_pls, iqf_dsp, iqf_ms_dsp)
from amcq_arch.train import main

if __name__ == '__main__':
    argv = sys.argv[1:]
    if '--final-test' not in argv and '--validation-only' not in argv:
        argv.append('--validation-only')        # never touch the held-out test set while choosing a variant
    main(argv)
