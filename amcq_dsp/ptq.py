"""Run amcq_arch.ptq with the DSP models registered. Same arguments as amcq_arch.ptq."""
import sys
import amcq_dsp  # noqa: F401  (registers iqf_cum, iqf_pls, iqf_dsp, iqf_ms_dsp)
from amcq_arch.ptq import main

if __name__ == '__main__':
    main()
