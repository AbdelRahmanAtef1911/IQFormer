"""Run amcq_arch.qat with the DSP models registered. Same arguments as amcq_arch.qat."""
import sys
import amcq_dsp  # noqa: F401  (registers iqf_cum, iqf_pls, iqf_dsp, iqf_ms_dsp)
from amcq_arch.qat import main

if __name__ == '__main__':
    main()
