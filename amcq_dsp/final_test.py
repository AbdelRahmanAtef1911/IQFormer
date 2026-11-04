"""One held-out test evaluation of a chosen run (same arguments as amcq_arch.final_arch_test). Use once, after
the variant has been chosen on validation."""
import runpy
import amcq_dsp  # noqa: F401

if __name__ == '__main__':
    runpy.run_module('amcq_arch.final_arch_test', run_name='__main__')
