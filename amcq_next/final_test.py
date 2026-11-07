"""One held-out test evaluation of a chosen run (same arguments as amcq_arch.final_arch_test). Use ONCE, after
the choice has been made on validation. Refuses the oracle diagnostic."""
import runpy
import sys
import amcq_next  # noqa: F401

if __name__ == '__main__':
    if any('oracle' in a for a in sys.argv):
        raise SystemExit('the oracle model is a diagnostic and never runs on the test set')
    runpy.run_module('amcq_arch.final_arch_test', run_name='__main__')
