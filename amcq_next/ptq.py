"""amcq_arch.ptq with the iqfx_* model names registered (same arguments)."""
import runpy
import amcq_next  # noqa: F401

if __name__ == '__main__':
    runpy.run_module('amcq_arch.ptq', run_name='__main__')
