"""amcq_arch.qat with the iqfx_* model names registered (same arguments)."""
import runpy
import amcq_next  # noqa: F401

if __name__ == '__main__':
    runpy.run_module('amcq_arch.qat', run_name='__main__')
