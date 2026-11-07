"""Guarded rotation TTA for amcq_next runs (validation by default; --final-test once at the end).

  python -m amcq_next.tta --runs runs_next/e3_phase_s1 ... runs_next/e3_phase_s5
  python -m amcq_next.tta --runs ... --methods guard mean --final-test
Same script as amcq_dsp.tta_detail, with the iqfx_* model names registered.
"""
import amcq_next  # noqa: F401
from amcq_dsp.tta_detail import main

if __name__ == '__main__':
    main()
