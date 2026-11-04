"""DSP-feature experiments for IQFormer (higher-order cumulants, power-law spectra, test-time augmentation).

Installed beside amcq/ and amcq_arch/ in the IQFormer repository root. Nothing in amcq/, amcq_arch/ or model/
is modified: importing this package registers the new model names with amcq_arch.models.build_model, so the
existing amcq_arch train / PTQ / QAT / final-test scripts can build and reload the new models.
"""
from pathlib import Path
import sys

_repo = Path(__file__).resolve().parent.parent
_toolkit = _repo / 'amcq'
if (_toolkit / 'amcq' / '__init__.py').is_file() and str(_toolkit) not in sys.path:
    sys.path.insert(0, str(_toolkit))

import amcq_arch.models as _arch_models            # noqa: E402
from .models import build_model as _dsp_build      # noqa: E402

if getattr(_arch_models.build_model, '__name__', '') != 'dsp_build_model':
    _arch_models._original_arch_build = _arch_models.build_model
    _arch_models.build_model = _dsp_build
