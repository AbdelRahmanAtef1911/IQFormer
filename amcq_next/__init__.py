"""Next-round experiments for IQFormer, from the literature review (amcq_next).

Installed beside amcq/, amcq_arch/ and amcq_dsp/ in the IQFormer repository root. Nothing in those folders or
in model/ is modified. Importing this package registers the new 'iqfx_*' model names with
amcq_arch.models.build_model (after amcq_dsp's names, if amcq_dsp is present), so the existing PTQ / QAT /
final-test / TTA scripts can rebuild the new models from a run folder.
"""
from pathlib import Path
import sys

_repo = Path(__file__).resolve().parent.parent
_toolkit = _repo / 'amcq'
if (_toolkit / 'amcq' / '__init__.py').is_file() and str(_toolkit) not in sys.path:
    sys.path.insert(0, str(_toolkit))

try:                                                # keep the amcq_dsp names available too
    import amcq_dsp  # noqa: F401,E402
except ImportError:
    pass
import amcq_arch.models as _arch_models            # noqa: E402
from .models import build_model as _next_build     # noqa: E402

if getattr(_arch_models.build_model, '__name__', '') != 'next_build_model':
    _arch_models._before_next_build = _arch_models.build_model
    _arch_models.build_model = _next_build
