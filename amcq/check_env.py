"""Step 1 check: Python packages, a REAL GPU computation, the repository, the dataset, and that the GPU
STFT reproduces scipy's STFT used by the released code.

  python amcq/check_env.py            (from the IQFormer repository root)
"""
import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ok = True
for pkg in ('torch', 'numpy', 'scipy', 'sklearn', 'pandas', 'matplotlib', 'timm', 'einops', 'tqdm'):
    try:
        m = importlib.import_module(pkg)
        print(f'[ok]   {pkg:10s} {getattr(m, "__version__", "")}')
    except Exception as e:                                         # noqa: BLE001
        ok = False
        print(f'[MISS] {pkg:10s} -> pip install {pkg if pkg != "sklearn" else "scikit-learn"}  ({e})')

import torch                                                       # noqa: E402
if torch.cuda.is_available():
    try:
        a = torch.randn(4096, 4096, device='cuda')
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(10):
            b = a @ a
        torch.cuda.synchronize()
        tf = 10 * 2 * 4096 ** 3 / (time.time() - t0) / 1e12
        print(f'[ok]   GPU {torch.cuda.get_device_name(0)}: real 4096x4096 matmul ran, {tf:.1f} TFLOP/s fp32')
    except Exception as e:                                         # noqa: BLE001
        ok = False
        print(f'[FAIL] torch sees a GPU but a real kernel failed: {e}\n'
              f'       install a PyTorch build that supports this GPU (see README step 2).')
else:
    print('[warn] no CUDA GPU visible: everything runs on the CPU (very slow for training).')

from amcq import setup_repo, default_data                          # noqa: E402
repo = setup_repo(None)
print(f'[ok]   repository: {repo}')
data = default_data(repo)
print(('[ok]   dataset: ' if os.path.exists(data) else '[MISS] dataset not found: ') + data)
ok &= os.path.exists(data)

import numpy as np                                                 # noqa: E402
from scipy.signal import stft                                      # noqa: E402
from amcq.data import stft_features                                # noqa: E402
x = np.random.randn(8, 2, 128).astype(np.float32) * 0.01
ref = np.stack([stft(x[i, 0], 1.0, 'blackman', 31, 30, 128)[2][:32] for i in range(8)])
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
mine = stft_features(torch.from_numpy(x).to(dev), 'real')[:, 0].cpu().numpy()
err = np.abs(mine - ref.real).max() / np.abs(ref.real).max()
print(f'[{"ok" if err < 1e-4 else "FAIL"}]   GPU STFT vs scipy (released preprocessing): relative max error {err:.2e}')
ok &= err < 1e-4
print('\nALL CHECKS PASSED' if ok else '\nSOME CHECKS FAILED - fix them before training')
