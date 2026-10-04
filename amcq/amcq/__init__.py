"""amcq: experiments for 'Evaluating low-precision quantization of IQFormer' (ICS 590)."""
import os
import sys


def setup_repo(repo=None):
    """Make the authors' repository importable (model/IQFormer.py etc.).
    Default: the folder that contains this 'amcq' project folder, or $IQFORMER_REPO."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo = repo or os.environ.get('IQFORMER_REPO') or os.path.dirname(here)
    repo = os.path.abspath(repo)
    if not os.path.exists(os.path.join(repo, 'model', 'IQFormer.py')):
        raise SystemExit(f'IQFormer repository not found at {repo}. Put the amcq folder inside the '
                         f'repository (next to main.py) or pass --repo /path/to/IQFormer.')
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return repo


def default_data(repo):
    return os.path.join(repo, 'dataset', 'RML2016.10a_dict.pkl')
