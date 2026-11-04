"""Compare validation accuracy by matched seeds; no test data is read."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t
from amcq import setup_repo, default_data
from amcq.data import load_rml2016a, to_device
from amcq.metrics import predict
from amcq_arch.ptq import load_run

p=argparse.ArgumentParser()
p.add_argument('--repo',default=None)
p.add_argument('--data',default=None)
p.add_argument('--root',default='runs')
p.add_argument('--seeds',type=int,nargs='+',default=[1,2,3,4,5])
p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
a=p.parse_args()
repo=setup_repo(a.repo)
d=load_rml2016a(a.data or default_data(repo))
dev=torch.device(a.device); x=torch.from_numpy(d['Xva']).to(dev)
band=np.isin(d['sva'],[-6,-4,-2,0]); results={}
for variant in ('iqformer','iqf_ms','iqf_ms_gate','iqf_ms_snr'):
    rows=[]
    for seed in a.seeds:
        run=Path(a.root)/f'{variant}_s{seed}'
        if not (run/'best.pt').exists():
            raise FileNotFoundError(run/'best.pt')
        if variant=='iqformer':
            model=load_run(str(run),dev,a)
            pred=predict(model,x).argmax(1).numpy()
        else:
            with np.load(run/'val_pred.npz') as f: pred=f['pred']
        rows.append([100*(pred[band]==d['yva'][band]).mean(),100*(pred==d['yva']).mean()])
    results[variant]=np.array(rows)
    print(variant,'band/overall mean',results[variant].mean(0).round(3),'per-seed',results[variant].round(3).tolist())
base=results['iqformer']
for variant in ('iqf_ms','iqf_ms_gate','iqf_ms_snr'):
    delta=results[variant]-base
    half=t.ppf(.975,len(delta)-1)*delta.std(0,ddof=1)/np.sqrt(len(delta))
    print(variant,'paired improvement pp [band,overall]:',delta.mean(0).round(3),
          '95% CI',np.stack([delta.mean(0)-half,delta.mean(0)+half],axis=1).round(3).tolist())
print('Select one variant from these VALIDATION scores, then run final_arch_test.py once.')
