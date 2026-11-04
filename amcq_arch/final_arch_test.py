"""One final held-out evaluation after variant selection; do not use for tuning."""
import argparse
from pathlib import Path
import numpy as np
import torch
from amcq import setup_repo, default_data
from amcq.data import load_rml2016a
from amcq.metrics import predict, summarize, save_json
from amcq_arch.ptq import load_run

p=argparse.ArgumentParser()
p.add_argument('--run',required=True)
p.add_argument('--repo',default=None)
p.add_argument('--data',default=None)
p.add_argument('--out',required=True)
p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
a=p.parse_args()
repo=setup_repo(a.repo)
d=load_rml2016a(a.data or default_data(repo))
model=load_run(a.run,torch.device(a.device),a)
logits=predict(model,torch.from_numpy(d['Xte']).to(a.device))
pred=logits.argmax(1).numpy()
result=summarize(pred,d['yte'],d['ste'])
result['transition_band']=float(np.mean([result['per_snr'][s] for s in (-6,-4,-2,0)]))
result['run']=a.run
Path(a.out).mkdir(parents=True,exist_ok=True)
save_json(result,str(Path(a.out)/'final_test_metrics.json'))
np.savez_compressed(Path(a.out)/'final_test_pred.npz',pred=pred,y=d['yte'],snr=d['ste'])
print('TEST overall',100*result['overall'],'band',100*result['transition_band'])
