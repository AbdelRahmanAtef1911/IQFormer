"""Latency and throughput of a model (the real-time argument). Includes the GPU STFT.

Batch size 1 is the real-time case (one frame classified as soon as it arrives); it crashes in the
released code because of torch.squeeze() and works here (squeeze(dim=2)).

Example
  python amcq/latency.py --run runs/iqformer_s1 --out results/latency_iqformer.json
  python amcq/latency.py --model mlp --out results/latency_mlp.json
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amcq import setup_repo                                     # noqa: E402


def timeit(model, x, n, dev):
    ts = []
    with torch.no_grad():
        for _ in range(n):
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    ts.sort()
    return dict(median_ms=1e3 * statistics.median(ts), p99_ms=1e3 * ts[int(0.99 * (len(ts) - 1))])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--repo', default=None)
    p.add_argument('--run', default=None)
    p.add_argument('--model', default='iqformer')
    p.add_argument('--out', required=True)
    args = p.parse_args(argv)
    setup_repo(args.repo)
    from amcq.models import build_model
    res = {}
    devs = [torch.device('cpu')] + ([torch.device('cuda')] if torch.cuda.is_available() else [])
    for dev in devs:
        if args.run:
            from ptq import load_run
            model = load_run(args.run, dev, args)
        else:
            model = build_model(args.model).to(dev).eval()
        if dev.type == 'cpu':
            torch.set_num_threads(1)
        for bs in (1, 64, 400):
            x = torch.randn(bs, 2, 128, device=dev)
            n_warm, n = (20, 200) if dev.type == 'cuda' or bs == 1 else (3, 10)
            try:        # some repository models fail here: FEA-T hard-codes .cuda() (no CPU run), and models that
                        # call torch.squeeze() without a dim (released IQFormer, PET-CGDNN) crash at batch size 1
                timeit(model, x, n_warm, dev)
            except Exception as e:
                res[f'{dev.type}_bs{bs}'] = {'error': f'{type(e).__name__}: {str(e).splitlines()[0]}'}
                print(f'{dev.type:4s} batch {bs:4d}: FAILED ({res[f"{dev.type}_bs{bs}"]["error"]})')
                continue
            r = timeit(model, x, n, dev)
            r['per_frame_ms'] = r['median_ms'] / bs
            r['frames_per_s'] = 1e3 * bs / r['median_ms']
            res[f'{dev.type}_bs{bs}'] = r
            print(f'{dev.type:4s} batch {bs:4d}: {r["median_ms"]:8.3f} ms/batch (p99 {r["p99_ms"]:.3f})  '
                  f'{r["per_frame_ms"]:.4f} ms/frame  {r["frames_per_s"]:.0f} frames/s')
    res['note'] = 'CPU measured with 1 thread. GPU timings include the STFT and host synchronisation.'
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
