"""Size and compute of every block group, and the cost of a precision spec (no dataset needed, CPU, seconds).

Per group: weights, multiply-accumulates (MACs) per frame and activation values per frame that enter a
quantized matrix product (counted from the layer shapes of one forward pass).
Per spec: the share of MACs whose input activations are 4-bit or lower, and the weight memory.

  python amcq/cost.py                                   # IQFormer, group table + the 'a8path' specs
  python amcq/cost.py --specs "all=W4A4,FUSION=W4A8" --out results/model_cost.csv
"""
import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amcq import setup_repo                                    # noqa: E402


def group_costs(model):
    from amcq.quant import quantize_model, QConv, QLinear, QLSTM
    q = quantize_model(model, 'all=W8A8')
    cost, hooks = {}, []

    def add(g, w, mac, act):
        c = cost.setdefault(g, dict(weights=0, macs=0, act_values=0, layers=0))
        c['weights'] += w; c['macs'] += mac; c['act_values'] += act; c['layers'] += 1

    for name, mod in q.named_modules():
        if isinstance(mod, QConv):
            def h(m, inp, out):
                w = m.conv.weight
                add(m.group, w.numel(), out[0].numel() * w[0].numel(), inp[0][0].numel())
            hooks.append(mod.register_forward_hook(h))
        elif isinstance(mod, QLinear):
            def h(m, inp, out):
                w = m.lin.weight
                add(m.group, w.numel(), inp[0][0].numel() // w.shape[1] * w.numel(), inp[0][0].numel())
            hooks.append(mod.register_forward_hook(h))
        elif isinstance(mod, QLSTM):
            def h(m, inp, out):
                T = inp[0].shape[1]
                nw = sum(p.numel() for n, p in m.lstm.named_parameters() if n.startswith('weight'))
                acts = inp[0][0].numel() + (m.L - 1) * T * m.H * m.D + m.L * m.D * T * m.H   # x_t of each layer + h
                add('LSTM', nw, T * nw, acts)
            hooks.append(mod.register_forward_hook(h))
    with torch.no_grad():
        q(torch.randn(1, 2, 128))
    for hk in hooks:
        hk.remove()
    return cost


def spec_cost(cost, spec):
    from amcq.quant import parse_spec, bits_for
    sp = parse_spec(spec)
    tot = sum(c['macs'] for c in cost.values())
    a4 = sum(c['macs'] for g, c in cost.items() if (bits_for(g, sp)[1] or 32) <= 4)
    wbits = sum(c['weights'] * (bits_for(g, sp)[0] or 32) for g, c in cost.items())
    return dict(spec=spec, macs_a4_pct=100 * a4 / tot, macs_a8plus_pct=100 * (tot - a4) / tot,
                weight_kB=wbits / 8 / 1024)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--repo', default=None)
    p.add_argument('--model', default='iqformer')
    p.add_argument('--specs', nargs='*', default=None, help="default: the 'a8path' preset of ptq.py")
    p.add_argument('--out', default=None, help='CSV file for the group table (the spec table goes next to it)')
    args = p.parse_args(argv)
    setup_repo(args.repo)
    from amcq.models import build_model
    from ptq import PRESETS
    torch.manual_seed(0)
    cost = group_costs(build_model(args.model).eval())
    tot = {k: sum(c[k] for c in cost.values()) for k in ('weights', 'macs', 'act_values', 'layers')}
    rows = [dict(group=g, **c, weights_pct=100 * c['weights'] / tot['weights'], macs_pct=100 * c['macs'] / tot['macs'],
                 act_pct=100 * c['act_values'] / tot['act_values']) for g, c in cost.items()]
    print(f"{'group':8s} {'layers':>6s} {'weights':>8s} {'%':>6s} {'MACs/frame':>11s} {'%':>6s} {'act values':>10s} {'%':>6s}")
    for r in rows:
        print(f"{r['group']:8s} {r['layers']:6d} {r['weights']:8d} {r['weights_pct']:6.1f} {r['macs']:11d} "
              f"{r['macs_pct']:6.1f} {r['act_values']:10d} {r['act_pct']:6.1f}")
    print(f"{'total':8s} {tot['layers']:6d} {tot['weights']:8d} {'':6s} {tot['macs']:11d} {'':6s} {tot['act_values']:10d}")
    specs = args.specs if args.specs is not None else PRESETS['a8path']
    srows = [spec_cost(cost, s) for s in specs]
    print(f"\n{'spec':70s} {'MACs A4 %':>9s} {'weights kB':>10s}")
    for r in srows:
        print(f"{r['spec']:70s} {r['macs_a4_pct']:9.1f} {r['weight_kB']:10.1f}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        sp = os.path.splitext(args.out)[0] + '_specs.csv'
        with open(sp, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(srows[0])); w.writeheader(); w.writerows(srows)
        print(f'\nwrote {args.out} and {sp}')


if __name__ == '__main__':
    main()
