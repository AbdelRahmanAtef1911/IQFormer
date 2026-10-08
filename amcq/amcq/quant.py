"""Simulated (fake) integer quantization for IQFormer and the baselines, written in plain PyTorch.

Scheme (what an integer FPGA datapath computes):
  * weights: symmetric, per output channel, signed b-bit  (q in [-(2^(b-1)-1), 2^(b-1)-1])
  * activations: affine (scale + zero point), per tensor, unsigned b-bit, quantized at the INPUT of
    every convolution / linear layer and of every LSTM matrix product (x_t and h_{t-1})
  * batch normalization that follows a convolution is folded into it (exact); batch normalization
    that precedes a layer stays a per-channel affine step in front of the next quantizer
  * bias, accumulation, sigmoid/tanh/GELU/L2-normalization are kept in floating point (in hardware:
    32-bit accumulators and look-up tables or integer approximations)
  * the bare parameters layer_scale and w_g are rounded to the weight precision of their block

Unlike module-substitution libraries, the LSTM is fully covered: QLSTM unrolls nn.LSTM with the same
weights, so its weights and its matrix-product inputs (including the recurrent state) are quantized.

Also provides LSQ learnable step sizes (Esser et al., ICLR 2020) with the straight-through estimator
for quantization-aware training.
"""
import copy
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F


def round_ste(x):
    return (x.round() - x).detach() + x


def grad_scale(x, g):
    return (x - x * g).detach() + x * g


# ----------------------------------------------------------------------------- quantizers
class ActQuant(nn.Module):
    """Per-tensor activation quantizer with a calibration observer and optional LSQ scale.
    scheme 'affine'   : unsigned b-bit with zero point (default)
    scheme 'symmetric': signed b-bit, zero point 0 (as Brevitas Int8ActPerTensorFloat, used in the
                        project's earlier Brevitas runs)"""

    def __init__(self, bits=None, scheme='affine'):
        super().__init__()
        self.bits = bits
        self.scheme = scheme
        self.mode = 'off'                      # off | observe | quant
        self.pct = 99.99                       # percentile used by 'percentile' calibration
        self.register_buffer('lo', torch.tensor(float('inf')))
        self.register_buffer('hi', torch.tensor(float('-inf')))
        self.batch_stats = []                  # one entry per observed call, computed on the FULL tensor
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.register_buffer('zp', torch.tensor(0.0))
        self.learnable = False

    @staticmethod
    def _pctl(v, q):
        """Exact q-th percentile (0..100) of a 1-D tensor: deterministic (no sampling), any size."""
        n = v.numel()
        k = min(n, max(1, int(math.ceil(q / 100.0 * n))))
        return torch.kthvalue(v, k).values

    def forward(self, x):
        if self.bits is None or self.mode == 'off':
            return x
        if self.mode == 'observe':
            with torch.no_grad():
                self.lo = torch.minimum(self.lo, x.min())
                self.hi = torch.maximum(self.hi, x.max())
                flat = x.detach().flatten().float()
                a = flat.abs()
                st = torch.stack([a.max(), self._pctl(a, 99.9), self._pctl(a, 99.0),
                                  self._pctl(flat, 100.0 - self.pct), self._pctl(flat, self.pct),
                                  self._pctl(a, self.pct)])
                self.batch_stats.append(st)
            return x
        if self.scheme == 'symmetric':
            qmin, qmax = -2 ** (self.bits - 1), 2 ** (self.bits - 1) - 1
        else:
            qmin, qmax = 0, 2 ** self.bits - 1
        s = self.scale.abs() + 1e-12
        if self.learnable and self.training:
            s = grad_scale(s, 1.0 / math.sqrt(x.numel() * qmax))
        q = torch.clamp(round_ste(x / s) + self.zp, qmin, qmax)
        return (q - self.zp) * s

    @torch.no_grad()
    def calibrate(self, method='minmax'):
        """minmax: smallest/largest value over all calibration batches.
        percentile: the self.pct percentile computed exactly on each batch, averaged over batches
        (the running-average statistic Brevitas uses for Int8ActPerTensorFloat)."""
        if self.bits is None:
            return
        use_pct = method == 'percentile' and len(self.batch_stats) > 0
        if use_pct:
            m = torch.stack(self.batch_stats).mean(0)
            lo, hi, thr_abs = m[3].item(), m[4].item(), m[5].item()
        else:
            lo, hi = self.lo.item(), self.hi.item()
            thr_abs = max(abs(lo), abs(hi))
        self.batch_stats = []
        if self.scheme == 'symmetric':
            self.scale.data.fill_(max(thr_abs, 1e-8) / (2 ** (self.bits - 1) - 1))
            self.zp.fill_(0.0)
            return
        lo, hi = min(lo, 0.0), max(hi, 0.0)       # the range must contain zero (exact zero padding)
        qmax = 2 ** self.bits - 1
        s = max(hi - lo, 1e-8) / qmax
        self.scale.data.fill_(s)
        self.zp.fill_(float(round(-lo / s)))

    def stats(self):
        """Range summary of the observed activations (for act_stats.py)."""
        m = torch.stack(self.batch_stats) if self.batch_stats else torch.zeros(1, 6)
        return dict(min=self.lo.item(), max=self.hi.item(), absmax=m[:, 0].max().item(),
                    p999=m[:, 1].mean().item(), p99=m[:, 2].mean().item())


class ChannelActQuant(ActQuant):
    """Affine activation quantizer with one scale per channel group instead of one per tensor
    (min-max calibration). groups=None: every channel has its own scale; groups=[[0..7], [8..15]]:
    one scale per branch, e.g. the IQ and STFT halves of the fusion layer's concatenated input.
    In hardware a per-branch scale means splitting the 1x1 convolution into one partial sum per branch;
    a per-channel scale corresponds to folding an input-channel factor into the weights
    (cross-layer equalization / SmoothQuant), so treat per-channel results as an upper bound."""

    def __init__(self, bits=None, dim=1, groups=None):
        super().__init__(bits, 'affine')
        self.dim, self.groups = dim, groups
        self.cmin = self.cmax = None
        self.register_buffer('cscale', torch.ones(1))
        self.register_buffer('czp', torch.zeros(1))

    def _view(self, t, x):
        shape = [1] * x.dim()
        shape[self.dim] = -1
        return t.view(shape)

    def forward(self, x):
        if self.bits is None or self.mode == 'off':
            return x
        if self.mode == 'observe':
            with torch.no_grad():
                xm = x.detach().movedim(self.dim, -1).reshape(-1, x.shape[self.dim]).float()
                lo, hi = xm.min(0).values, xm.max(0).values
                self.cmin = lo if self.cmin is None else torch.minimum(self.cmin, lo)
                self.cmax = hi if self.cmax is None else torch.maximum(self.cmax, hi)
            return x
        qmax = 2 ** self.bits - 1
        s, zp = self._view(self.cscale, x), self._view(self.czp, x)
        q = torch.clamp(round_ste(x / s) + zp, 0, qmax)
        return (q - zp) * s

    @torch.no_grad()
    def calibrate(self, method='minmax'):
        if self.bits is None:
            return
        lo, hi = self.cmin.clone(), self.cmax.clone()
        for g in (self.groups or []):
            idx = torch.as_tensor(g, device=lo.device)
            lo[idx], hi[idx] = lo[idx].min(), hi[idx].max()
        lo, hi = lo.clamp(max=0.0), hi.clamp(min=0.0)       # each range contains zero
        s = (hi - lo).clamp(min=1e-8) / (2 ** self.bits - 1)
        self.cscale, self.czp = s, torch.round(-lo / s)
        self.cmin = self.cmax = None
        self.batch_stats = []


def use_channel_scales(model, names=None, groups=None):
    """Replace the per-tensor input quantizer of quantized conv/linear layers by ChannelActQuant.
    names: substrings selecting the layers (None = every layer whose activations are quantized).
    groups: channel groups for those layers (None = one scale per channel). Returns the layer names."""
    done = []
    for name, m in model.named_modules():
        if isinstance(m, (QConv, QLinear)) and m.aq.bits is not None:
            if names is None or any(n in name for n in names):
                m.aq = ChannelActQuant(m.aq.bits, dim=1 if isinstance(m, QConv) else -1, groups=groups)
                done.append(name)
    return done


class WeightQuant(nn.Module):
    """Symmetric per-output-channel weight quantizer (scale from the weights, or LSQ-learnable)."""

    def __init__(self, bits=None, ch_axis=0):
        super().__init__()
        self.bits, self.ch_axis = bits, ch_axis
        self.enabled = False
        self.learnable = False
        self.scale = None

    def init_scale(self, w):
        qmax = 2 ** (self.bits - 1) - 1
        dims = [d for d in range(w.dim()) if d != self.ch_axis]
        amax = w.detach().abs().amax(dim=dims, keepdim=True).clamp_min(1e-8)
        self.scale = nn.Parameter(amax / qmax, requires_grad=self.learnable)

    def forward(self, w):
        if self.bits is None or not self.enabled:
            return w
        if self.scale is None or self.scale.shape[self.ch_axis] != w.shape[self.ch_axis]:
            self.init_scale(w)
        qmax = 2 ** (self.bits - 1) - 1
        s = self.scale.abs() + 1e-12
        if self.learnable and self.training:
            s = grad_scale(s, 1.0 / math.sqrt(w[0].numel() * qmax))
        return torch.clamp(round_ste(w / s), -qmax, qmax) * s


# ----------------------------------------------------------------------------- quantized layers
class QConv(nn.Module):
    def __init__(self, conv, group):
        super().__init__()
        self.conv, self.group = conv, group
        self.aq, self.wq = ActQuant(), WeightQuant()

    def forward(self, x):
        c = self.conv
        w = self.wq(c.weight)
        x = self.aq(x)
        if isinstance(c, nn.Conv1d):
            return F.conv1d(x, w, c.bias, c.stride, c.padding, c.dilation, c.groups)
        return F.conv2d(x, w, c.bias, c.stride, c.padding, c.dilation, c.groups)


class QLinear(nn.Module):
    def __init__(self, lin, group):
        super().__init__()
        self.lin, self.group = lin, group
        self.aq, self.wq = ActQuant(), WeightQuant()

    def forward(self, x):
        return F.linear(self.aq(x), self.wq(self.lin.weight), self.lin.bias)


class QLSTM(nn.Module):
    """Unrolled multi-layer (bi)directional LSTM with the weights of an nn.LSTM (batch_first)."""

    def __init__(self, lstm, group='LSTM'):
        super().__init__()
        assert lstm.batch_first
        self.lstm, self.group = lstm, group
        L, D = lstm.num_layers, 2 if lstm.bidirectional else 1
        self.L, self.D, self.H = L, D, lstm.hidden_size
        self.in_q = nn.ModuleList([ActQuant() for _ in range(L)])
        self.h_q = nn.ModuleList([ActQuant() for _ in range(L * D)])
        self.wih_q = nn.ModuleList([WeightQuant() for _ in range(L * D)])
        self.whh_q = nn.ModuleList([WeightQuant() for _ in range(L * D)])

    def _p(self, name, layer, d):
        return getattr(self.lstm, f'{name}_l{layer}' + ('_reverse' if d == 1 else ''))

    def forward(self, x, hx=None):
        B, T, _ = x.shape
        inp = x
        for layer in range(self.L):
            xq = self.in_q[layer](inp)
            outs = []
            for d in range(self.D):
                k = layer * self.D + d
                Wih = self.wih_q[k](self._p('weight_ih', layer, d))
                Whh = self.whh_q[k](self._p('weight_hh', layer, d))
                b = self._p('bias_ih', layer, d) + self._p('bias_hh', layer, d)
                pre = F.linear(xq, Wih, b)                          # (B,T,4H) all time steps at once
                h = x.new_zeros(B, self.H)
                c = x.new_zeros(B, self.H)
                hs = [None] * T
                steps = range(T - 1, -1, -1) if d == 1 else range(T)
                for t in steps:
                    g = pre[:, t] + F.linear(self.h_q[k](h), Whh)
                    i, f, gg, o = g.chunk(4, 1)
                    c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(gg)
                    h = torch.sigmoid(o) * torch.tanh(c)
                    hs[t] = h
                outs.append(torch.stack(hs, 1))
            inp = torch.cat(outs, -1) if self.D == 2 else outs[0]
            if layer < self.L - 1 and self.lstm.dropout > 0 and self.training:
                inp = F.dropout(inp, self.lstm.dropout, True)
        return inp, None


# ----------------------------------------------------------------------------- BN folding
@torch.no_grad()
def _fold(conv, bn):
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    shape = [-1] + [1] * (conv.weight.dim() - 1)
    conv.weight.mul_(s.view(shape))
    b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    new_b = (b - bn.running_mean) * s + bn.bias
    conv.bias = nn.Parameter(new_b)


@torch.no_grad()
def fold_bn(model):
    """Fold every BatchNorm that directly follows a Conv (inside nn.Sequential, or dwconv->norm pairs)."""
    model.eval()
    n = 0
    for mod in model.modules():
        if isinstance(mod, nn.Sequential):
            kids = list(mod._modules.items())
            for (na, a), (nb, b) in zip(kids, kids[1:]):
                if isinstance(a, (nn.Conv1d, nn.Conv2d)) and isinstance(b, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    _fold(a, b); mod._modules[nb] = nn.Identity(); n += 1
        if hasattr(mod, 'dwconv') and isinstance(getattr(mod, 'norm', None), nn.BatchNorm1d):
            _fold(mod.dwconv, mod.norm); mod.norm = nn.Identity(); n += 1
    return n


# ----------------------------------------------------------------------------- groups
def group_of(name):
    """Block family of a module, from its qualified name in IQFormerNet (others -> 'CONV'/'HEAD')."""
    if 'patch_LSTM' in name or name.endswith('lstm') or '.gru' in name:
        return 'LSTM'
    if '.attn.' in name or name.endswith('.attn'):
        return 'ATTN'
    if name.endswith('head') or 'classifier' in name:
        return 'HEAD'
    if 'patch_embed' in name:
        return 'STEM'
    if 'fusion' in name:
        return 'FUSION'
    if 'local_representation' in name:
        return 'LOCAL'
    if '.linear.' in name:
        return 'FFN'
    if 'network' in name:
        return 'CONVENC'
    return 'CONV'


FAMILY = {'STEM': 'CONV', 'FUSION': 'CONV', 'CONVENC': 'CONV', 'LOCAL': 'CONV', 'FFN': 'CONV',
          'CONV': 'CONV', 'ATTN': 'ATTN', 'LSTM': 'LSTM', 'HEAD': 'HEAD'}


def parse_spec(spec):
    """'all=W8A8'  or  'CONV=W4A8,LSTM=W8A8,ATTN=W8A8,HEAD=W8A8'  or 'LSTM=W4A16' ('A16'/'W32' = float).
    Keys may be families (CONV, ATTN, LSTM, HEAD), fine groups (STEM, FUSION, CONVENC, LOCAL, FFN) or 'all'.
    Returns dict key -> (wbits or None, abits or None)."""
    out = {}
    if spec in (None, '', 'fp32', 'FP32'):
        return out
    for part in spec.split(','):
        k, v = part.split('=')
        m = re.fullmatch(r'W(\d+)A(\d+)', v.strip().upper())
        wb, ab = int(m.group(1)), int(m.group(2))
        out[k.strip().upper()] = (None if wb >= 16 else wb, None if ab >= 16 else ab)
    return out


def bits_for(group, spec):
    for key in (group, FAMILY.get(group, group), 'ALL'):
        if key in spec:
            return spec[key]
    return (None, None)


# ----------------------------------------------------------------------------- model conversion
def _replace(model, prefix=''):
    for name, child in list(model.named_children()):
        full = f'{prefix}.{name}' if prefix else name
        if isinstance(child, (nn.Conv1d, nn.Conv2d)):
            setattr(model, name, QConv(child, group_of(full)))
        elif isinstance(child, nn.Linear):
            setattr(model, name, QLinear(child, group_of(full)))
        elif isinstance(child, nn.LSTM):
            setattr(model, name, QLSTM(child))
        elif isinstance(child, nn.MultiheadAttention):
            continue                           # fused op reads .weight directly; left in floating point
        else:
            _replace(child, full)


def quantize_model(model, spec, fold=True, quantize_bare=True, act_scheme='affine'):
    """Return a quantized copy of `model` (activation quantizers still uncalibrated).
    act_scheme 'symmetric' + fold=False + percentile 99.999 calibration approximates the project's
    earlier Brevitas setup (Int8WeightPerChannelFloat / Int8ActPerTensorFloat, BN not folded)."""
    q = copy.deepcopy(model).eval()
    if fold:
        fold_bn(q)
    _replace(q)
    q.eval()                                   # the new wrapper modules are created in training mode
    for m in q.modules():
        if isinstance(m, ActQuant):
            m.scheme = act_scheme
    spec = parse_spec(spec) if isinstance(spec, str) else spec
    for mod in q.modules():
        if isinstance(mod, (QConv, QLinear)):
            wb, ab = bits_for(mod.group, spec)
            mod.wq.bits, mod.aq.bits = wb, ab
            mod.wq.ch_axis = 0
            mod.wq.enabled = wb is not None
        elif isinstance(mod, QLSTM):
            wb, ab = bits_for('LSTM', spec)
            for w in list(mod.wih_q) + list(mod.whh_q):
                w.bits, w.enabled = wb, wb is not None
            for a in list(mod.in_q) + list(mod.h_q):
                a.bits = ab
    if quantize_bare:
        with torch.no_grad():
            for name, p in q.named_parameters():
                if name.endswith('layer_scale') or name.endswith('w_g'):
                    wb, _ = bits_for(group_of(name), spec)
                    if wb is not None:
                        qmax = 2 ** (wb - 1) - 1
                        s = p.abs().max().clamp_min(1e-8) / qmax
                        p.copy_(torch.clamp((p / s).round(), -qmax, qmax) * s)
    return q


def act_quantizers(model):
    return [m for m in model.modules() if isinstance(m, ActQuant) and m.bits is not None]


@torch.no_grad()
def calibrate(model, x_cal, method='minmax', pct=99.99, batch=256, auto_bits=4):
    """method: 'minmax', 'percentile' (clip at the pct-th percentile), or 'auto' = percentile for activation
    quantizers of auto_bits bits or fewer and min-max for wider ones (Steps 5c/6: clipping rescues 4-bit
    activations but costs ~0.35 pp at 8 bits, where every value already has enough levels)."""
    model.eval()
    aqs = act_quantizers(model)
    for a in aqs:
        a.mode = 'observe'
        a.lo.fill_(float('inf')); a.hi.fill_(float('-inf')); a.batch_stats = []; a.pct = pct
    for i in range(0, len(x_cal), batch):
        model(x_cal[i:i + batch])
    for a in aqs:
        m = method if method != 'auto' else ('percentile' if a.bits <= auto_bits else 'minmax')
        a.calibrate(m)
        a.mode = 'quant'
    return len(aqs)


def set_learnable(model, on=True):
    """Make all step sizes LSQ-learnable (for QAT)."""
    for m in model.modules():
        if isinstance(m, ActQuant) and m.bits is not None and m.mode == 'quant':
            m.learnable = on
            m.scale.requires_grad_(on)
        if isinstance(m, WeightQuant) and m.enabled:
            m.learnable = on
            if m.scale is not None:
                m.scale.requires_grad_(on)


def quant_summary(model):
    rows = []
    for name, m in model.named_modules():
        if isinstance(m, (QConv, QLinear)):
            rows.append((name, m.group, m.wq.bits, m.aq.bits))
        elif isinstance(m, QLSTM):
            rows.append((name, 'LSTM', m.wih_q[0].bits, m.in_q[0].bits))
    return rows
