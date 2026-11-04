"""IQFormer with DSP-feature conditioning of the fused tokens.

The features modulate the 32 x 128 fused token tensor z (after the fusion layer, before the BiLSTM) with FiLM:
    z <- z * (1 + gamma(f)) + beta(f)
The last FiLM layer is zero-initialised, so at the start every variant computes exactly the original IQFormer.

Model names
  iqf_cum     cumulant vector (7)            -> MLP -> FiLM
  iqf_pls     power-law spectra (3 x 128)    -> small 1-D CNN (max + mean pooled) -> FiLM
  iqf_dsp     both feature sets              -> FiLM
  iqf_ms_dsp  the multi-scale adapter of amcq_arch (iqf_ms) + both feature sets (only if both help separately)
The true SNR is never used. The features are computed from the raw I/Q window only.
"""
import torch
from torch import nn
import amcq_arch.models as _arch
from amcq.data import stft_features, iq_input
from amcq.models import IQFormerNet
from amcq_arch.ms_adapter import MultiScaleAdapter
from .features import cumulant_features, power_spectra

_ORIGINAL_BUILD = getattr(_arch, '_original_arch_build', _arch.build_model)
DSP_MODELS = {'iqf_cum': ('cum',), 'iqf_pls': ('pls',), 'iqf_dsp': ('cum', 'pls'), 'iqf_ms_dsp': ('cum', 'pls')}


class DSPConditioner(nn.Module):
    def __init__(self, use=('cum', 'pls'), channels=32):
        super().__init__()
        self.use = tuple(use)
        dim = 0
        if 'cum' in self.use:
            self.cum_mlp = nn.Sequential(nn.Linear(7, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
            dim += 32
        if 'pls' in self.use:
            self.pls_conv = nn.Sequential(nn.Conv1d(3, 16, 7, padding=3), nn.ReLU(),
                                          nn.Conv1d(16, 16, 5, padding=2), nn.ReLU())
            self.pls_mlp = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
            dim += 32
        self.film = nn.Linear(dim, 2 * channels)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def features(self, x):
        parts = []
        if 'cum' in self.use:
            parts.append(self.cum_mlp(cumulant_features(x)))
        if 'pls' in self.use:
            h = self.pls_conv(power_spectra(x))
            parts.append(self.pls_mlp(torch.cat([h.amax(dim=-1), h.mean(dim=-1)], dim=1)))
        return torch.cat(parts, dim=1)

    def forward(self, x_raw, z):
        gamma, beta = self.film(self.features(x_raw)).chunk(2, dim=1)
        return z * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)


class DSPIQFormerNet(IQFormerNet):
    def __init__(self, use, with_ms=False, num_classes=11, stft='real', iq='iq', act='gelu'):
        super().__init__(num_classes=num_classes, stft=stft, iq=iq, act=act)
        self.dsp = DSPConditioner(use)
        self.ms_adapter = MultiScaleAdapter('ms') if with_ms else None

    def forward(self, x):
        m = self.m
        raw = x
        s = stft_features(x, self.stft_mode)
        x = iq_input(x, self.iq_mode)
        iq = m.patch_embedIQ(m.BN(x))
        tf = m.patch_embedSTFT(m.BN_stft(s)).squeeze(2)
        z = m.fusion(iq, tf)
        if self.ms_adapter is not None:
            z, _ = self.ms_adapter(iq, tf, z)
        z = self.dsp(raw, z)
        z, _ = m.patch_LSTM(z.permute(0, 2, 1))
        z = m.forward_tokens(z.permute(0, 2, 1))
        return m.head(m.globalavgpool(m.norm(z)))


def dsp_build_model(name, num_classes=11, stft='real', iq='iq', act='gelu', input_norm=False):
    key = name.lower()
    if key not in DSP_MODELS:
        return _ORIGINAL_BUILD(name, num_classes, stft, iq, act, input_norm)
    if input_norm:
        raise ValueError('Keep input_norm=False for matched IQFormer experiments')
    return DSPIQFormerNet(DSP_MODELS[key], with_ms=(key == 'iqf_ms_dsp'), num_classes=num_classes,
                          stft=stft, iq=iq, act=act)


build_model = dsp_build_model
