"""Register new architecture arms without editing amcq.models or model.IQFormer."""
import torch
from torch import nn
from amcq.data import stft_features, iq_input
from amcq.models import IQFormerNet, build_model as original_build_model, count_params
from .ms_adapter import MultiScaleAdapter


class ExtendedIQFormerNet(IQFormerNet):
    def __init__(self, mode, num_classes=11, stft='real', iq='iq', act='gelu'):
        super().__init__(num_classes=num_classes, stft=stft, iq=iq, act=act)
        self.ms_adapter = MultiScaleAdapter(mode)

    def forward(self, x, return_snr=False):
        m = self.m
        s = stft_features(x, self.stft_mode)
        x = iq_input(x, self.iq_mode)
        iq = m.patch_embedIQ(m.BN(x))
        tf = m.patch_embedSTFT(m.BN_stft(s)).squeeze(2)
        z, snr_logits = self.ms_adapter(iq, tf, m.fusion(iq, tf))
        z, _ = m.patch_LSTM(z.permute(0, 2, 1))
        z = m.forward_tokens(z.permute(0, 2, 1))
        logits = m.head(m.globalavgpool(m.norm(z)))
        return (logits, snr_logits) if return_snr else logits


def build_model(name, num_classes=11, stft='real', iq='iq', act='gelu', input_norm=False):
    name = name.lower()
    if name not in ('iqf_ms', 'iqf_ms_gate', 'iqf_ms_snr'):
        return original_build_model(name, num_classes, stft, iq, act, input_norm)
    if input_norm:
        raise ValueError('Keep input_norm=False for matched IQFormer experiments')
    mode = {'iqf_ms': 'ms', 'iqf_ms_gate': 'gate', 'iqf_ms_snr': 'snr'}[name]
    return ExtendedIQFormerNet(mode, num_classes, stft, iq, act)
