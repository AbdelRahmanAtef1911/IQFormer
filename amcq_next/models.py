"""IQFormer variants for the next round of experiments.

Model names are 'iqfx' followed by options joined with '_', for example 'iqfx_snr', 'iqfx_snr_c2d', 'iqfx_slim':

  snr     blind SNR branch (MSTFFNet-style). A small CNN on the batch-normalised I/Q window predicts 8 SNR bins
          (5 dB wide, -20..+20 dB); the softmax probabilities weight 8 learned 16-number embeddings (a bias-free
          Linear 8->16) and that 16-number summary is CONCATENATED with the pooled features before the classifier.
          Nothing multiplies the features. The true SNR is never an input; it can supervise the bin head during
          training only (train.py --snr-loss).
  oracle  DIAGNOSTIC ONLY: the true SNR bin (one-hot) replaces the estimate. Measures the ceiling of SNR
          conditioning. It needs the SNR at inference, so it is refused on the test set and never reported as a result.
  c2d     complex STFT of z = I + jQ (Blackman 31, hop 1, 32-point two-sided FFT, zero frequency centred), four
          channels (real, imaginary, log magnitude, phase/pi), encoded by two 2-D convolutions and pooled two ways
          (mean over frequency, and 4 frequency bands), replacing IQFormer's I-only real STFT stem.
  slim    half width: embed_dims 32 instead of 64 (same depth).
  ctr     adds two small projection heads used only by the training-time InfoNCE loss (train.py --ctr-weight);
          inference is unchanged.

With no option ('iqfx') the model is IQFormer as released (same parameters as amcq's IQFormerNet).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from amcq.data import stft_features, stft_bins, iq_input, STFT_CHANNELS
from amcq.models import _swap_act, count_params  # noqa: F401

SNR_BINS = 8
OPTIONS = ('snr', 'oracle', 'c2d', 'slim', 'ctr')


def snr_bin(snr):
    """True SNR (dB, tensor) -> bin index 0..7 for [-20,-15), [-15,-10), ..., [15,20)."""
    return ((snr + 20) // 5).clamp(0, SNR_BINS - 1).long()


def parse_name(name):
    parts = name.lower().split('_')
    if parts[0] != 'iqfx':
        return None
    opts = set(parts[1:])
    bad = opts - set(OPTIONS)
    if bad:
        raise ValueError(f'unknown iqfx option(s) {sorted(bad)}; allowed {OPTIONS}')
    if {'snr', 'oracle'} <= opts:
        raise ValueError('choose snr OR oracle')
    return opts


def complex_stft4(x):
    """(B,2,128) -> (B,4,32,128): STFT of I+jQ, two-sided, fftshifted; real, imag, log1p|Z|, angle/pi."""
    z = torch.complex(x[:, 0].float(), x[:, 1].float())
    s = torch.fft.fftshift(stft_bins(z, nbins=32, nfft=32), dim=1)
    return torch.stack([s.real, s.imag, torch.log1p(s.abs()), torch.angle(s) / torch.pi], 1)


class ComplexSTFTStem(nn.Module):
    """4 x 32 x 128 complex STFT -> (out_ch, 128) tokens, keeping IQFormer's time resolution."""
    def __init__(self, out_ch, mid=16, bands=4):
        super().__init__()
        self.bn = nn.BatchNorm2d(4)
        self.conv = nn.Sequential(
            nn.Conv2d(4, mid, (5, 3), padding=(2, 1)), nn.BatchNorm2d(mid), nn.ReLU(),
            nn.Conv2d(mid, mid, (5, 3), stride=(2, 1), padding=(2, 1)), nn.BatchNorm2d(mid), nn.ReLU())
        self.bands = bands
        self.proj = nn.Sequential(nn.Conv1d(mid * (1 + bands), out_ch, 1), nn.BatchNorm1d(out_ch), nn.ReLU())

    def forward(self, s):
        h = self.conv(self.bn(s))                                         # (B, mid, 16, 128)
        mean_path = h.mean(dim=2)                                         # envelope over frequency
        band_path = F.adaptive_avg_pool2d(h, (self.bands, h.shape[-1])).flatten(1, 2)   # 4 frequency bands
        return self.proj(torch.cat([mean_path, band_path], 1))


class SNRBranch(nn.Module):
    """Blind SNR-bin estimator on the normalised I/Q window -> 16-number condition (soft embedding)."""
    def __init__(self, emb=16):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(2, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.BatchNorm1d(32), nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, SNR_BINS))
        self.embed = nn.Linear(SNR_BINS, emb, bias=False)                 # = probability-weighted embeddings

    def forward(self, xb):
        h = self.feat(xb)
        logits = self.mlp(torch.cat([h.mean(-1), h.amax(-1)], 1))
        return logits, self.embed(torch.softmax(logits, 1))


class NextIQFormer(nn.Module):
    def __init__(self, opts, num_classes=11, stft='real', iq='iq', act='gelu'):
        super().__init__()
        from model.IQFormer import IQFormer                               # the authors' file, unchanged
        self.opts = set(opts)
        self.needs_snr = 'oracle' in self.opts
        self.stft_mode, self.iq_mode, self.act = stft, iq, act
        D = 32 if 'slim' in self.opts else 64
        self.m = IQFormer([2, 3, 2], embed_dims=[D, D, D], mlp_ratios=4, act_layer=nn.GELU,
                          num_classes=num_classes, down_patch_size=3, down_stride=2, down_pad=1,
                          drop_rate=0.2, drop_path_rate=0., use_layer_scale=False,
                          layer_scale_init_value=1e-5, fork_feat=False, vit_num=1)
        tf_ch = D // 8                                                    # channels of each released stem
        if 'c2d' in self.opts:
            self.m.BN_stft = nn.Identity()
            self.m.patch_embedSTFT = ComplexSTFTStem(tf_ch)
        else:
            c = STFT_CHANNELS[stft]
            if c != 1:
                self.m.BN_stft = nn.BatchNorm2d(c)
                self.m.patch_embedSTFT = nn.Sequential(nn.Conv2d(c, tf_ch, kernel_size=(32, 1)),
                                                       nn.BatchNorm2d(tf_ch), nn.ReLU())
        if iq == 'iqap':
            self.m.BN = nn.BatchNorm1d(4)
            self.m.patch_embedIQ = nn.Sequential(nn.Conv1d(4, tf_ch, kernel_size=5, padding=2, groups=4),
                                                 nn.BatchNorm1d(tf_ch))
        self.cond = 0
        if 'snr' in self.opts:
            self.snr_branch = SNRBranch()
            self.cond = 16
        elif 'oracle' in self.opts:
            self.oracle_embed = nn.Linear(SNR_BINS, 16, bias=False)
            self.cond = 16
        if self.cond:
            self.cls_head = nn.Linear(D + self.cond, num_classes)           # replaces m.head
            self.m.head = nn.Identity()
        if 'ctr' in self.opts:
            self.proj_iq = nn.Sequential(nn.Linear(2 * tf_ch, 32), nn.ReLU(), nn.Linear(32, 32))
            self.proj_tf = nn.Sequential(nn.Linear(2 * tf_ch, 32), nn.ReLU(), nn.Linear(32, 32))
        _swap_act(self.m, act)

    def _stft(self, x):
        return complex_stft4(x) if 'c2d' in self.opts else stft_features(x, self.stft_mode)

    def forward(self, x, snr=None, return_aux=False):
        m = self.m
        s = self._stft(x)
        xb = m.BN(iq_input(x, self.iq_mode))
        iq = m.patch_embedIQ(xb)
        tf = m.patch_embedSTFT(m.BN_stft(s))
        if tf.dim() == 4:
            tf = tf.squeeze(2)                                            # batch-size-1 safe
        z = m.fusion(iq, tf)
        z, _ = m.patch_LSTM(z.permute(0, 2, 1))
        z = m.forward_tokens(z.permute(0, 2, 1))
        feat = m.globalavgpool(m.norm(z))
        aux = {}
        if 'snr' in self.opts:
            snr_logits, h = self.snr_branch(xb[:, :2])
            aux['snr_logits'] = snr_logits
            logits = self.cls_head(torch.cat([feat, h], 1))
        elif 'oracle' in self.opts:
            if snr is None:
                raise RuntimeError('iqfx_oracle is a diagnostic and needs the true SNR; it is never run on test data')
            h = self.oracle_embed(F.one_hot(snr_bin(snr), SNR_BINS).float())
            logits = self.cls_head(torch.cat([feat, h], 1))
        else:
            logits = m.head(feat)
        if return_aux:
            if 'ctr' in self.opts:
                pool = lambda t: torch.cat([t.mean(-1), t.amax(-1)], 1)      # noqa: E731
                aux['z_iq'] = F.normalize(self.proj_iq(pool(iq)), dim=1)
                aux['z_tf'] = F.normalize(self.proj_tf(pool(tf)), dim=1)
            return logits, aux
        return logits


def build_model(name, num_classes=11, stft='real', iq='iq', act='gelu', input_norm=False):
    opts = parse_name(name)
    if opts is None:
        from amcq_arch import models as am
        prev = getattr(am, '_before_next_build', None)
        if prev is None:
            from amcq.models import build_model as prev
        return prev(name, num_classes, stft, iq, act, input_norm)
    if input_norm:
        raise ValueError('input_norm is not used for iqfx models')
    return NextIQFormer(opts, num_classes, stft, iq, act)


build_model.__name__ = 'next_build_model'
