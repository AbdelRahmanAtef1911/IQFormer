"""Model zoo.

IQFormerNet wraps the authors' released IQFormer (imported from the repository, unchanged) and
 - computes the STFT on the GPU (same numbers as scipy, much faster than the per-sample CPU STFT),
 - fixes the batch-size-1 crash (torch.squeeze -> squeeze(dim=2)),
 - optionally changes the STFT input, the time-domain input, and the activation function
   (used for the proposed model and its ablation).
With default options it is numerically the released model, and it loads weight.pt files saved
by the released main.py (load_legacy).

Baselines for the rubric ('MLP, CNN, and transformer'): MLP, CNN1D (VGG-style), and the repository's
FEA-T (transformer), MCformer, MCLDNN, PET-CGDNN, AMC-Net.
"""
import sys
import torch
import torch.nn as nn
from .data import stft_features, iq_input, STFT_CHANNELS

ACTS = {'gelu': nn.GELU, 'relu': nn.ReLU, 'hswish': nn.Hardswish, 'silu': nn.SiLU, 'relu6': nn.ReLU6}


def _swap_act(module, act):
    if act == 'gelu':
        return
    for name, child in module.named_children():
        if isinstance(child, nn.GELU):
            setattr(module, name, ACTS[act]())
        else:
            _swap_act(child, act)


class IQFormerNet(nn.Module):
    def __init__(self, num_classes=11, stft='real', iq='iq', act='gelu'):
        super().__init__()
        from model.IQFormer import IQFormer           # the authors' file, from the repository root
        self.stft_mode, self.iq_mode, self.act = stft, iq, act
        self.m = IQFormer([2, 3, 2], embed_dims=[64, 64, 64], mlp_ratios=4, act_layer=nn.GELU,
                          num_classes=num_classes, down_patch_size=3, down_stride=2, down_pad=1,
                          drop_rate=0.2, drop_path_rate=0., use_layer_scale=False,
                          layer_scale_init_value=1e-5, fork_feat=False, vit_num=1)
        c = STFT_CHANNELS[stft]
        if c != 1:
            self.m.BN_stft = nn.BatchNorm2d(c)
            self.m.patch_embedSTFT = nn.Sequential(nn.Conv2d(c, 8, kernel_size=(32, 1)), nn.BatchNorm2d(8), nn.ReLU())
        if iq == 'iqap':                               # 4 time-domain channels -> grouped stem, 2 filters each
            self.m.BN = nn.BatchNorm1d(4)
            self.m.patch_embedIQ = nn.Sequential(nn.Conv1d(4, 8, kernel_size=5, padding=2, groups=4), nn.BatchNorm1d(8))
        _swap_act(self.m, act)

    def forward(self, x):
        m = self.m
        s = stft_features(x, self.stft_mode)
        x = iq_input(x, self.iq_mode)
        x = m.BN(x)
        s = m.BN_stft(s)
        x = m.patch_embedIQ(x)
        s = m.patch_embedSTFT(s).squeeze(2)            # fixed: keeps the batch dimension for batch size 1
        x = m.fusion(x, s)
        x, _ = m.patch_LSTM(x.permute(0, 2, 1))
        x = m.forward_tokens(x.permute(0, 2, 1))
        x = m.norm(x)
        return m.head(m.globalavgpool(x))

    def load_legacy(self, path, device='cpu'):
        """Load a weight.pt written by the released main.py (keys without the 'm.' prefix)."""
        sd = torch.load(path, map_location=device)
        if any(k.startswith('m.') for k in sd):
            self.load_state_dict(sd)
        else:
            self.m.load_state_dict(sd)
        return self


class MLP(nn.Module):
    """Fully connected baseline on the flattened 2x128 frame."""
    def __init__(self, num_classes=11, hidden=(512, 256, 128), drop=0.3):
        super().__init__()
        layers, d = [nn.Flatten()], 256
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(drop)]
            d = h
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    """VGG-style 1-D CNN (the VGG10 design of O'Shea et al. 2018 / Jentzsch et al. 2022,
    5 conv-pool blocks for 128-sample frames)."""
    def __init__(self, num_classes=11, ch=64, blocks=5, drop=0.3):
        super().__init__()
        layers, c = [], 2
        for _ in range(blocks):
            layers += [nn.Conv1d(c, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.ReLU(), nn.MaxPool1d(2)]
            c = ch
        self.features = nn.Sequential(*layers)
        n = 128 // 2 ** blocks
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(ch * n, 128), nn.ReLU(), nn.Dropout(drop),
                                        nn.Linear(128, 128), nn.ReLU(), nn.Dropout(drop), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name, num_classes=11, stft='real', iq='iq', act='gelu'):
    name = name.lower()
    if name == 'iqformer':
        return IQFormerNet(num_classes, stft=stft, iq=iq, act=act)
    if name == 'mlp':
        return MLP(num_classes)
    if name == 'cnn':
        return CNN1D(num_classes)
    if name == 'feat':                                  # transformer baseline (repository)
        from model.FEA_T128 import FEA_T
        return FEA_T(num_class=num_classes)
    if name == 'mcformer':
        from model.MCFormer import MCformer
        return MCformer(num_classes=num_classes)
    if name == 'mcldnn':
        from model.MCLDNN import MCLDNN
        return MCLDNN(frame_length=128, num_classes=num_classes)
    if name == 'petcgdnn':
        from model.PETCGDNN import PETCGDNN
        return PETCGDNN(num_classes=num_classes, frame_length=128)
    if name == 'amcnet':
        from model.AMCNET import AMC_Net
        return AMC_Net(num_classes=num_classes, sig_len=128)
    raise ValueError(name)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def add_repo_to_path(repo):
    if repo not in sys.path:
        sys.path.insert(0, repo)
