"""Three progressively larger, residual extensions to the released IQFormer fusion.

The true SNR is never an input. The 5-bin head is supervised only during training.
"""
import torch
from torch import nn


class MultiScaleAdapter(nn.Module):
    def __init__(self, mode='ms'):
        super().__init__()
        if mode not in ('ms', 'gate', 'snr'):
            raise ValueError(mode)
        self.mode = mode
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(8, 8, k, padding=k // 2, groups=8, bias=False),
                          nn.BatchNorm1d(8), nn.ReLU())
            for _ in range(2) for k in (3, 7, 15)
        ])
        self.proj = nn.Conv1d(48, 32, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        if mode == 'snr':
            self.snr_head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 5))
            self.snr_embed = nn.Linear(5, 8)
        if mode != 'ms':
            self.gate = nn.Sequential(nn.Linear(40 if mode == 'snr' else 32, 16),
                                      nn.ReLU(), nn.Linear(16, 32), nn.Sigmoid())

    def forward(self, iq_stem, tf_stem, fused):
        if iq_stem.shape[1:] != (8, 128) or tf_stem.shape[1:] != (8, 128) or fused.shape[1:] != (32, 128):
            raise ValueError('Expected stem [B,8,128] twice and fusion [B,32,128]')
        features = [branch(stem) for stem, branches in
                    ((iq_stem, self.branches[:3]), (tf_stem, self.branches[3:]))
                    for branch in branches]
        residual = self.proj(torch.cat(features, dim=1))
        pooled = fused.mean(dim=-1)
        snr_logits = None
        if self.mode == 'snr':
            snr_logits = self.snr_head(pooled)
            soft_bins = torch.softmax(snr_logits, dim=1)
            gate_input = torch.cat((pooled, self.snr_embed(soft_bins)), dim=1)
        else:
            gate_input = pooled
        if self.mode != 'ms':
            residual = residual * self.gate(gate_input).unsqueeze(-1)
        return fused + residual, snr_logits
