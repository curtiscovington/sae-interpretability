from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    def __init__(self, d_model: int, d_sae: int, sparsity_mode: str = "relu_l1", topk: int = 64):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_model, bias=False)
        self.sparsity_mode = sparsity_mode
        self.topk = topk
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.encoder(x)
        h = F.relu(pre)
        if self.sparsity_mode == "topk":
            k = min(self.topk, h.shape[-1])
            vals, idx = torch.topk(h, k=k, dim=-1)
            mask = torch.zeros_like(h)
            mask.scatter_(dim=-1, index=idx, src=torch.ones_like(vals))
            h = h * mask
        return h

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        recon = self.decoder(h)
        return recon, h


@dataclass
class LossOutput:
    total: torch.Tensor
    recon: torch.Tensor
    l1: torch.Tensor


def sae_loss(x: torch.Tensor, recon: torch.Tensor, h: torch.Tensor, l1_coeff: float) -> LossOutput:
    recon_loss = F.mse_loss(recon.float(), x.float())
    l1 = h.abs().mean()
    total = recon_loss + l1_coeff * l1
    return LossOutput(total=total, recon=recon_loss, l1=l1)
