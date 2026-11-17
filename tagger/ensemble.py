"""Independent linear/MLP readouts, vectorized over a condition axis."""
import torch
from torch import nn
from torch.nn import functional as F

from .model import build_tagger


class IndependentReadouts(nn.Module):
    def __init__(self, cfg, dimension, labels, heads):
        super().__init__()
        base = build_tagger(cfg, dimension, labels)
        self.linear = cfg.name == "linear"
        self.dropout = 0. if self.linear else cfg.dropout
        first = base.net if self.linear else base.net[0]
        self.weight = nn.Parameter(first.weight.T.detach()[None].repeat(heads, 1, 1))
        self.bias = nn.Parameter(first.bias.detach()[None].repeat(heads, 1))
        if not self.linear:
            self.output_weight = nn.Parameter(base.net[3].weight.T.detach()[None].repeat(heads, 1, 1))
            self.output_bias = nn.Parameter(base.net[3].bias.detach()[None].repeat(heads, 1))

    def forward(self, x, heads=None):
        index = slice(None) if heads is None else heads
        h = torch.einsum("ncd,cdh->nch", x, self.weight[index]) + self.bias[index]
        if self.linear:
            return h
        h = F.relu(h)
        # Common random numbers across independent conditions; each head has standard dropout.
        mask = F.dropout(torch.ones(h.shape[0], 1, h.shape[-1], device=h.device, dtype=h.dtype),
                         p=self.dropout, training=self.training)
        return torch.einsum("nch,chl->ncl", h * mask, self.output_weight[index]) + self.output_bias[index]


def clip_heads(module, maximum):
    parameters = list(module.parameters())
    norms = sum(p.grad.float().flatten(1).square().sum(-1) for p in parameters).sqrt()
    factor = (maximum / (norms + 1e-6)).clamp_max(1)
    for p in parameters:
        p.grad.mul_(factor.reshape(-1, *([1] * (p.ndim - 1))))
