"""Small interchangeable transformations before the classifier."""
import torch
from torch import nn
from torch.nn import functional as F


class Layer(nn.Module):
    def __init__(self, cfg, dimension):
        super().__init__()
        self.layer = cfg.layer

    def forward(self, features, heads=None):
        return features[:, :, self.layer], features.new_empty(0)


class LayerMix(nn.Module):
    def __init__(self, cfg, dimension):
        super().__init__()
        self.layers = list(cfg.layers)
        self.kind, self.temperature = cfg.name, cfg.temperature
        if self.kind == "global_mix":
            self.logits = nn.Parameter(torch.zeros(len(self.layers)))
        else:
            self.selector = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, cfg.hidden),
                                          nn.ReLU(), nn.Linear(cfg.hidden, len(self.layers)))

    def forward(self, features, heads=None):
        selected = features[:, :, self.layers]
        if self.kind == "global_mix":
            weights = self.logits.softmax(-1).expand(*features.shape[:2], -1)
        else:
            weights = (self.selector(features[:, :, -1]) / self.temperature).softmax(-1)
        gates = weights
        if self.kind == "hard_mix":
            hard = F.one_hot(weights.argmax(-1), len(self.layers)).to(weights.dtype)
            gates = hard + weights - weights.detach()
        return (selected * gates[..., None]).sum(-2), weights


class HeadMix(nn.Module):
    def __init__(self, cfg, encoder):
        super().__init__()
        self.heads = encoder.num_heads
        self.width = encoder.dimension // self.heads
        self.register_buffer("projections", encoder.head_projections())
        self.logits = nn.Parameter(torch.zeros(len(self.projections), self.heads))

    def forward(self, features, heads):
        weights = self.logits.flatten().softmax(-1).view_as(self.logits)
        context = heads * weights.repeat_interleave(self.width, -1)
        mixed = torch.einsum("bwld,lod->bwo", context, self.projections)
        return F.layer_norm(mixed, mixed.shape[-1:]), weights.flatten().expand(*features.shape[:2], -1)


class LayerHeadMix(nn.Module):
    def __init__(self, cfg, encoder):
        super().__init__()
        from omegaconf import OmegaConf
        layer_cfg = OmegaConf.merge(cfg, {"name": "global_mix"})
        self.layer_mix = LayerMix(layer_cfg, encoder.dimension)
        self.head_mix = HeadMix(cfg, encoder)
        self.logits = nn.Parameter(torch.zeros(2))

    def forward(self, features, heads):
        layer, _ = self.layer_mix(features)
        head, _ = self.head_mix(features, heads)
        gates = self.logits.softmax(-1)
        mixed = gates[0] * F.layer_norm(layer, layer.shape[-1:]) + gates[1] * head
        return mixed, gates.expand(*features.shape[:2], -1)


class ContextSelector(nn.Module):
    """Score [focal; candidate; sentence mean] from ordinary layer-12 vectors."""
    def __init__(self, cfg, dimension):
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(dimension)
        self.net = nn.Sequential(nn.Linear(3 * dimension, cfg.hidden), nn.ReLU(), nn.Linear(cfg.hidden, 1))
        self.register_buffer("rhos", torch.tensor(list(cfg.rhos)))

    def forward(self, words, valid, focal, teacher=None):
        rows = torch.arange(len(words), device=words.device)
        context = valid.clone()
        context[rows, focal] = False
        if self.cfg.name == "oracle_mask":
            return teacher
        if self.cfg.name == "random_mask":
            keep = (self.rhos[torch.randint(len(self.rhos), (len(words),), device=words.device)]
                    * context.sum(-1)).round()
            noise = torch.rand(valid.shape, device=words.device).masked_fill(~context, 2)
            rank = noise.argsort(-1).argsort(-1)
            gates = (rank < keep[:, None]).to(words.dtype)
        else:
            h = self.norm(words)
            f = h[rows, focal, None].expand_as(h)
            mean = ((h * valid[..., None]).sum(1) / valid.sum(-1, keepdim=True))[:, None].expand_as(h)
            scores = self.net(torch.cat([f, h, mean], -1)).squeeze(-1).masked_fill(~valid, -torch.inf)
            if self.cfg.gate == "anchored":
                gates = ((scores - scores.amax(-1, keepdim=True)) / self.cfg.temperature).exp()
            else:
                gates = (scores / self.cfg.temperature).sigmoid()
            gates = gates.masked_fill(~valid, 0)
        return gates.scatter(1, focal[:, None], 1)


def build_method(cfg, encoder):
    if cfg.kind == "context_mask":
        return ContextSelector(cfg, encoder.dimension)
    if cfg.name in {"last", "layer"}:
        return Layer(cfg, encoder.dimension)
    if cfg.name == "head_mix":
        return HeadMix(cfg, encoder)
    if cfg.name == "layer_head_mix":
        return LayerHeadMix(cfg, encoder)
    return LayerMix(cfg, encoder.dimension)
