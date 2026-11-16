"""Frozen BERT with differentiable log-gate key masks and optional head contexts."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import BertModel


class FrozenBERT(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.model = BertModel.from_pretrained(name).eval().requires_grad_(False)
        self.dimension = self.model.config.hidden_size
        self.num_heads = self.model.config.num_attention_heads

    def head_projections(self):
        return torch.stack([layer.attention.output.dense.weight.detach()
                            for layer in self.model.encoder.layer])

    def train(self, mode=True):
        # The encoder stays frozen/eval even when its downstream modules train.
        super().train(False)
        return self

    def forward(self, ids, attention_mask, heads=False, all_layers=True):
        h = self.model.embeddings(ids)
        layers, contexts = [h] if all_layers else [], []
        bias = attention_mask[:, None, None, :].to(h.dtype).clamp_min(1e-9).log()
        for layer in self.model.encoder.layer:
            attention = layer.attention.self
            shape = (*h.shape[:2], attention.num_attention_heads, attention.attention_head_size)
            q, k, v = (projection(h).view(shape).transpose(1, 2)
                       for projection in (attention.query, attention.key, attention.value))
            if bias.requires_grad and h.is_cuda:
                with sdpa_kernel(SDPBackend.MATH):
                    context = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
            else:
                context = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
            context = context.transpose(1, 2).reshape_as(h)
            if heads:
                contexts.append(context)
            a = layer.attention.output(context, h)
            h = layer.output(layer.intermediate(a), a)
            if all_layers:
                layers.append(h)
        output = {"features": torch.stack(layers, dim=2) if all_layers else h.unsqueeze(2)}
        if heads:
            output["heads"] = torch.stack(contexts, dim=2)
        return output


def build_encoder(cfg):
    return {"bert": FrozenBERT}[cfg.family](cfg.name)


def word_features(encoded, positions):
    """Gather each word's first WordPiece: [B,T,L,D] -> [B,W,L,D]."""
    return {key: value.gather(1, positions[..., None, None].expand(
        -1, -1, value.shape[2], value.shape[3])) for key, value in encoded.items()}


def expand_gates(gates, word_ids, attention_mask):
    words = gates.gather(1, word_ids.clamp_min(0))
    return torch.where(word_ids >= 0, words, attention_mask.to(words.dtype))
