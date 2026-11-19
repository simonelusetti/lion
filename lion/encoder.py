"""Frozen BERT with differentiable log-gate key masks and optional head contexts."""
from contextvars import ContextVar

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import BertModel, BertForMaskedLM

from .runtime import parallel


_final_positions = ContextVar("bert_final_positions", default=None)


def select_positions(hidden, positions):
    if positions.dtype == torch.bool:
        return hidden[positions]
    rows = torch.arange(len(hidden), device=hidden.device)
    return hidden[rows, positions] if positions.ndim == 1 else hidden[rows[:, None], positions]


def select_final_attention(_module, _inputs, output):
    positions = _final_positions.get()
    if positions is None:
        return output
    if isinstance(output, tuple):
        return (select_positions(output[0], positions), *output[1:])
    return select_positions(output, positions)


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


class FrozenMaskedBERT(nn.Module):
    """The same pretrained BERT backbone and its own MLM head, at requested positions."""
    def __init__(self,name):
        super().__init__()
        self.model=BertForMaskedLM.from_pretrained(name).eval().requires_grad_(False)
        last_layer = self.model.bert.encoder.layer[-1]
        self.select_before_feed_forward = last_layer.chunk_size_feed_forward == 0
        if self.select_before_feed_forward:
            # All mixing between tokens has finished before the last pointwise feed-forward block.
            last_layer.attention.register_forward_hook(select_final_attention)

    def train(self,mode=True):
        super().train(False)
        return self

    def forward(self, ids, attention_mask, positions=None, mlm=False,
                replacement_ids=None, gates=None):
        if replacement_ids is None:
            inputs = {"input_ids": ids}
        else:
            embeddings = self.model.bert.embeddings.word_embeddings
            original = embeddings(ids)
            mixed = torch.lerp(embeddings(replacement_ids), original, gates[..., None].to(original.dtype))
            inputs = {"inputs_embeds": mixed}
        request = _final_positions.set(positions if self.select_before_feed_forward else None)
        try:
            hidden = self.model.bert(**inputs, attention_mask=attention_mask, return_dict=False)[0]
        finally:
            _final_positions.reset(request)
        if positions is not None and not self.select_before_feed_forward:
            hidden = select_positions(hidden, positions)
        return {"logits": self.model.cls(hidden)} if mlm else {"features": hidden}


def build_masked_encoder(cfg):
    return {"bert":FrozenMaskedBERT}[cfg.family](cfg.name)


def load_masked_encoder(cfg):
    fp16 = cfg.runtime.device.startswith("cuda") and cfg.runtime.amp == "fp16"
    dtype = torch.float16 if fp16 else torch.float32
    encoder = build_masked_encoder(cfg.encoder).to(device=cfg.runtime.device, dtype=dtype).eval()
    return parallel(encoder, cfg, frozen=True)


def word_features(encoded, positions):
    """Gather each word's first WordPiece: [B,T,L,D] -> [B,W,L,D]."""
    return {key: value.gather(1, positions[..., None, None].expand(
        -1, -1, value.shape[2], value.shape[3])) for key, value in encoded.items()}


def expand_gates(gates, word_ids, attention_mask):
    words = gates.gather(1, word_ids.clamp_min(0))
    return torch.where(word_ids >= 0, words, attention_mask.to(words.dtype))
