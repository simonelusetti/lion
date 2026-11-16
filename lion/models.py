import torch
from torch import nn

from tagger import build_tagger
from .encoder import expand_gates, word_features
from .methods import build_method


class SequenceModel(nn.Module):
    def __init__(self, cfg, encoder, num_labels):
        super().__init__()
        self.encoder = encoder
        self.tagger = build_tagger(cfg.tagger, encoder.dimension, num_labels)
        self.method = build_method(cfg.method, encoder)
        self.masked = cfg.method.kind == "context_mask"
        self.use_heads = cfg.method.name in {"head_mix", "layer_head_mix"}

    @torch.no_grad()
    def prepare(self, batch):
        return word_features(self.encoder(batch["ids"], batch["attention_mask"], all_layers=False), batch["positions"])["features"][:, :, -1]

    def forward(self, features=None, heads=None, ids=None, attention_mask=None,
                positions=None, words=None, valid=None, focal=None, word_ids=None, teacher=None):
        if self.masked:
            gates = self.method(words, valid, focal, teacher)
            attention = expand_gates(gates, word_ids, attention_mask)
            encoded = self.encoder(ids, attention, all_layers=False)["features"]
            rows = torch.arange(len(ids), device=ids.device)
            h = encoded[rows, positions[rows, focal], -1]
        else:
            if features is None:
                with torch.no_grad():
                    encoded = word_features(self.encoder(ids, attention_mask, self.use_heads), positions)
                features, heads = encoded["features"], encoded.get("heads")
            h, gates = self.method(features, heads)
        return {"logits": self.tagger(h), "gates": gates}
