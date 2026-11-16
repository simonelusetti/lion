"""Classifiers see word vectors; representation methods live in lion.methods."""
from torch import nn


class LinearTagger(nn.Module):
    def __init__(self, dimension, num_labels):
        super().__init__()
        self.net = nn.Linear(dimension, num_labels)

    def forward(self, words):
        return self.net(words)


class MLPTagger(nn.Module):
    def __init__(self, dimension, num_labels, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, words):
        return self.net(words)


def build_tagger(cfg, dimension, num_labels):
    if cfg.name == "linear":
        return LinearTagger(dimension, num_labels)
    return MLPTagger(dimension, num_labels, cfg.hidden, cfg.dropout)
