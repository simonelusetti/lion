"""Three independent CRT readouts sharing only their frozen input vectors."""
import torch
from torch import nn
from .model import build_tagger


class CRTReadouts(nn.Module):
    def __init__(self,cfg,classes,seed):
        super().__init__()
        self.models=nn.ModuleList()
        for labels in classes:
            torch.manual_seed(seed)
            self.models.append(build_tagger(cfg,768,labels))

    def forward(self,x):
        return [model(x) for model in self.models]
