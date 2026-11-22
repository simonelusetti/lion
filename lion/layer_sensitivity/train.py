"""Matched baseline recipe, one head per encoder layer plus a learned scalar mixture."""
import time
import torch
from torch import nn
from torch.nn import functional as F
from tagger.ensemble import IndependentReadouts, clip_heads
from ..augmentation.train import schedule
from ..partition.train import metric_vectors
from ..certification.classify import evaluate
from ..runtime import autocast
from ..diagnostics.common import absolute, write_csv, save_json
from . import data

MIX = 13  # head index used for the scalar mixture


class Head(nn.Module):
    """A layer selector (or softmax-weighted mixture with a scale) in front of the baseline MLP."""
    def __init__(self, src, labels, layer):
        super().__init__()
        self.layer = layer
        self.core = IndependentReadouts(src.tagger, 768, labels, 1)
        if layer == MIX:
            self.weights = nn.Parameter(torch.zeros(13))
            self.scale = nn.Parameter(torch.ones(()))

    def vector(self, x):
        if self.layer != MIX:
            return x[:, self.layer].float()
        return self.scale * torch.einsum('nld,l->nd', x.float(), self.weights.softmax(0))

    def forward(self, x):
        return self.core(self.vector(x)[:, None])[:, 0]


def head_name(layer):
    return 'mixture' if layer == MIX else f'layer{layer}'


@torch.no_grad()
def logits(model, x, src, size=8192):
    model.eval()
    parts = []
    for part in x.split(size):
        with autocast(src):
            parts.append(model(part.to(src.runtime.device)).float().cpu())
    return torch.cat(parts)


def fit(src, training, validation, labels, layer, seed):
    torch.manual_seed(seed)
    model = Head(src, len(labels), layer).to(src.runtime.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=src.train.lr, weight_decay=src.train.weight_decay)
    batches, _ = schedule(training['sentence'], src, seed)
    x, y = training['features'], training['y'].to(src.runtime.device)
    best, history = -float('inf'), []
    for number, batch in enumerate(batches):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        ids = batch['baseline'].to(x.device)
        with autocast(src):
            output = model(x[ids]).float()
        F.cross_entropy(output, y[ids]).backward()
        clip_heads(model.core, src.train.clip)
        optimizer.step()
        epoch = batch['epoch']
        if number + 1 < len(batches) and batches[number + 1]['epoch'] == epoch:
            continue
        scores = metric_vectors(logits(model, validation['features'], src)[:, None], validation['y'], len(labels))
        criterion = float(scores['macro_f1'][0])
        history.append({'epoch': epoch, 'validation_macro_f1': criterion, 'validation_accuracy': float(scores['accuracy'][0])})
        if criterion > best:
            best, best_epoch = criterion, epoch
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model, {'epoch': best_epoch, 'validation_macro_f1': best}, history


def experiment(cfg, output):
    records, mixtures, timing = [], [], []
    for name in cfg.layers.datasets:
        folder = output / name
        folder.mkdir(parents=True, exist_ok=True)
        src = data.source(cfg, name)
        sentences = data.bank(cfg, name)
        task = data.data.task_names(src)[0]
        labels = sentences['labels'][task]
        started = time.monotonic()
        bert = data.encoder(src)
        rows = sentences['sets']['train']
        training = data.layer_features(bert, [r for r in rows if r['fold'] in ('g1', 'g2')], src)
        validation = data.layer_features(bert, sentences['sets']['validation'], src)
        if not (folder / 'test_features.pt').exists():
            torch.save(data.layer_features(bert, sentences['sets']['id'], src, labelled=False), folder / 'test_features.pt')
        del bert
        torch.cuda.empty_cache()
        timing.append({'dataset': name, 'step': 'features', 'seconds': time.monotonic() - started})
        test = torch.load(folder / 'test_features.pt', weights_only=False)
        training['features'] = training['features'].to(src.runtime.device)
        source = 'conll2003' if task == 'ner' else 'en_ewt'
        for layer in [*cfg.layers.layers, MIX]:
            for seed in cfg.layers.seeds:
                target = folder / head_name(layer) / f'seed{seed}'
                started = time.monotonic()
                if not (target / 'model.pt').exists():
                    target.mkdir(parents=True, exist_ok=True)
                    model, chosen, history = fit(src, training, validation, labels, layer, seed)
                    torch.save({'state': model.state_dict(), 'layer': layer, 'labels': len(labels), **chosen}, target / 'model.pt')
                    write_csv(target / 'history.csv', history)
                    torch.save({'logits': logits(model, test['features'], src)}, target / 'test.pt')
                saved = torch.load(target / 'model.pt', weights_only=False)
                result = evaluate(torch.load(target / 'test.pt')['logits'][:, None], test, source, source, labels, 'native')
                records.append({'dataset': name, 'head': head_name(layer), 'layer': layer, 'seed': seed, 'epoch': saved['epoch'],
                                'validation_macro_f1': saved['validation_macro_f1'], **{k: float(v[0]) for k, v in result.items()}})
                if layer == MIX:
                    weights = saved['state']['weights'].softmax(0)
                    mixtures.append({'dataset': name, 'seed': seed, 'scale': float(saved['state']['scale']),
                                     **{f'w{l}': float(weights[l]) for l in range(13)}})
                timing.append({'dataset': name, 'step': f'{head_name(layer)}/seed{seed}', 'seconds': time.monotonic() - started})
                print(f'{name} {head_name(layer)} seed{seed}: dev macro-F1 {saved["validation_macro_f1"]:.4f}', flush=True)
        del training
        torch.cuda.empty_cache()
    write_csv(output / 'performance.csv', records)
    write_csv(output / 'mixture_weights.csv', mixtures)
    write_csv(output / 'timing.csv', timing)
    return {'heads': len(records)}
