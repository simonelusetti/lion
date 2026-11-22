"""All-layer focal vectors for original text, cached fills and one-[MASK]-per-word versions."""
import torch
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
from ..certification import data
from ..encoder import load_masked_encoder
from ..runtime import autocast
from ..diagnostics.common import absolute

FOLDS = ('g1', 'g2')
KINDS = ('selector', 'random')
VERSIONS = 5  # four cached fills, then the masked version


def source(cfg, name):
    src = OmegaConf.load(absolute(cfg.layers.datasets[name].source_config))
    src.runtime = OmegaConf.merge(src.runtime, cfg.runtime)
    return src


def bank(cfg, name):
    return torch.load(absolute(cfg.layers.datasets[name].source_config).parent / 'prepare' / 'sentences.pt', weights_only=False)


def fills(cfg, name):
    folder = absolute(cfg.layers.datasets[name].fills)
    return [torch.load(folder / f'{fold}.pt', mmap=True, weights_only=False) for fold in FOLDS]


def encoder(src):
    # The same frozen BERT backbone that produced the cached block-12 vectors.
    model = load_masked_encoder(src)
    return getattr(model, 'module', model).model.bert


def hidden_states(bert, ids, attention, src):
    with autocast(src):
        return bert(input_ids=ids, attention_mask=attention, output_hidden_states=True, return_dict=True).hidden_states


@torch.no_grad()
def layer_features(bert, rows, src, labelled=True, dtype=torch.float16):
    """Mean WordPiece vector of every word at embeddings + 12 blocks: [words, 13, 768]."""
    parts = []
    device = src.runtime.device
    for batch in data.loader(rows, src):
        ids, attention, words = (batch[k].to(device) for k in ('ids', 'attention', 'word_ids'))
        hidden = hidden_states(bert, ids, attention, src)
        owner, focal = batch['owner'].to(device), batch['focal'].to(device)
        width = int(batch['focal'].max()) + 1
        parts.append(torch.stack([data.pool_words(h, words, width)[owner, focal] for h in hidden], 1).to('cpu', dtype))
    features = torch.cat(parts)
    task = next(iter(rows[0]['gold']))
    y = torch.cat([row['gold'][task] for row in rows])
    sentence = torch.repeat_interleave(torch.tensor([row['sentence'] for row in rows]), torch.tensor([len(row['focals']) for row in rows]))
    keep = y >= 0 if labelled else torch.ones_like(y, dtype=torch.bool)
    return {'features': features[keep], 'y': y[keep], 'sentence': sentence[keep]}


def row_pieces(cache):
    lengths = cache['piece_lengths'].tolist()
    return cache['original_ids'].long().split(lengths), cache['word_ids'].long().split(lengths)


def nearest_dropped(cache, kind):
    """Distance from the focal word to the closest dropped word; -1 when nothing is dropped."""
    length = cache['word_lengths']
    owner = torch.repeat_interleave(torch.arange(len(length)), length)
    position = torch.arange(int(length.sum())) - cache['word_offsets'][owner]
    distance = (position - cache['focal'][owner]).abs().masked_fill(~cache[kind + '_drop_words'], 1 << 30)
    nearest = torch.full((len(length),), 1 << 30).scatter_reduce(0, owner, distance, 'amin')
    return nearest.masked_fill(nearest == 1 << 30, -1)


def perturbed(cache, kind, pieces, mask_id):
    """Sequences that differ from the original: (row, version, ids, word_ids)."""
    originals, word_ids = pieces
    tokens = cache[kind + '_token_ids'].long().split((cache['piece_lengths'] * 4).tolist())
    drops = cache[kind + '_drop_words'].split(cache['word_lengths'].tolist())
    items = []
    for row, (ids, words, drawn, drop) in enumerate(zip(originals, word_ids, tokens, drops)):
        for version, fill in enumerate(drawn.view(4, -1)):
            if not torch.equal(fill, ids):
                items.append((row, version, fill, words))
        if drop.any():
            dropped = (words >= 0) & drop[words.clamp_min(0)]
            first = torch.cat([torch.ones(1, dtype=torch.bool), words[1:] != words[:-1]])
            keep = ~dropped | first
            items.append((row, 4, ids.masked_fill(dropped, mask_id)[keep], words[keep]))
    return items


@torch.no_grad()
def encode(bert, items, focal, src, budget):
    """Stream focal vectors at all layers for length-sorted chunks: yields (rows, versions, [n,13,768])."""
    device = src.runtime.device
    order = sorted(range(len(items)), key=lambda i: len(items[i][2]))
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and len(items[order[end]][2]) * (end - start + 1) <= budget:
            end += 1
        chunk = [items[i] for i in order[start:end]]
        start = end
        rows = torch.tensor([c[0] for c in chunk])
        lengths = torch.tensor([len(c[2]) for c in chunk], device=device)
        ids = pad_sequence([c[2] for c in chunk], batch_first=True).to(device)
        words = pad_sequence([c[3] for c in chunk], batch_first=True, padding_value=-1).to(device)
        attention = torch.arange(ids.shape[1], device=device)[None] < lengths[:, None]
        hidden = hidden_states(bert, ids, attention, src)
        weight = (words == focal[rows].to(device)[:, None]).float()
        weight = weight / weight.sum(-1, keepdim=True)
        vectors = torch.stack([torch.einsum('bt,btd->bd', weight, h.float()) for h in hidden], 1)
        yield rows, torch.tensor([c[1] for c in chunk]), vectors
