"""Learning curves: every layer head and the mixture on nested subsets, at least the full-data step count."""
import copy
import math
import time
import torch
from torch.nn import functional as F
from tagger.ensemble import clip_heads
from ..augmentation.train import schedule
from ..partition.train import metric_vectors
from ..certification.classify import evaluate, span_counts
from ..runtime import autocast
from ..diagnostics.common import write_csv
from ..layer_sensitivity import data as layer_data
from ..layer_sensitivity.train import Head, MIX, head_name, logits
from . import data


def epochs_for(sentences, total, cfg_train):
    """Epoch cap giving at least as many optimizer steps as ten epochs over all sentences."""
    full_steps = cfg_train.epochs * math.ceil(total / cfg_train.batch_size)
    return math.ceil(full_steps / math.ceil(sentences / cfg_train.batch_size))


def fit(src, training, rows, validation, labels, layer, seed, epochs):
    local = copy.deepcopy(src)
    local.train.epochs = epochs
    torch.manual_seed(seed)
    model = Head(src, len(labels), layer).to(src.runtime.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=src.train.lr, weight_decay=src.train.weight_decay)
    batches, _ = schedule(training['sentence'][rows], local, seed)
    rows = rows.to(src.runtime.device)
    x, y = training['features'], training['y'].to(src.runtime.device)
    best, history = -float('inf'), []
    for number, batch in enumerate(batches):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        ids = rows[batch['baseline'].to(rows.device)]
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
        history.append({'epoch': epoch, 'step': number + 1, 'validation_macro_f1': criterion})
        if criterion > best:
            best, chosen = criterion, {'epoch': epoch, 'step': number + 1, 'validation_macro_f1': criterion}
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model, {**chosen, 'epochs_cap': epochs, 'total_steps': len(batches)}, history


def metrics(prediction_logits, test, labels, name):
    source = 'conll2003' if name == 'conll2003' else 'en_ewt'
    result = evaluate(prediction_logits[:, None], test, source, source, labels, 'native')
    return {k: float(result[k][0]) for k in ('span_f1', 'accuracy', 'macro_f1') if k in result}


def experiment(cfg, output, fractions):
    records, mixtures, timing = [], [], []
    for name in cfg.coverage.datasets:
        src = data.source(cfg, name)
        bank = data.full_bank(cfg, name)
        key = data.task(bank)
        labels = bank['labels'][key]
        prepared = torch.load(output.parent / 'prepare' / f'{name}.pt', weights_only=False)
        started = time.monotonic()
        bert = layer_data.encoder(src)
        rows = bank['sets']['train']
        training = layer_data.layer_features(bert, rows, src, labelled=False)
        validation = layer_data.layer_features(bert, bank['sets']['validation'], src)
        test = layer_data.layer_features(bert, bank['sets']['id'], src, labelled=False)
        del bert
        torch.cuda.empty_cache()
        timing.append({'dataset': name, 'run': 'features', 'seconds': time.monotonic() - started})
        # Training rows keep every word so that sentence positions line up; unlabelled words are never sampled.
        sentence_of_row = torch.repeat_interleave(torch.arange(len(rows)), torch.tensor([len(r['focals']) for r in rows]))
        training['sentence'] = sentence_of_row
        training['features'] = training['features'].to(src.runtime.device)
        validation['features'] = validation['features'].to(src.runtime.device)
        labelled = training['y'] >= 0
        for fraction in fractions:
            member = torch.zeros(len(rows), dtype=torch.bool)
            member[prepared['subsets'][fraction]] = True
            subset_rows = (member[sentence_of_row] & labelled).nonzero().flatten()
            epochs = epochs_for(int(member.sum()), len(rows), src.train)
            for layer in [*cfg.coverage.layers, MIX]:
                for seed in cfg.coverage.seeds:
                    folder = output / name / f'f{fraction}' / head_name(layer) / f'seed{seed}'
                    clock = time.monotonic()
                    if not (folder / 'result.pt').exists():
                        folder.mkdir(parents=True, exist_ok=True)
                        model, chosen, history = fit(src, training, subset_rows, validation, labels, layer, seed, epochs)
                        test_logits = logits(model, test['features'], src)
                        saved = {'chosen': chosen, 'prediction': test_logits.argmax(-1).to(torch.int16), 'metrics': metrics(test_logits, test, labels, name)}
                        if layer == MIX:
                            saved['weights'] = model.weights.detach().softmax(0).cpu()
                            saved['scale'] = float(model.scale)
                        torch.save(saved, folder / 'result.pt')
                        write_csv(folder / 'history.csv', history)
                    saved = torch.load(folder / 'result.pt', weights_only=False)
                    records.append({'dataset': name, 'fraction': fraction, 'sentences': int(member.sum()), 'head': head_name(layer), 'layer': layer, 'seed': seed,
                                    **saved['chosen'], **saved['metrics']})
                    if layer == MIX:
                        mixtures.append({'dataset': name, 'fraction': fraction, 'seed': seed, 'scale': saved['scale'],
                                         **{f'w{l}': float(saved['weights'][l]) for l in range(13)}})
                    timing.append({'dataset': name, 'run': f'f{fraction}/{head_name(layer)}/seed{seed}', 'seconds': time.monotonic() - clock})
                print(f'{name} f={fraction} {head_name(layer)}: dev macro-F1 ' + ' '.join(f'{r["validation_macro_f1"]:.4f}' for r in records[-3:]), flush=True)
                write_csv(output / 'performance.csv', records)
        torch.save({'y': test['y'], 'sentence': test['sentence']}, output / f'{name}_test.pt')
        del training, validation
        torch.cuda.empty_cache()
    write_csv(output / 'performance.csv', records)
    write_csv(output / 'mixture_weights.csv', mixtures)
    write_csv(output / 'timing.csv', timing)
    return {'runs': len(records)}
