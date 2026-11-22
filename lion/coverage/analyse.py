"""Frequency-bin accuracy and paired gaps; learning-curve summaries and gaps."""
from collections import Counter
import torch
from ..diagnostics.common import absolute, write_csv, read_csv
from . import data

BINS = ['0', '1–2', '3–9', '10–99', '100+']


def stats(x, mask):
    """x [seeds, rows]: mean ± seed SD of per-seed row means; row SE of seed-averaged rows."""
    values = x[:, mask].double()
    n = values.shape[1]
    point = values.mean(1) if n else torch.full((x.shape[0],), torch.nan)
    return {'rows': n, 'mean': float(point.mean()), 'seed_sd': float(point.std()) if len(point) > 1 else 0.0,
            'row_se': float(values.mean(0).std() / n ** .5) if n > 1 else float('nan')}


def best_layer(rows, key='validation_macro_f1'):
    means = {}
    for r in rows:
        if r['head'] != 'mixture':
            means.setdefault(r['head'], []).append(float(r[key]))
    return max(means, key=lambda h: sum(means[h]) / len(means[h]))


def frequency_tables(dataset, scope, forms, gold, test_forms, test_gold, labels, predictions, best, edges):
    """predictions: {head: [seeds, rows] predicted labels}. Gaps are per row, then averaged."""
    count = Counter(forms)
    seen = {}
    for form, label in zip(forms, gold.tolist()):
        if label >= 0:
            seen.setdefault(form, set()).add(label)
    frequency = torch.tensor([count.get(f, 0) for f in test_forms])
    bins = torch.bucketize(frequency, torch.tensor(edges[1:]), right=True)
    other = torch.tensor([bool(seen.get(f)) and label not in seen[f] for f, label in zip(test_forms, test_gold.tolist())])
    valid = test_gold >= 0
    groups = {'gold_O': test_gold == labels.index('O'), 'gold_entity': (test_gold != labels.index('O')) & valid} if 'O' in labels else {'all': valid}
    correct = {h: (p == test_gold[None]).double() for h, p in predictions.items()}
    accuracy, gaps, cross = [], [], []
    for group, gmask in groups.items():
        for b, label in enumerate(BINS):
            mask = gmask & (bins == b)
            base = {'dataset': dataset, 'scope': scope, 'group': group, 'bin': label, 'best_layer': best}
            for head, c in correct.items():
                accuracy.append({**base, 'head': head, **stats(c, mask)})
            for name, head in (('best_minus_layer12', best), ('mixture_minus_layer12', 'mixture')):
                gaps.append({**base, 'gap': name, **stats(correct[head] - correct['layer12'], mask)})
            row = {**base, 'rows': int(mask.sum()), 'rows_seen_only_with_other_labels': int((mask & other).sum())}
            for head, c in correct.items():
                row[f'{head}_error_seen_other'] = float(1 - c[:, mask & other].mean()) if (mask & other).any() else float('nan')
                row[f'{head}_error_all'] = float(1 - c[:, mask].mean()) if mask.any() else float('nan')
            cross.append(row)
    return accuracy, gaps, cross


def experiment(cfg, output):
    root = absolute(cfg.coverage.results)
    layer_root = absolute(cfg.coverage.layer_run) / 'train'
    performance = read_csv(root / 'train' / 'performance.csv')
    accuracy, gaps, cross, curve, curve_gaps, best_rows = [], [], [], [], [], []
    for name in cfg.coverage.datasets:
        prepared = torch.load(root / 'prepare' / f'{name}.pt', weights_only=False)
        labels = prepared['labels']
        # Part A: layer-run heads trained on T1∪T2.
        best = best_layer([r for r in read_csv(layer_root / 'performance.csv') if r['dataset'] == name])
        heads = list(dict.fromkeys([*(f'layer{l}' for l in cfg.coverage.part_a_layers), best, 'mixture']))
        predictions = {h: torch.stack([torch.load(layer_root / name / h / f'seed{s}' / 'test.pt')['logits'].argmax(-1) for s in cfg.coverage.seeds]) for h in heads}
        a, g, c = frequency_tables(name, 'part_a_T1uT2', prepared['t12_forms'], prepared['t12_gold'], prepared['test_forms'], prepared['test_gold'], labels, predictions, best, list(cfg.coverage.bins))
        accuracy += a; gaps += g; cross += c
        # Part B: per fraction.
        rows = [r for r in performance if r['dataset'] == name]
        metrics = ['span_f1', 'accuracy'] if name == 'conll2003' else ['accuracy', 'macro_f1']
        for fraction in sorted({float(r['fraction']) for r in rows}):
            here = [r for r in rows if float(r['fraction']) == fraction]
            best_f = best_layer(here)
            best_rows.append({'dataset': name, 'fraction': fraction, 'sentences': here[0]['sentences'], 'best_layer': best_f})
            for head in dict.fromkeys(r['head'] for r in here):
                values = [r for r in here if r['head'] == head]
                record = {'dataset': name, 'fraction': fraction, 'head': head, 'best_by_dev': head == best_f,
                          'selected_step_mean': sum(float(r['step']) for r in values) / len(values),
                          'total_steps': values[0]['total_steps'], 'epochs_cap': values[0]['epochs_cap']}
                for m in metrics + ['validation_macro_f1']:
                    x = torch.tensor([float(r[m]) for r in values], dtype=torch.float64)
                    record[f'{m}_mean'], record[f'{m}_seed_sd'] = float(x.mean()), float(x.std())
                curve.append(record)
            by = lambda head, m: torch.tensor([float(r[m]) for r in sorted((r for r in here if r['head'] == head), key=lambda r: int(r['seed']))], dtype=torch.float64)
            for gap, head in (('best_minus_layer12', best_f), ('mixture_minus_layer12', 'mixture')):
                for m in metrics:
                    d = by(head, m) - by('layer12', m)
                    curve_gaps.append({'dataset': name, 'fraction': fraction, 'best_layer': best_f, 'gap': gap, 'metric': m, 'mean': float(d.mean()), 'seed_sd': float(d.std())})
            # Frequency breakdown with frequencies counted in this fraction's subset (CSV only).
            index = prepared['subsets'][fraction].tolist()
            forms = [w for i in index for w in prepared['full_forms'][i]]
            gold = torch.cat([prepared['full_gold'][i] for i in index])
            fheads = list(dict.fromkeys([*(f'layer{l}' for l in cfg.coverage.part_a_layers), best_f, 'mixture']))
            preds = {h: torch.stack([torch.load(root / 'train' / name / f'f{fraction}' / h / f'seed{s}' / 'result.pt', weights_only=False)['prediction'].long() for s in cfg.coverage.seeds]) for h in fheads}
            a, g, c = frequency_tables(name, f'part_b_f{fraction}', forms, gold, prepared['test_forms'], prepared['test_gold'], labels, preds, best_f, list(cfg.coverage.bins))
            accuracy += a; gaps += g; cross += c
        print(f'Analysed {name}', flush=True)
    for key, values in (('frequency_accuracy', accuracy), ('frequency_gaps', gaps), ('frequency_crosstab', cross),
                        ('curve_performance', curve), ('curve_gaps', curve_gaps), ('curve_best_layer', best_rows)):
        write_csv(output / f'{key}.csv', values)
    return {'gap_rows': len(gaps), 'curve_rows': len(curve)}
