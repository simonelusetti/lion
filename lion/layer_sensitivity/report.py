"""Markdown report: one main table per dataset/group, one distance table, everything else in CSVs."""
import json
import shutil
import time
from ..diagnostics.common import ROOT, absolute, read_csv


def number(value, digits=4):
    return '—' if value in ('', None) or value != value else f'{float(value):.{digits}f}'


def pm(row, scale=1.0, digits=4):
    mean, sd = float(row['mean']) * scale, float(row['seed_sd'] or 0) * scale
    return '—' if mean != mean else f'{mean:.{digits}f} ± {sd:.{digits}f}'


def pair(get, *key, scale=1.0, digits=4):
    return ' / '.join(pm(get(*key, mask), scale, digits) for mask in ('selector', 'random'))


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join('---' for _ in header) + ' |', *('| ' + ' | '.join(map(str, r)) + ' |' for r in rows)])


def experiment(cfg, output):
    root = absolute(cfg.layers.results)
    analysis = root / 'analyse'
    load = lambda name: read_csv(analysis / f'{name}.csv')
    sensitivity, ratios, distance, context, check = (load(n) for n in ('sensitivity', 'ratios', 'distance', 'context', 'baseline_check'))
    performance = read_csv(root / 'train' / 'performance.csv')
    weights = read_csv(root / 'train' / 'mixture_weights.csv')
    estimate = json.loads((root / 'estimate' / 'estimate.json').read_text())
    timing = {stage: json.loads((root / stage / 'runtime.json').read_text())['seconds'] for stage in ('estimate', 'train', 'measure', 'analyse')}
    started = json.loads((root / 'timing.json').read_text())['started_unix']
    lines = ['# Context sensitivity across encoder layers', '',
             'Frozen bert-base-cased. A head reads the mean WordPiece vector of the focal word at one layer (0 = embedding output, 12 = last block), or a learned softmax-weighted scalar mixture of all 13 layers. '
             'The existing baseline classifiers and the scorer of both datasets read **layer 12** (final hidden state; source configs `method.kind=layers, method.layer=12`). Selectors, cuts, test rows and the four cached fills per row, mask kind and fold are reused unchanged; no scorer or selector was trained.', '',
             f'Estimate {estimate["total_hours"]:.2f} h; ' + ('over six hours, so single-layer heads were restricted to ' + str(estimate['layers']) + '.' if estimate['restricted'] else 'under six hours, so all 13 single-layer heads were run.') +
             ' Actual: ' + ', '.join(f'{k} {v / 60:.1f} min' for k, v in timing.items()) + f'; wall-clock {(time.time() - started) / 3600:.2f} h.', '',
             'S = KL(original ‖ perturbed), γ=.01 over native labels; flips change the argmax. Natural = cached MLM fills; masked = one [MASK] per dropped word, same dropped sets. Shift = 1 − cos at the head\'s layer (mixture: mixed vector), divided by the mean cosine distance of random word pairs (different sentences and forms) at that layer. '
             'Gain = mean S / mean normalized shift. Contextuality = mean normalized distance of each occurrence from its word type\'s test mean, types with ≥5 test occurrences. '
             'Values are mean ± SD over three seeds; single-layer shift and contextuality do not depend on the seed (SD 0). Row SEs and every non-tabled measure are in the CSVs. Draws and folds are averaged within row and never count as rows. * = fewer than 50 rows (excluded from ratios).', '',
             'Layer 0 under natural fills: fills keep the focal pieces and all piece counts, so the focal vector is unchanged up to float rounding (max cosine shift 2.4e-7; mean KL ≈1e-6). Its natural S, flips and shift are rounding noise from the bf16 head, and its natural gain and matched ratio are shown as —. Masked versions move layer 0 only through position embeddings.', '',
             'CoNLL O: the calibrated cut retains no context (ρ=0 in both folds), so the selector and random masks drop the same set (all context words) in every row. Masked versions are identical and every nearest dropped word is at distance 1; natural fills differ only by independent MLM draws. O does not contrast selector with random.', '',
             '## Existing baseline versus the new layer-12 head', '',
             table(['Dataset', 'Seed', 'Metric', 'Existing', 'New layer 12', 'Difference', 'Test prediction agreement', 'Cached vs recomputed block-12 min cos'],
                   [[r['dataset'], r['seed'], r['metric'], number(r['existing_baseline']), number(r['new_layer12_head']), number(r['difference']),
                     number(r['test_prediction_agreement']), number(r['cached_vs_recomputed_block12_min_cosine'], 6)] for r in check]), '']
    for name in cfg.layers.datasets:
        rows = [r for r in performance if r['dataset'] == name]
        heads = list(dict.fromkeys(r['head'] for r in rows))
        stats = {}
        for head in heads:
            values = [r for r in rows if r['head'] == head]
            stats[head] = {k: [float(r[k]) for r in values] for k in values[0] if k in ('validation_macro_f1', 'span_f1', 'accuracy', 'macro_f1') and values[0][k] != ''}
        mean = lambda xs: sum(xs) / len(xs)
        sd = lambda xs: (sum((x - mean(xs)) ** 2 for x in xs) / (len(xs) - 1)) ** .5
        best = max((h for h in heads if h != 'mixture'), key=lambda h: mean(stats[h]['validation_macro_f1']))
        metrics = ['span_f1'] if 'span_f1' in stats[heads[0]] else ['accuracy', 'macro_f1']
        perf = lambda h: ' / '.join(f'{100 * mean(stats[h][k]):.2f} ± {100 * sd(stats[h][k]):.2f}' for k in metrics)
        lines += [f'## {name}', '', f'Best single layer on raw-dev token macro-F1: **{best}**. Performance column: ' + ('span F1 %' if metrics == ['span_f1'] else 'accuracy % / macro-F1 %') + '.', '',
                  'Learned mixture weights (softmax) and scale, mean over seeds:', '',
                  table(['Scale', *[f'w{l}' for l in range(13)]],
                        [[f'{mean([float(w["scale"]) for w in weights if w["dataset"] == name]):.3f}',
                          *[f'{mean([float(w[f"w{l}"]) for w in weights if w["dataset"] == name]):.3f}' for l in range(13)]]]), '']
        for group in cfg.layers.datasets[name].groups:
            get = lambda head, perturbation, scope, metric, mask: next(r for r in sensitivity if (r['dataset'], r['group'], r['head'], r['perturbation'], r['scope'], r['metric'], r['mask']) == (name, group, head, perturbation, scope, metric, mask))
            ratio = lambda head, kind, perturbation, scope: next(r for r in ratios if (r['dataset'], r['group'], r['head'], r['kind'], r['perturbation'], r['scope']) == (name, group, head, kind, perturbation, scope))
            ctx = lambda head: next(r for r in context if (r['dataset'], r['group'], r['head']) == (name, group, head))
            body = []
            for head in heads:
                body.append([head + (' (best)' if head == best else ''), perf(head),
                             pair(get, head, 'natural', 'all_draws', 'S'), pair(get, head, 'natural', 'all_draws', 'flip', scale=100, digits=2),
                             '—' if head == 'layer0' else pm(ratio(head, 'matched_selector_random', 'natural', 'non_null')),
                             pair(get, head, 'masked', 'all_rows', 'S'), pair(get, head, 'masked', 'all_rows', 'flip', scale=100, digits=2),
                             pair(get, head, 'natural', 'all_draws', 'normalized_shift'), pair(get, head, 'masked', 'all_rows', 'normalized_shift'),
                             '—' if head == 'layer0' else pair(get, head, 'natural', 'all_draws', 'gain'), pair(get, head, 'masked', 'all_rows', 'gain'), pm(ctx(head))])
            lines += [f'### {name}, group {group} ({get(heads[0], "natural", "all_draws", "S", "selector")["rows"]} rows)', '',
                      table(['Head', 'Performance', 'Natural S sel / rand', 'Natural flips % sel / rand', 'Matched ratio', 'Masked S sel / rand', 'Masked flips % sel / rand',
                             'Norm. shift natural sel / rand', 'Norm. shift masked sel / rand', 'Gain natural sel / rand', 'Gain masked sel / rand', 'Contextuality'], body), '']
            chosen = list(dict.fromkeys([*(f'layer{l}' for l in cfg.layers.table_layers if f'layer{l}' in heads), best, 'mixture']))
            chosen = sorted(chosen, key=lambda h: 99 if h == 'mixture' else int(h[5:]))
            body = []
            for head in chosen:
                for b in ('1', '2', '3', '4–5', '6–10', '>10'):
                    cell = lambda metric, mask: next(r for r in distance if (r['dataset'], r['group'], r['head'], r['bin'], r['metric'], r['mask']) == (name, group, head, b, metric, mask))
                    n = cell('S', 'selector')['rows']
                    body.append([head, b + ('*' if int(n) < 50 else ''), n, ' / '.join(pm(cell('S', m)) for m in ('selector', 'random')),
                                 ' / '.join(pm(cell('flip', m), 100, 2) for m in ('selector', 'random'))])
            lines += [f'Masked perturbation by distance to the nearest dropped word ({name}, {group}); rows with at least one dropped word in the bin, common to both masks:', '',
                      table(['Head', 'Distance', 'Rows', 'S sel / rand', 'Flips % sel / rand'], body), '']
    lines += ['[Implementation](IMPLEMENTATION_13.md) · [CSV tables](tables/layers/) · [Previous: EWT dependency relations](old/12_deprel_replication/RESULTS_12.md)', '']
    tables = ROOT / 'report' / 'tables' / 'layers'
    tables.mkdir(parents=True, exist_ok=True)
    for path in [*analysis.glob('*.csv'), root / 'train' / 'performance.csv', root / 'train' / 'mixture_weights.csv', root / 'measure' / 'timing.csv']:
        shutil.copy2(path, tables / (('measure_' if path.parent.name == 'measure' else '') + path.name))
    shutil.copy2(root / 'estimate' / 'estimate.json', tables / 'estimate.json')
    (ROOT / 'report' / 'RESULTS_13.md').write_text('\n'.join(lines))
    return {'report': 'report/RESULTS_13.md'}
