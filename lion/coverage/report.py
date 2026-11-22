"""Markdown report: frequency-bin gaps (part A) and learning-curve gaps (part B)."""
import json
import shutil
import time
from ..diagnostics.common import ROOT, absolute, read_csv
from .analyse import BINS


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join('---' for _ in header) + ' |', *('| ' + ' | '.join(map(str, r)) + ' |' for r in rows)])


def pct(row, se=True):
    text = f'{100 * float(row["mean"]):+.2f} ± {100 * float(row["seed_sd"]):.2f}' if row['mean'] != 'nan' else '—'
    return text + (f' ({100 * float(row["row_se"]):.2f})' if se and row.get('row_se') not in (None, '', 'nan') else '')


def level(row):
    return '—' if row['mean'] == 'nan' else f'{100 * float(row["mean"]):.2f} ± {100 * float(row["seed_sd"]):.2f}'


def monotone(values):
    """Non-increasing along the given order."""
    return 'yes' if all(b <= a for a, b in zip(values, values[1:])) else 'no'


def experiment(cfg, output):
    root = absolute(cfg.coverage.results)
    analysis = root / 'analyse'
    load = lambda n: read_csv(analysis / f'{n}.csv')
    accuracy, gaps, curve, curve_gaps, best = (load(n) for n in ('frequency_accuracy', 'frequency_gaps', 'curve_performance', 'curve_gaps', 'curve_best_layer'))
    weights = read_csv(root / 'train' / 'mixture_weights.csv')
    subsets = read_csv(root / 'prepare' / 'subsets.csv')
    estimate = json.loads((root / 'estimate' / 'estimate.json').read_text())
    timing = {s: json.loads((root / s / 'runtime.json').read_text())['seconds'] for s in ('prepare', 'estimate', 'train', 'analyse')}
    started = json.loads((root / 'timing.json').read_text())['started_unix']
    lines = ['# Is the layer gap a coverage effect?', '',
             'Frozen bert-base-cased; heads read the mean WordPiece vector at one layer or the learned scalar mixture, with the existing head recipe (MLP 768→256, AdamW, 64-sentence batches, raw-dev token macro-F1 checkpoint). No scorer, selector, fills or certificates. Native dev selects checkpoints and native test evaluates every run.', '',
             f'Estimate {estimate["total_hours"]:.2f} h; ' + ('over six hours, so the 1/16 fraction was dropped.' if estimate['dropped_smallest_fraction'] else 'under six hours, so all five fractions were run.') +
             ' Actual: ' + ', '.join(f'{k} {v / 60:.1f} min' for k, v in timing.items()) + f'; wall-clock {(time.time() - started) / 3600:.2f} h.', '',
             'Training frequency = occurrences of the test word’s exact case-sensitive form in the head’s training set. Accuracy is token accuracy over native tags. Gaps are computed per test row (correct − correct), then averaged. Values: mean ± seed SD (row SE) over three seeds, in percentage points. "Monotonically decreasing" means non-increasing along the stated order.', '',
             '## Part A: gaps by training frequency (layer-run heads, trained on T1∪T2)', '']
    for name in cfg.coverage.datasets:
        scope = [r for r in gaps if r['dataset'] == name and r['scope'] == 'part_a_T1uT2']
        best_layer = scope[0]['best_layer']
        acc = lambda group, b, head: next(r for r in accuracy if (r['dataset'], r['scope'], r['group'], r['bin'], r['head']) == (name, 'part_a_T1uT2', group, b, head))
        gap = lambda group, b, g: next(r for r in scope if (r['group'], r['bin'], r['gap']) == (group, b, g))
        body, statements = [], []
        for group in dict.fromkeys(r['group'] for r in scope):
            for b in BINS:
                body.append([group, b, gap(group, b, 'best_minus_layer12')['rows'], level(acc(group, b, 'layer12')), level(acc(group, b, best_layer)), level(acc(group, b, 'mixture')),
                             pct(gap(group, b, 'best_minus_layer12')), pct(gap(group, b, 'mixture_minus_layer12'))])
            for g in ('best_minus_layer12', 'mixture_minus_layer12'):
                values = [float(gap(group, b, g)['mean']) for b in BINS if int(gap(group, b, g)['rows'])]
                statements.append(f'{group}, {g.replace("_", " ")}: {monotone(values)}')
        lines += [f'### {name} (best single layer in the layer run: {best_layer})', '',
                  table(['Group', 'Training frequency', 'Rows', 'Acc. layer 12 %', f'Acc. {best_layer} %', 'Acc. mixture %', f'{best_layer} − layer 12', 'Mixture − layer 12'], body), '',
                  f'**{name}: gap decreases monotonically with training frequency (0 → 100+)?** ' + '; '.join(statements) + '.', '']
    lines += ['Per-bin row counts and each head’s error on words seen in training only with other labels: [frequency_crosstab.csv](tables/coverage/frequency_crosstab.csv).', '',
              '## Part B: gaps by training fraction', '',
              'Nested random sentence subsets of the full native training set (seed 0). Every run gets at least as many optimizer steps as ten epochs on the full set: the epoch cap is scaled up for small fractions and the best dev checkpoint is kept. The best single layer is chosen by mean dev macro-F1 at each fraction.', '']
    for name in cfg.coverage.datasets:
        metrics = ['span_f1', 'accuracy'] if name == 'conll2003' else ['accuracy', 'macro_f1']
        here = [r for r in curve if r['dataset'] == name]
        fractions = sorted({float(r['fraction']) for r in here})
        cell = lambda f, head, m: next(r for r in here if float(r['fraction']) == f and r['head'] == head)
        g = lambda f, gap, m: next(r for r in curve_gaps if r['dataset'] == name and float(r['fraction']) == f and r['gap'] == gap and r['metric'] == m)
        body = []
        for f in fractions:
            best_f = next(r['best_layer'] for r in best if r['dataset'] == name and float(r['fraction']) == f)
            sentences = next(r['sentences'] for r in best if r['dataset'] == name and float(r['fraction']) == f)
            row = [f'{f:g}', sentences, best_f, f'{float(cell(f, best_f, None)["selected_step_mean"]):.0f} / {cell(f, best_f, None)["total_steps"]}']
            for m in metrics:
                lp = lambda head: f'{100 * float(cell(f, head, m)[m + "_mean"]):.2f} ± {100 * float(cell(f, head, m)[m + "_seed_sd"]):.2f}'
                row += [lp('layer12'), lp(best_f), lp('mixture'), pct(g(f, 'best_minus_layer12', m), se=False), pct(g(f, 'mixture_minus_layer12', m), se=False)]
            body.append(row)
        header = ['Fraction', 'Sentences', 'Best layer', 'Best-layer selected step / steps']
        for m in metrics:
            label = {'span_f1': 'span F1', 'accuracy': 'acc.', 'macro_f1': 'macro-F1'}[m]
            header += [f'L12 {label}', f'Best {label}', f'Mixture {label}', f'Best − L12 {label}', f'Mixture − L12 {label}']
        statements = [f'{gap.replace("_", " ")}, {m}: {monotone([float(g(f, gap, m)["mean"]) for f in fractions])}' for gap in ('best_minus_layer12', 'mixture_minus_layer12') for m in metrics]
        primary = metrics[0] if name == 'conll2003' else 'macro_f1'
        heads = list(dict.fromkeys(r['head'] for r in here))
        per_head = [[h, *(f'{100 * float(cell(f, h, primary)[primary + "_mean"]):.2f} ± {100 * float(cell(f, h, primary)[primary + "_seed_sd"]):.2f}' for f in fractions)] for h in heads]
        mix = [[f'{f:g}', *(f'{sum(float(w[k]) for w in weights if w["dataset"] == name and float(w["fraction"]) == f) / len(cfg.coverage.seeds):.3f}' for k in ['scale', *(f'w{l}' for l in range(13))])] for f in fractions]
        lines += [f'### {name}', '', table(header, body), '',
                  f'**{name}: gap decreases monotonically with training size (smallest → full)?** ' + '; '.join(statements) + '.', '',
                  f'Test {primary.replace("_", " ")} % per head and fraction (all metrics and selected steps per head in [curve_performance.csv](tables/coverage/curve_performance.csv)):', '',
                  table(['Head', *(f'{f:g}' for f in fractions)], per_head), '',
                  'Learned mixture weights (softmax), mean over seeds:', '',
                  table(['Fraction', 'Scale', *(f'w{l}' for l in range(13))], mix), '']
    lines += ['Subset sizes: ' + '; '.join(f'{r["dataset"]} {r["fraction"]}: {r["sentences"]} sentences / {r["words"]} words' for r in subsets) + '.', '',
              'Per-fraction frequency breakdowns (frequencies counted in that subset) are in [frequency_gaps.csv](tables/coverage/frequency_gaps.csv) and [frequency_accuracy.csv](tables/coverage/frequency_accuracy.csv).', '',
              '[Implementation](IMPLEMENTATION_14.md) · [CSV tables](tables/coverage/) · [Previous: layer sensitivity](old/13_layer_sensitivity/RESULTS_13.md)', '']
    tables = ROOT / 'report' / 'tables' / 'coverage'
    tables.mkdir(parents=True, exist_ok=True)
    for path in [*analysis.glob('*.csv'), root / 'train' / 'performance.csv', root / 'train' / 'mixture_weights.csv', root / 'train' / 'timing.csv', root / 'prepare' / 'subsets.csv']:
        shutil.copy2(path, tables / path.name)
    shutil.copy2(root / 'estimate' / 'estimate.json', tables / 'estimate.json')
    (ROOT / 'report' / 'RESULTS_14.md').write_text('\n'.join(lines))
    return {'report': 'report/RESULTS_14.md'}
