"""One table per dataset: layers 0–12 and the seed-averaged mixture."""
import json
import shutil
import time
from ..diagnostics.common import ROOT, absolute, read_csv

COLUMNS = ['normalized_total_spread', 'share', 'baseline', 'share_minus_baseline', 'standardized_share', 'standardized_share_minus_baseline']


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join('---' for _ in header) + ' |', *('| ' + ' | '.join(map(str, r)) + ' |' for r in rows)])


def experiment(cfg, output):
    root = absolute(cfg.label_share.results)
    summary = read_csv(root / 'measure' / 'summary.csv')
    timing = {s: json.loads((root / s / 'runtime.json').read_text())['seconds'] for s in ('measure',)}
    started = json.loads((root / 'timing.json').read_text())['started_unix']
    k = cfg.label_share.minimum_per_label
    lines = ['# Label-aligned share of contextual variation, per layer', '',
             'Frozen bert-base-cased word vectors (mean of WordPieces) for every word of the full native training set, layers 0–12; plus the mixed vector of the three full-data mixture heads from the coverage run (fraction 1.0, seeds 0–2), whose numbers are computed per seed and then averaged. Nothing is trained.', '',
             f'Labels: CoNLL BIO tags collapsed to entity type plus O; EWT base dependency relations. Eligible word types: exact case-sensitive forms with at least two labels that each occur ≥{k} times in training; a type keeps only the occurrences of those labels. '
             'Per type w: T_w = mean squared distance of occurrences from the type mean; B_w = Σ_y p(y|w)·‖μ_wy − μ_w‖²; share = B_w/T_w. '
             f'Baseline: share with labels shuffled within the type (label counts kept), averaged over {cfg.label_share.shuffles} shuffles. Types are aggregated weighted by occurrences. '
             f'Total spread is divided by the mean squared distance between {cfg.label_share.random_pairs} random training-word pairs from different sentences with different forms, at the same layer. '
             '"Standardized" repeats the computation after z-scoring each dimension over all training words of that layer.', '',
             'Measure: ' + ', '.join(f'{k} {v / 60:.1f} min' for k, v in timing.items()) + f'; wall-clock {(time.time() - started) / 3600:.2f} h.', '']
    for name in cfg.label_share.datasets:
        rows = [r for r in summary if r['dataset'] == name]
        single = [r for r in rows if r['view'].startswith('layer')]
        mixture = [r for r in rows if r['view'].startswith('mixture')]
        averaged = {c: sum(float(r[c]) for r in mixture) / len(mixture) for c in COLUMNS}
        sd = {c: (sum((float(r[c]) - averaged[c]) ** 2 for r in mixture) / (len(mixture) - 1)) ** .5 for c in COLUMNS}
        peak = max(single, key=lambda r: float(r['share_minus_baseline']))['view']
        peak_std = max(single, key=lambda r: float(r['standardized_share_minus_baseline']))['view']
        body = [[r['view'] + (' (peak)' if r['view'] == peak else '') + (' (standardized peak)' if r['view'] == peak_std else ''), *(f'{float(r[c]):.4f}' for c in COLUMNS)] for r in single]
        body.append(['mixture (mean ± SD of 3 seeds)', *(f'{averaged[c]:.4f} ± {sd[c]:.4f}' for c in COLUMNS)])
        lines += [f'## {name}', '', f'{rows[0]["types"]} eligible word types, {rows[0]["occurrences"]} occurrences.', '',
                  table(['View', 'Total spread / random pairs', 'Share', 'Permutation baseline', 'Share − baseline', 'Standardized share', 'Standardized share − baseline'], body), '',
                  *([f'Types with no spread at a view (identical occurrences, share undefined) are left out of that view\'s aggregates: ' + '; '.join(f'{r["view"]}: {r["undefined_types"]} raw / {r["standardized_undefined_types"]} standardized' for r in rows if int(r['undefined_types']) or int(r['standardized_undefined_types'])) + '.', ''] if any(int(r['undefined_types']) or int(r['standardized_undefined_types']) for r in rows) else []),
                  f'Share − baseline peaks at **{peak}** (raw) and **{peak_std}** (standardized) among layers 0–12; the mixture is {averaged["share_minus_baseline"]:.4f} (raw) and {averaged["standardized_share_minus_baseline"]:.4f} (standardized).', '']
    lines += ['Per-word-type values for every view, with the most frequent types flagged: [per_type.csv](tables/label_share/per_type.csv). Raw distances and pair references: [summary.csv](tables/label_share/summary.csv).', '',
              '[Implementation](IMPLEMENTATION_15.md) · [CSV tables](tables/label_share/) · [Previous: coverage](old/14_coverage/RESULTS_14.md)', '']
    tables = ROOT / 'report' / 'tables' / 'label_share'
    tables.mkdir(parents=True, exist_ok=True)
    for path in (root / 'measure').glob('*.csv'):
        shutil.copy2(path, tables / path.name)
    (ROOT / 'report' / 'RESULTS_15.md').write_text('\n'.join(lines))
    return {'report': 'report/RESULTS_15.md'}
