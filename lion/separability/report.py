"""One table per dataset: per-word separability, cross-word transfer and direction alignment per layer."""
import json
import shutil
import time
from ..diagnostics.common import ROOT, absolute, read_csv

COLUMNS = [('per_word_balanced_accuracy', 'Per-word held-out bal. acc.', 4), ('d_prime', "Held-out d′ (top-2 labels)", 3),
           ('transfer_balanced_accuracy', 'Cross-word transfer bal. acc.', 4), ('transfer_over_per_word_above_chance', 'Transfer / per-word (above chance)', 3),
           ('direction_cosine', 'Direction cosine across words', 3)]


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join('---' for _ in header) + ' |', *('| ' + ' | '.join(map(str, r)) + ' |' for r in rows)])


def experiment(cfg, output):
    spec = cfg.separability
    root = absolute(spec.results)
    summary = read_csv(root / 'measure' / 'summary.csv')
    measure_seconds = json.loads((root / 'measure' / 'runtime.json').read_text())['seconds']
    started = json.loads((root / 'timing.json').read_text())['started_unix']
    lines = ['# Per-word linear separability and cross-word alignment, per layer', '',
             'Same eligible word types and occurrences as the label-share run (full native training sets; forms with ≥2 labels each ≥10 times; only those labels kept; CoNLL tags collapsed to entity type plus O; EWT base relations). '
             'Vectors: frozen bert-base-cased layers 0–12 and the mixed vectors of the three full-data mixture heads (coverage run, fraction 1.0); mixture numbers are per seed, then mean ± SD. '
             'Every view is z-scored per dimension over all training words before fitting, so one L2 grid applies to all layers. Nothing is trained except these probes.', '',
             f'- **Per-word separability**: for each type, a multinomial logistic regression on that type\'s occurrences only, {spec.folds}-fold stratified cross-validation; every occurrence is predicted once by a model that did not see it. Balanced accuracy (mean per-label recall) per type, aggregated weighted by occurrences.',
             '- **d′**: held-out separation-to-width along each fold\'s own dividing direction (w_a − w_b, unit length) between the type\'s two most frequent labels: |mean_a − mean_b| / √((var_a + var_b)/2) of held-out projections.',
             f'- **Cross-word transfer**: one classifier over all labels, trained on the occurrences of the other word types ({spec.folds} folds of types), applied to each held-out type with the argmax restricted to that type\'s own labels.',
             '- **Transfer / per-word (above chance)**: (transfer − chance) / (per-word − chance), chance = 1 / number of the type\'s labels, each aggregated over types.',
             '- **Direction cosine across words**: for every label pair shared by ≥2 types, the cosine between different types\' label-mean difference vectors (z-scored space), averaged over all type pairs.', '',
             f'L2 grid {list(spec.l2)}; per view, the per-word and transfer columns each use the value with the highest aggregate held-out balanced accuracy (all values in the CSV). '
             f'Measure {measure_seconds / 60:.1f} min; wall-clock {(time.time() - started) / 3600:.2f} h.', '']
    for name in spec.datasets:
        rows = [r for r in summary if r['dataset'] == name]
        single = [r for r in rows if r['view'].startswith('layer')]
        mixture = [r for r in rows if r['view'].startswith('mixture')]
        peaks = {c: max(single, key=lambda r: float(r[c]))['view'] for c, _, _ in COLUMNS}
        body = []
        for r in single:
            cells = []
            for c, _, d in COLUMNS:
                cells.append(f'{float(r[c]):.{d}f}' + (' ▲' if peaks[c] == r['view'] else ''))
            body.append([r['view'], f'{float(r["chance_balanced_accuracy"]):.4f}', *cells, f'{r["per_word_l2"]} / {r["transfer_l2"]}'])
        mean = lambda c: sum(float(r[c]) for r in mixture) / len(mixture)
        sd = lambda c: (sum((float(r[c]) - mean(c)) ** 2 for r in mixture) / (len(mixture) - 1)) ** .5
        body.append(['mixture (3 seeds)', f'{float(mixture[0]["chance_balanced_accuracy"]):.4f}', *(f'{mean(c):.{d}f} ± {sd(c):.{d}f}' for c, _, d in COLUMNS), '—'])
        lines += [f'## {name}', '', f'{rows[0]["types"]} eligible word types, {rows[0]["occurrences"]} occurrences. ▲ = peak among layers 0–12.', '',
                  table(['View', 'Chance', *(h for _, h, _ in COLUMNS), 'L2 per-word / transfer'], body), '',
                  'Peaks: ' + '; '.join(f'{h}: {peaks[c]}' for c, h, _ in COLUMNS) + '.', '']
    lines += ['Per-type values: [per_type.csv](tables/separability/per_type.csv) (20 most frequent types flagged); per label pair cosines: [label_pairs.csv](tables/separability/label_pairs.csv); all L2 values: [summary.csv](tables/separability/summary.csv).', '',
              '[Implementation](IMPLEMENTATION_16.md) · [CSV tables](tables/separability/) · [Previous: label-aligned share](old/15_label_share/RESULTS_15.md)', '']
    tables = ROOT / 'report' / 'tables' / 'separability'
    tables.mkdir(parents=True, exist_ok=True)
    for path in (root / 'measure').glob('*.csv'):
        shutil.copy2(path, tables / path.name)
    (ROOT / 'report' / 'RESULTS_16.md').write_text('\n'.join(lines))
    return {'report': 'report/RESULTS_16.md'}
