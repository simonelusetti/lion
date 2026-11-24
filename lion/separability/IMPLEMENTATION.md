# Per-word linear separability and cross-word alignment, per layer

Run: `scripts/run_separability.sh` (`python -B -m lion.separability.run`). Config:
`conf/separability.yaml`; stages measure and report are Forge/Hydra runs.
Raw results: `results/separability/`; report: `report/RESULTS_16.md`.

Question: does the best layer win by dividing each ambiguous word's label piles
better (per-word separability), or by lining up different words' piles
(cross-word transfer and direction alignment)?

## Data

Identical to the label-share run: full native training sets (CoNLL 14,041
sentences; EWT 12,544, uncapped bank), case-sensitive word forms from WordPiece IDs,
CoNLL BIO tags collapsed to entity type plus O, EWT base relations. Eligible types:
forms with at least two labels that each occur ≥10 times; only those labels'
occurrences are kept. Vectors: frozen bert-base-cased, mean WordPiece vector per word,
layers 0–12, plus the mixed vectors of the three full-data mixture heads of the
coverage run (fraction 1.0); mixture results are per seed, then mean ± SD.

Each view is z-scored per dimension (mean and SD over all training words of that
view) before any probe is fitted, so one L2 grid serves every layer.

## Probes

- **Per-word separability.** For each type, multinomial logistic regression over its
  own labels, fitted only on its own occurrences. Folds: within each (type, label)
  group a random order is assigned round-robin to 5 folds (every label has ≥2
  occurrences per fold). All types are fitted together as one objective, the sum of each
  type's mean cross-entropy plus L2 on the weights, so the fits are independent.
  Full-batch L-BFGS (strong Wolfe, ≤200 iterations). Score: per-type balanced accuracy
  (mean per-label recall) over held-out predictions, every occurrence predicted
  exactly once.
- **d′ (separation-to-width).** For each fold and type, the unit direction
  w_a − w_b between the weight vectors of the type's two most frequent labels.
  Held-out occurrences of a and b are projected on their own fold's direction;
  d′ = |mean_a − mean_b| / √((var_a + var_b)/2).
- **Cross-word transfer.** Types are split randomly into 5 folds. One multinomial
  logistic regression over all labels is trained on the occurrences of the other
  folds' types (mean cross-entropy + L2) and applied to each held-out type, with the
  argmax restricted to that type's own labels. Score: the same balanced accuracy.
- **Direction alignment.** For every label pair (a, b) shared by at least two types,
  each type's unit vector μ_wa − μ_wb (full data, z-scored space); the mean cosine over all
  pairs of different types, pooled across label pairs.

L2 grid {0.001, 0.01, 0.1}. In the main table, per-word columns (accuracy, d′) use the
value with the highest occurrence-weighted per-word balanced accuracy for that view,
and transfer uses its own best value. All values are in `summary.csv`. Choosing among
three values on the evaluation folds biases every layer upward by the same procedure.

Aggregates over types weight by occurrence counts. Chance = 1 / number of the type's
labels. Transfer / per-word (above chance) = (transfer − chance) / (per-word − chance),
each aggregated first.
