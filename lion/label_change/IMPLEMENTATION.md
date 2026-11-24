# Matched pilot: context-changing, focal-label-changing augmentation

Run `bash scripts/run_label_change.sh`. Forge/Hydra stages: prepare, scorer,
generate, certify, train, report. Config `conf/label_change.yaml`.
Results `results/label_change/`; report `report/RESULTS_18.md`.

- Fixed seed-0 3,000-sentence CoNLL native-training pilot, matching the previous
  pilot scale. Reserve 600 sentences using seed 20261123; exclude them from
  scorer fitting and both final arms. The remaining **2,400 sentences** supply
  every original training example for scorer/baseline/augmented. Native dev
  selects checkpoints identically. Raw native CoNLL test and raw WNUT/WikiANN
  test evaluate both arms with the existing native/shared span metrics.
- Reuse original frozen bert-base-cased block-12 mean-WordPiece caches, the
  768→256→9 BIO MLP, ten epochs, original AdamW/dropout/clip settings and
  validation token macro-F1 criterion. Train the scorer from scratch, seed 0.
  Final classifiers use paired seeds 0/1/2.
- One proposal per training focal. Uniformly choose one existing context word
  at word distance 1–5 in either direction. Mask that word's pieces, leaving
  every focal piece and every other context word unchanged. Use the existing
  single MLM pass, drawing each masked piece from its conditional distribution
  with special tokens excluded and preserving piece counts. Multi-piece word
  sampling is factorized; this approximation is part of the unknown generator
  discrepancy. There are no retries or selector models.
- Accept iff the decoded context word actually changes (case-sensitive), the
  scorer's native BIO argmax differs from original gold, and confidence ≥.9.
  Store accepted focal features/pseudo-labels once for every seed/epoch. Other
  augmented tokens have no loss. Preserve all original gold labels. Report
  BIO-only versus entity-type/O changes separately, without changing acceptance.
- The held-out calibration corpus defines a fixed empirical uniform-word
  reference. Sample 4,000 focal slots **with replacement** for original scorer
  error and another independent 4,000 for proposal acceptance, using independent
  seeds. Repeats are intentional IID draws conditional on that empirical corpus.
  They do not provide a population generalization guarantee. Neither stream
  selects thresholds, changes generation, or supplies final training examples.
- e counts the **joint** event error AND confidence ≥.9, divided by all 4,000
  original observations. Also report ordinary accuracy/error, confidence
  coverage and conditional error among confident predictions. a is accepted
  proposals / all 4,000 candidate trials, including restored/no-neighbour trials.
- One-sided exact Clopper–Pearson limits use beta quantiles: e_U at tail .025,
  a_L at tail .025, giving at least .95 joint coverage by Bonferroni. Handle
  zero/all successes with the usual endpoints. Report max(0,1−e_U/a_L) only as
  the ideal-sampler conditional bound, and δ_max=.05 a_L−e_U. If a_L=0, the bound
  is vacuous (zero). δ_Q remains unknown; text-only TV is insufficient: the
  discrepancy must control the joint text/true-modified-label distribution and
  any reference-to-generator shift. No unconditional certification is claimed.
  Method reference: https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html
- Comparability: identical initial parameters, original batches/order, original
  token presentations, optimizer steps, dev criterion and raw evaluations.
  Add accepted focal examples to augmented batches in proportion N_aug/N_original;
  cycle/shuffle them across epochs. Baseline padding labels are ignored. Loss
  for each head is the mean over its actual labelled tokens. Thus augmentation
  has extra labelled tokens, explicitly counted in budgets.csv, rather than
  silently dropping original labels or giving augmentation a 50% weight.
- Save per-seed span F1/precision/recall/accuracy, paired absolute differences,
  complete proposal/transition tables and accepted feature arrays. A descriptive
  mechanism CSV measures prediction flips and scorer-label agreement on the
  accepted held-out calibration candidates; these are not gold-labelled
  modified examples and agreement is not accuracy.

Estimated runtime about **5–10 minutes** from previous cached generation and
readout runs; allow **10–15 minutes** for model/data loading and reporting.
Actual runtime is saved. This pilot tests yield and downstream utility, not
whether the unknown MLM discrepancy satisfies the conditional precision bound.
