# Cross-fitted word ranking and certification

`python3 -B -m lion.certification.run` runs preparation, scorer training,
selector training, calibration, certification, cleaned-feature caching, and
classifier comparison through Forge and Hydra. Results go under
`results/certification/<config hash>/`. Stage overrides use
`stage.experiment=<stage>` with the same experiment configuration.

## Words and the fixed scorer

Training and evaluation use one BIO tag per original word. The frozen
bert-base-cased encoder contextualizes the WordPieces. The selector, scorer,
and downstream heads receive their mean at each original word slot. The raw
word vectors are cached and reused. Resampling preserves word slot lengths.
The focal word and special tokens stay intact. Consequently the tests and
certificate measure information beyond kept words and the original piece
counts of dropped words.

The scorer trains on its own partition and remains frozen for the selector
and all later stages. Selector g1 trains on T1 and is applied to T2. Selector
g2 trains on T2 and is applied to T1.

## Selector training

The three-layer MLP scores each context word using its pooled vector and the
focal vector. Each example samples a uniform kept fraction and uses the
corresponding rounded word budget. Its hard top-k kept words stay visible
while every piece of every dropped word is masked in the same MLM pass.
Each masked piece draws independently from its MLM distribution. The
training loss interpolates original and sampled piece embeddings with the
word's differentiable pairwise-rank gate, re-encodes the variants, and takes
the negative log of the mean gold-label probability over the draws. There
is no size penalty. The rank and gate temperatures and the training draw
count are Hydra options. The selector training cap limits sampled labeled
word focals independently in each fold.

Select `certification=conll2003_pilot` for the smaller CoNLL-2003 run with
WNUT and WikiANN transfer. The preset samples 3,000 source training
sentences before the scorer and selector split, using the experiment seed.
It caps selector fitting at 1,024 word focals and CRT sampling at 80 test
sentences per fold. The CRT timing pilot targets 30 minutes per fold while
preserving 50 draws per row. Source validation and the ID and transfer test
splits retain their full sizes. Tune the preset with
`certification.train_sentence_cap`, `certification.selector_train_focals_per_fold`,
`certification.test_documents` and `certification.test_budget_seconds`.

## Calibration and held-out tests

The scorer's raw prediction assigns each NER word to `O` or entity. Each
opposite fold is split by sentence into calibration and test sets. Each
sampled sentence contributes at most one random word from each predicted
group present. The default calibration share is 70%. Non-NER tasks without
an outside label use one group.

For each group, calibration compares the scorer's uniform-smoothed log loss
on raw text with its loss after averaging independent-piece fills. The
default grid runs from 1.00 to 0.00 in steps of 0.05. Identical rounded
budgets within a row reuse the same fills and scores. Fixed-sequence testing
starts at 1.00, which has exact zero loss. At each later cut it uses the
one-sided normal-approximation p-value and stops selecting cuts at the
first non-rejection. The cut is the last rejected fraction at level
`delta / number of groups` within each selector. The curve reports sample
size, mean, standard deviation, the requested 1.28 standard-error margin,
and p-value for every grid point. The risk claim uses the CLT approximation
on the calibration fill distribution and independent sentence rows.

On separate test sentences, sufficiency samples every piece of the dropped
words left to right, conditional on the kept pieces and visible piece
counts. Minimality selects one retained context word and resamples all its
pieces while leaving the other words visible. Candidates preserve the
selector's word mask and the row's scorer-predicted group. The test reuses
the scorer output from this replay. The pooled CRT compares sums of gold
log probabilities, with ties counted against rejection. The effect is the
mean original log probability minus the log of the mean resampled
probability. The certificate averages the accepted probabilities, adds
uniform smoothing, and reports CE against the unsmoothed full-context
scorer on the same rows, the gap, and both normal and Hoeffding bounds.

The first 1% of selected test rows times the chained tests and contributes
to their results. The row budget can reduce remaining rows while preserving
the configured draw count. The held-out gate requires sufficiency p above
`alpha` and certificate gap at most `epsilon` in every fold and group.

## Classifiers and artifacts

A passing gate triggers opposite-fold cleaning on all labeled training
words and a comparison against the raw baseline. Cleaning uses the same
one-pass independent sampling as calibration. The cache stores three pooled
feature draws per training word by default, with one sampled by the
classifier's DataLoader each epoch. Raw validation and raw ID/OOD evaluation
use the same mean pooling for both classifiers.

- `selector/<task>/<fold>/` records the checkpoint, training history, timing
  estimate, and labeled sample coverage when the cap is set.
- `calibrate/curve.csv` and `fractions.csv` report per-group calibration
  and selection. Per-fold artifacts retain the fractions and test sentence IDs.
- `certify/crt.csv`, `certificate.csv`, `mask_sizes.csv`, and `gate.csv`
  record per-group held-out results and the downstream decision.
- `clean/<task>/<fold>.pt` stores feature draws with the original word keys
  and BIO labels.
- `classify/performance.csv` compares the matched heads on raw text. Each
  stage records measured wall time.

On Linux, `runtime.selector_cpu_groups=[0-7,8-15]` with
`runtime.threads=16` runs the fold stages on two independent eight-core
groups. MLM outputs and re-encoding are batched within configurable memory
limits.
