# Cross-fitted context selector

`python3 -B -m lion.certification.run` runs `prepare`, `scorer`, `selector`,
`certify`, `clean`, and `classify` through Forge and Hydra. The stages write
under `results/certification/<config hash>/`. Run a single stage with
`stage.experiment=certify` and the same configuration overrides.

The NER run trains on CoNLL-2003 and uses its native validation and in-domain
test sets. The target list can be restricted to WNUT and WikiANN with
`certification.ner_targets=[wnut,wikiann]`. Transfer scores use the pilot's
shared O/PER/ORG/LOC mapping. Both readouts evaluate the same source and target
examples. Training and cleaning use each labeled subword as a focal with the
word's gold tag. Validation and test score the original word-aligned positions.

## Scorer and selector

The frozen bert-base-cased encoder supplies vectors and its MLM head supplies
fill-ins. Split A is document-disjoint from the two selector folds. Its MLP
scorers train on raw full-context examples and are fixed for all later stages.
The selector's three-layer MLP reads the focal and candidate BERT vectors.
Each candidate gets a stretched HardKuma gate. The focal and special tokens
are kept.

The selector precomputes four MLM samples per position by masking that one
position in the real sentence. All epochs and focals in that sentence reuse
those samples. Training interpolates the original and sampled embeddings
using the gates and runs BERT and the fixed scorer on each variant. The loss
is the negative log mean gold-label probability plus `0.05` times the
expected number of nonzero context gates. Inference keeps a candidate when
the point mass at one is at least as large as the point mass at zero.

`certification.selector_train_focals_per_fold` optionally samples a fixed
number of labeled focals within each selector's own training fold. Smaller
caps use prefixes of the same seeded sample. `training_sample.csv` records
per-label coverage. Scorer training, cross-fit assignments, cleaning and
evaluation still use the original splits.

## Tests and certificate

Each selector is tested on the opposite training half. The seeded row sample
chooses one labeled focal per sampled sentence and at most one sentence per
document when document IDs are available. The initial proposal is 50 accepted
draws per row. The first 1% of selected rows are timed and reused. The runner
prints an ETA and reduces the remaining row count against
`certification.test_budget_seconds`, keeping the draw count fixed.

Sufficiency masks the entire dropped block and fills it left to right.
Minimality samples one kept context subword and replaces only that subword
conditional on all the others. Both conditions score the full sentence, replay
the selector, and accept only candidates with identical masks. The pooled test
compares sums of original and resampled log gold-label probabilities, counting
ties against rejection. Each effect is the mean row-wise original log
probability minus the log mean resampled probability. Rows with too few
accepted draws contribute to acceptance and coverage statistics.

The certificate uses complete sufficiency draws. For an empty dropped block,
each draw is the original sentence. The scorer probabilities are mixed with
uniform mass 0.01. The report includes the mean cross-entropy, full-context
cross-entropy on the same rows, their gap, the normal-approximation upper
bound, and the bounded-loss Hoeffding upper bound. The concentration bounds
use independent rows. CRT calibration depends on the MLM conditional
approximation.

## Cleaning and transfer

The selector fitted on T1 cleans T2 and the selector fitted on T2 cleans T1.
Each training focal uses one same-mask accepted joint fill, with its original
focal position and label. Existing first sufficiency fills are reused for the
tested focals. An empty dropped block needs no substitution. Cleaning reports
an error if a nonempty dropped block hits the proposal cap without an accepted
draw. The two fresh classifiers train on the cleaned and original versions of
the same examples. Both use raw source validation and raw in-domain and
out-of-domain test sentences.

Encoder forwards are batched over focal positions, training draws and CRT
conditions within the configured memory limits. Exact duplicate queries share
a forward pass. The frozen encoder computes only the requested positions in
its final feed-forward block. On Linux,
`runtime.selector_cpu_groups=[0-7,8-15]` runs the two folds concurrently during
selector fitting, testing and cleaning. `runtime.threads=16` gives each
eight-core fold eight threads and the other stages up to sixteen.

## Outputs

- `selector/<task>/<fold>/` contains the fitted selector, loss history, and
  training sample coverage when a focal cap is set.
- `certify/crt.csv`, `certificate.csv`, and `mask_sizes.csv` report both tests,
  both confidence bounds, and kept fractions. Per-fold timing files record
  the pilot, ETA and final runtime.
- `clean/<task>/<fold>.pt` records accepted token IDs, focal positions, labels,
  and re-encoded features. Fold timing files record fill reuse and proposal counts.
- `classify/performance.csv` reports in-domain and transfer metrics for the
  two readouts. `classify/runtime.json` records measured wall time.
