# Is the layer gap a coverage effect?

Run: `scripts/run_coverage.sh` (`python -B -m lion.coverage.run`). Config:
`conf/coverage.yaml`; stages prepare, estimate, train, analyse, report are
Forge/Hydra runs. Raw results: `results/coverage/`; report: `report/RESULTS_14.md`.
Heads only: no scorer, selector, fills or certificates.

## Data

- **CoNLL-2003 NER**: the existing bank already holds all 14,041 native training
  sentences, native dev (3,250) and test (3,453).
- **EWT dependency relations** (base relations, colon subtypes stripped, as before):
  the previous bank was capped at 10,531 training sentences. `prepare` rebuilds it
  with the same pinned revision and code, without the cap
  (`results/coverage/prepare/en_ewt_full`). Native dev and test are asserted
  identical, row for row, to those of the layer run.

Word forms are the case-sensitive strings recovered from each word's WordPiece
IDs (`##` removed), computed the same way for training and test words.

## Part A: frequency breakdown of the layer-run heads

Heads of `results/layer_sensitivity` (layers 4, 8, 12, the layer run's best
single layer by dev and the mixture; seeds 0–2), trained on T1∪T2. Their saved
test logits are reused; nothing is trained. Frequency = occurrences of the test
word's exact form among T1∪T2 words; bins 0, 1–2, 3–9, 10–99, 100+. CoNLL rows
are split by gold O / gold entity; EWT uses all words.

Per bin: token accuracy (native tags) per head, and the paired gaps best − layer 12
and mixture − layer 12, computed per row (correct − correct) and then averaged.
Seeds give mean ± SD; row SE uses seed-averaged row values. The cross-tab CSV adds
row counts and each head's error on words whose form was seen in training only
with other labels.

## Part B: learning curves

Nested subsets: one seed-0 permutation of the full native training sentences;
fraction f takes its first ⌈f·n⌉ sentences (f = 1/16, 1/8, 1/4, 1/2, 1). For each
fraction, heads for layers 0–12 and the mixture, seeds 0–2, with the existing
recipe (`layer_sensitivity.train.Head`; MLP 768→256→labels, dropout .1, AdamW lr .001,
weight decay .01, clip 1 on the MLP, 64-sentence batches from the seeded sentence
schedule, raw-dev token macro-F1 checkpoint).

Step matching: the full-data run takes 10 · ⌈n/64⌉ steps. A subset with m
sentences gets an epoch cap of ⌈10 · ⌈n/64⌉ / ⌈m/64⌉⌉ epochs, so every run has at
least the full-data step count. Dev is evaluated after every epoch and the best
checkpoint is kept; the selected step is reported.

Per fraction: test span F1 and token accuracy (CoNLL), accuracy and macro-F1
(EWT) per head; the best single layer by mean dev macro-F1; gaps best − layer 12
and mixture − layer 12 per metric, paired by seed (mean ± seed SD); learned mixture
weights. Part A's frequency breakdown is repeated per fraction, with frequencies
counted in that fraction's subset (CSV only).

All-layer training vectors are computed in memory and never saved; per run only
the selected checkpoint's test argmax, metrics and (for the mixture) weights are saved.

## Monotonicity lines

"Decreases monotonically" = the gap means are non-increasing along the frequency
bins (0 → 100+; empty bins skipped) or along the fractions (smallest → full).

## Runtime rule

The estimate stage times optimizer steps and dev evaluations at real tensor sizes
for a single-layer head and the mixture, multiplies by each run's step and epoch
counts, and adds feature extraction and analysis. If the total exceeds six hours,
the 1/16 fraction is dropped.
