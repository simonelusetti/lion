# EWT dependency-relation sensitivity replication

Run: `USE_TF=0 .venv/bin/python -B -m lion.deprel_replication.run`.
Config: `conf/deprel_replication.yaml`; every stage uses Forge/Hydra.
Raw results: `results/deprel_replication/`; final report: `report/RESULTS_12.md`.

## Task and data

Use the cached English EWT UD train/dev/test parquet files from
`universal-dependencies/universal_dependencies`, pinned revision
`c3fed78df6b9f8cd27ed24cc7142e9eccf1e6703`. Store exact file names, SHA-256
hashes and selected sentence/document IDs. Predict the **base dependency
relation of each word**, stripping colon subtypes as the existing certification
pipeline does. Include punctuation and root. Inputs contain text only: no gold
head positions. This is relation tagging, not dependency parsing; evaluation is
token accuracy and macro-F1, not LAS or span F1.

Seed 0 selects 10,531 source training sentences. Source roles are disjoint by
EWT document: 1,920 scorer fit sentences, 683 scorer validation, 4,414 T1,
3,514 T2. All three baseline seeds train on the same **7,928 T1∪T2 sentences**,
excluding scorer split A. Native dev: 2,001 sentences / 25,148 words; native test:
2,077 sentences / 25,094 words. Use native test once rather than duplicate rows
to match the NER word count. Scorer grouping fractions apply to documents,
which explains the changed sentence counts.

## Reused pipeline

Reuse frozen bert-base-cased block 12 and mean WordPiece representations;
MLP 768→256→labels; selector hidden layers 128/64. Scorer and baselines train
ten epochs; cross-fitted selectors train five epochs on all eligible words.
The matched baseline sentence schedule, optimizer, dropout and raw-dev token
macro-F1 checkpoint selection are reused. Baseline seeds are 0/1/2; no augmented
arms are trained.

Calibration uses one **all-token group**, since dependencies have no O/entity
partition. Reuse opposite-fold calibration (70% of sentences) and held-out
certificate (remaining 30%), one sampled focal per sentence, ten fills, retained
fraction grid 1,.95,…,0, Learn then Test ε=.05/δ=.1. Calibration γ is the new
scorer's error rate on its own validation data. Run the same held-out gate and
report absolute CE mean/SD/n/UCB next to CE increase. The gate checks whether
the increase lower bound exceeds ε; it does not assert that its upper bound
falls below ε. Failed groups still undergo diagnostics, marked UNCERTIFIED.

Generate four independent fills per row, mask and selector fold, recording
filled token IDs immediately. Selector/random masks have equal drop counts.
Reuse the one-pass MLM fill procedure and piece-count preservation. The same
cached fills serve all classifiers and all diagnostics. Measurement γ stays
.01; S is KL(original || filled), flips change argmax. Average draws within
fold, then folds within word. No selector or scorer is refitted after calibration.

## Diagnostics and reporting

Reuse restoration/null-swap rates; non-null sensitivity; changed-count and
nearest-change bins; dropped-position distances; size/distance cell-matched
ratios; ambiguity; flip concentration and top twenty forms. Replace entity-span
familiarity with case-sensitive **word/label familiarity** in T1∪T2: unseen
word, seen word with another label, seen word with the same label. Ambiguity
counts distinct base dependency labels per word. Add descriptive per-gold-label
sensitivity and original error to CSVs so easy relations cannot hide harder ones.

Selector/random tables use common word support and the same bins and weights.
Report mean ± sample SD across seeds and word-row SEs; do not count folds or
draws as independent rows. Flag bins below fifty rows and omit such cells from
ratios. Keep all detailed results in CSVs and the report focused on core tables.

Pre-run estimate from measured CoNLL timing scaled by token/WordPiece work:
Part 1 **2.08 h**, Part 2 **6.7 min**, total **2.19 h** (roughly ±50% at similar
GPU throughput). This falls below the inherited six-hour rule, so retain all
native test sentences. Actual stage and wall-clock times are saved automatically.
