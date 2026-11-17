# Context-partition diagnostics: implementation

The [follow-up request](original_request_2.md) tests linear locality (L), syntactic locality (S), lexical channels (K), shared token channels (N), and gold-label context (A). This follow-up is separate from E1–E7: old checkpoints, samples and results are preserved. Configuration: [partition/default.yaml](../../../conf/partition/default.yaml). Code: [lion/partition](../../../lion/partition/).

## Data and fitting

The exact previous focal manifests are reused: sample seed 1729 and classifier seeds 0/1/2. Datasets are EWT UPOS, EWT dependency relations, GUM discourse, shared NER, and the added GUM UPOS task. GUM UPOS uses the same GUM sentences and focal positions with UPOS targets derived from the already downloaded gold UD files.

| Dataset | Train / validation / test focal words | Conditional-probe labels |
|---|---:|---:|
| EWT UPOS | 4,017 / 998 / 1,971 | 10 |
| EWT dependency | 4,017 / 998 / 1,971 | 7 |
| GUM discourse | 16,309 / 3,138 / 2,916 | 14 |
| GUM UPOS | 16,309 / 3,138 / 2,916 | 11 |
| Shared NER | 7,762 / 1,951 / 3,922 | 4 |

The encoder is frozen `bert-base-cased`, block 12, first-WordPiece pooling, no CLS/SEP, maximum 512 WordPieces. Masking replaces all pieces of a word with `[MASK]`, keeps attention on, and preserves length. Y readouts use the original 768→256→labels MLP, ReLU, dropout 0.1, AdamW at 0.001, weight decay 0.01, norm clipping at 1, 64 sentences/step, 10 epochs, and validation macro-F1 checkpoint selection. Runtime settings control device, parallelism, extraction batches and chunks.

Independent readouts are batched over conditions in [tagger/ensemble.py](../../../tagger/ensemble.py). Parameters, losses, gradient clipping and validation-selected checkpoints remain separate for each condition. Initialization and dropout random numbers are shared across condition heads within a seed. This changes computational scheduling, not the classifier architecture or marginal dropout distribution.

EWT genres, GUM genres and the shared NER mapping are unchanged from the first study. Gold heads and UPOS are aligned by retokenizing the raw UD words and comparing token IDs. Tree distances are shortest paths in the undirected gold dependency tree, measured on original word nodes before gathering retained words. No offline parser is installed: shared NER has linear distance only, no gold UPOS, and sentence blocks in place of unavailable document blocks. Those shared-NER intervals are exploratory.

## Conditions and probes

- Linear radii: 0, 1, 2, 3, 5, 8 and full. Tree radii: 1, 2, 3 and full where available.
- Each finite nonzero window has a random control retaining exactly that focal's number of context words, sampled without replacement. Word-only and full are shared canonical endpoints across families.
- G masks every word, including the focal word. Every other condition retains the focal word.
- There are 19 canonical conditions for UD tasks and 13 for shared NER. Random masks are fixed across classifier seeds. Identical EWT and GUM feature cohorts are reused across their two tasks.
- Z probes are linear softmax classifiers with inverse-frequency training weights and validation balanced-accuracy selection, using the same optimizer and epoch budget.
- Z analyses use five document-grouped, genre-stratified folds over the pooled native sample. Each probe trains on three folds, validates on the next fold, and tests on the held-out fold. Domains must meet the original 32/8/8 label-conditional support thresholds in every fold. That common eligibility set is fixed across conditions.
- Y readouts retain native training/validation/test splits. Pooling samples for Z cross-validation does not retrain Y readouts on test labels.

## X0: amortization

A fresh dedicated Y readout is fitted for each canonical condition. An additional single readout samples a condition uniformly per training example on every optimizer step. Its checkpoint is selected by average validation macro F1 across all conditions. Test comparisons are accuracy loss, macro-F1 loss and mean KL(dedicated || amortized), per condition.

The operational acceptance tolerances, fixed before fitting, require upper bootstrap bounds of at most 1 percentage point for both losses and 0.02 nats for KL. These are practical tolerances, not a familywise equivalence test. Dedicated readouts are retained in X1 regardless of X0's outcome.

## X1: frontier and confirmatory comparisons

Each condition records native-test Y accuracy, macro F1 and cross-entropy. Conditional Z probes read (a) the focal representation, (b) the dedicated Y readout's log-probabilities, or (c) only WordPiece sentence length, first-WordPiece focal position and retained context-word count. Only the scalar geometry inputs are standardized from training-fold statistics. Nuisance is balanced-accuracy excess over that input type's G probe, paired by seed; chance is not used as the subtraction baseline.

The frontier x-axis is Y accuracy; its y-axis is predictor-nuisance excess. Random points are labelled by mean retained context count, although counts match each focal exactly.

The three prespecified primary comparisons are:

1. **Matched label information:** interpolate to word-only accuracy plus 90% of the positive full-minus-word-only gain. Use the first adjacent crossing in radius order; random conditions are ordered by mean retained count. Compare linear–random, tree–random, and tree–linear nuisance using paired document bootstrap intervals. Nonpositive full gain makes this comparison undefined, rather than reversing the target.
2. **Knee:** a finite window must recover at least 90% of that label gain and have at most 50% of the positive full-context predictor-nuisance excess. Nonpositive reference nuisance makes a knee undefined. Report seed agreement and bootstrap stability, rather than treating missing evidence as a knee.
3. **Shift:** repeat native single-source training for linear/tree windows, including word-only and full. Use the original minimum source/target supports. Plot window-minus-word-only accuracy in-domain and out-of-domain, averaging ordered source–target pairs equally. Peak-minus-full contrasts use a simultaneous band from the maximum bootstrap deviation over the fixed finite radii within each family. This addresses selecting a peak within that family, not multiplicity across the entire study.

Only these comparisons are confirmatory. Favoring a window is evidence consistent with L/S; a tree advantage over linear is consistent with S. Overlapping intervals do not establish N or equality of families.

## X2: word and distance-band attribution

For each physical feature cohort, three fixed independent Bernoulli(0.15) context-occlusion draws are cached per focal; the focal is retained. New Y readouts and conditional representation Z probes sample uniformly among those draws while training. This approximates fresh sparse occlusion without re-encoding BERT every epoch.

Up to 500 original native-test focals per dataset are selected with a fixed seed. Each other word is removed individually; scores are the full-minus-perturbed true-Y log probability r and true-Z log probability n. The nuisance score uses the held-document-out, within-label **representation** probe. Unsupported label/domain cases retain r but have undefined n, with coverage recorded. Scores are signed; negative drops are not clipped.

Whole linear bands 1, 2, 3–4, 5–8, 9+ and tree bands 1, 2, 3+ are also removed. Each has a uniformly sampled random-removal control of identical word count. This tests redundancy that single-word removal can miss.

Outputs include Pearson correlation over valid word pairs, top-20% Jaccard per focal, and Jaccard excess over the exact random-set expectation at that focal's set size. Distance, word class, training frequency and exact-form-match strata are recorded. Frequency uses full original training word counts; shared NER pools counts from both original corpora.

OLS models for r and n jointly include log(1+linear distance), log(1+tree distance), log(1+training frequency), exact-form match, and function/other-word indicators with content words as reference. Continuous predictors are standardized; regressions use all valid word pairs. Content UPOS is ADJ/ADV/NOUN/PROPN/VERB; function UPOS is ADP/AUX/CCONJ/DET/PART/PRON/SCONJ. Shared NER omits tree and word-class terms. Coefficients, block partial R² and word-class-minus-distance partial R² receive document-bootstrap intervals. These are descriptive partial associations, not causal effects.

## X3: gold label context

Linear probes predict Z from position-specific one-hot gold labels at radii 0/1/2 (A), and from those labels plus the old word-only vector (B). Padding and unsupported/unlabelled neighbors have separate categories. The same pooled document folds are used. Report balanced-accuracy gain B−A, cross-entropy gain A−B, and the paired k=2 minus k=0 gain trend. Shrinkage is consistent with A; linear probes cannot rule out nonlinear label interactions, and word-only vectors remain lexical genre proxies.

## X4: metadata-free context clusters

Two masked passes per focal provide outside and inside vectors. Outside keeps only words beyond radius 3; inside keeps only neighbors within radius 3. Both mask the focal. First-WordPiece block-12 vectors of visible words are mean-pooled. Cases with empty inside or outside sets are excluded from both channels.

Torch k-means uses Euclidean distance, k-means++ initialization, at most 100 iterations, and tolerance 1e-4. Each seed and outer fold fits K=domain count and 2K on the other four document folds. No labels enter clustering. Training-fold Z supplies a balanced Hungarian cluster-to-domain assignment, with capacity ceil(clusters/domains) per domain. Evaluation uses held-out documents. Within-label matching uses training examples from that label and eligible domains; the clusters themselves are not refitted with Y.

NMI uses arithmetic entropy normalization and is computed within fold before test-count-weighted aggregation, avoiding pooling incompatible cluster IDs. Balanced accuracy is scored overall and within label. Paired nonempty coverage is recorded.

## Statistics, outputs and execution

All intervals use 200 genre-stratified document-block resamples. Evaluation bootstraps reuse fitted predictions rather than refitting every classifier. Regressions are refitted from bootstrap-weighted document sufficient statistics. The same resampling weights pair conditions/seeds within an analysis. Three-seed SD describes optimization variation; the bootstrap describes held-out document variation conditional on fitted models. Undefined resamples are excluded from percentile intervals, and valid counts are saved.

Models and raw data are under `results/partition/` and a configuration-keyed `data/cache/partition/` folder. Aggregated CSVs and requested figures are copied to `report/old/02_partitions_x0_x4/tables/` and `report/old/02_partitions_x0_x4/figures/`. [RESULTS_2.md](RESULTS_2.md) is populated from completed measurements, with primary assessments and the protocol differences above. No old E1–E7 artifact is overwritten.

The sequential Forge/Hydra entry point is:

```bash
.venv/bin/python -B -m lion.partition.run runtime.workers=0 runtime.batch_size=32
```

Stages are prepare → cache → frontier (X0/X1) → shift → attribution (X2) → labels (X3) → clusters (X4) → report. No selectors, encoder updates, erasure training or invariance objectives are added. The run is launched in tmux and left unmonitored.
