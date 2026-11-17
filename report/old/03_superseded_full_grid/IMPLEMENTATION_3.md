# Fixed invariance signatures: implementation

Protocol: [original_request_3.md](original_request_3.md). Code: [lion/invariance](../lion/invariance/). Configuration: [invariance/default.yaml](../conf/invariance/default.yaml). Old experiments are preserved.

## Data and rotations

The study uses every labelled word after the existing baseline tokenization/truncation, not the old focal sample. It reuses cached full-view, first-WordPiece, block-12 vectors of frozen `bert-base-cased`; no new masking or encoder update occurs. The EWT cohort contains 254,820 words across native splits, and GUM contains 256,739. Metadata and labels align with the original sentence caches. GUM UPOS is derived from the same gold UD files.

Tasks are EWT UPOS, EWT dependency relations, GUM UPOS, and secondary GUM discourse. EWT uses all five test genres in alphabetical cyclic order: answers, email, newsgroup, reviews, weblog. GUM uses five prespecified test genres: academic, conversation, fiction, news, whow. Validation is the next genre in the full alphabetical 15-genre order, not the next selected test genre. Other genres train on native train sentences only. Validation/test genres use all their native splits. In-domain testing uses native test sentences of training genres.

## Lookup groups and controls

Every rotation recomputes exact, case-sensitive training-genre word frequencies and empirical label entropy. Only labelled native-training tokens from training genres enter these statistics. Thresholds are 0.2 and 0.5 nats; frequency ≥5 defines frequent types. Unambiguous types have entropy at most the threshold; ambiguous types exceed it; other and unseen forms are rare/unseen. No test label is used to assign a word group.

R1 uses the conditional MMD penalty for all tokens; R2 uses marginal MMD for all tokens. R3 assigns unambiguous types to conditional mode and the rest to marginal mode. R4 reverses this assignment. R5 has five fixed random word-type assignments per threshold, independent of classifier seed, with exactly the same training-token conditional/marginal shares as R3. Unseen types receive a fixed Bernoulli assignment at the target share. To match training shares exactly, the algorithm randomly orders nonsingleton types and reserves singleton types to close the integer token count. It is a constrained random allocation, not a uniform draw over all exactly matched subsets. Exact matching was checked for every configured rotation and threshold.

These assignments affect training losses, not the model's inputs. There is no learned gate. R3/R4/R5 each remain one MLP across all token groups. R1/R2/R0 do not depend on the entropy threshold and are shared between sensitivity analyses.

## Training and penalties

The MLP and optimizer match the previous setup: 768→256→labels, ReLU, dropout 0.1; AdamW with learning rate 0.001 and weight decay 0.01; norm clipping at 1; ten epochs. Classifier seeds are 0/1/2. BERT vectors are cached in float16 and read as float32; runtime configuration controls CUDA, bf16 autocast and GPU parallelism.

Each training batch has the same number of sentences from every training genre. The nominal 64-sentence batch is rounded down to a multiple of the number of training genres: 63 for EWT and 52 for GUM. Smaller genres are reshuffled and oversampled so that every training sentence is visited in each epoch. Equal sentence shares do not imply equal token shares. R0 receives the identical genre-balanced training schedule.

MMD is the differentiable, biased empirical RBF MMD², including diagonal terms. The bandwidth squared is the median positive squared distance between probability vectors in the current batch, detached from gradients. It is estimated separately for each independently trained head. Degenerate equal-probability batches use a constant kernel. All batch tokens are used; MMD is not subsampled.

- Marginal mode sums MMD² over training-genre pairs on its assigned tokens, weighted by their share of the batch. Empty genre cells are omitted.
- Conditional mode repeats the genre-pair sum within each gold label, omitting genre cells with fewer than eight assigned tokens. Each label is weighted by its token share within that mode, then by the mode's batch share; equivalently its assigned token count divided by all batch tokens.
- Pair sums are not normalized by the number of genres. Lambda is therefore selected separately within each task and rotation.

Lambda values are 0.1, 0.3, 1, 3 and 10. Lambda zero is one shared unregularized R0 candidate for every rule/control. Every head selects its epoch on validation-genre macro F1; each rule/draw then selects lambda on that same criterion, preferring smaller lambda on ties. This may select R0 for several rules. No positive penalty is forced.

There are 81 independently parameterized readouts per rotation/seed: R0, ten fixed-mode lambda candidates, and 70 split-rule candidates across two thresholds and five random draws. Heads are trained in chunks of eight with separate losses, per-head clipping and per-head checkpoint selection. Initialization, minibatches and dropout randomness are paired across heads. This is 4,860 trained classifier candidates across twenty rotations and three seeds. Training histories retain penalty components, valid genre-pair counts and conditional supported token mass to expose coverage differences.

An execution-only optimization computes label index groups once per batch on CPU and accumulates diagnostics on the training device until the epoch ends. A representative-batch comparison gave identical loss and gradients and 1.8–2.5× faster steps. Completed fits are retained on resume; an interrupted incomplete fit restarts from its original seed. No objective, batching, model or selection rule changes.

## Routing and measurements

The secondary route uses selected R1/R2 models. For each lookup group, validation-genre token accuracy chooses R1 or R2; ties prefer R1. Empty validation groups fall back to overall validation accuracy. Test routing looks up word types only. Controls are their probability average and five random type routes matching the route's training-token R1/R2 shares exactly.

Selected-rule OOD accuracy and fixed-label macro F1 are measured overall and by group. In-domain accuracy is also retained. R5 and random-route means average draw-level metrics; they are not probability ensembles. Complete lambda curves include test accuracy/F1, in-domain accuracy, validation F1, and selection decisions.

Training-only label shift is operationalized as equal-genre generalized Jensen–Shannon divergence of Bernoulli(label versus other) distributions. Group shift is the training-token-weighted mean over its labels. Label-shift tertile cuts come from the training-token-weighted distribution of label scores; ties are retained, so some strata can be empty. Test labels identify strata only for analysis. A label absent from all training genres has zero observed training shift; this cannot measure its unseen-domain shift.

## Primary comparisons and uncertainty

1. R1 minus R2 OOD accuracy, overall. This alone cannot identify either causal explanation.
2. The unambiguous group's R1−R2 advantage minus that of ambiguous/rare tokens combined, overall and within training-defined shift tertiles. The two component contrasts are also recorded. Positive within-tertile interactions are H_annot-consistent. Attenuation is compatible with H_shift, but a confidence interval containing zero is not evidence of equivalence.
3. R3 minus R4, minus the mean R5 draw, and minus the better fixed mode. The last comparator is the per-rotation/seed oracle maximum of R1/R2 test accuracy, used only in analysis. R3 must beat all three. Its rank among itself and the five draw-level mean performances is reported, with ties distinguished from wins.

Comparisons pair rules within rotation and seed. Rotations aggregate equally. Intervals use 200 independent document-block resamples within each held-out test genre, paired across rules and classifier seeds. The bootstrap averages the three seed results but does not resample seed identities or refit models/validation choices. Seed SD is computed after averaging rotations. Unsupported group/tertile rotations are excluded consistently from a contrast; contributing rotation and valid-bootstrap counts are reported. No large family of per-cell hypothesis tests is run.

## Execution and deliverables

```bash
.venv/bin/python -B -m lion.invariance.run runtime.workers=0 runtime.batch_size=64
```

Forge/Hydra stages: prepare → train → analyze → report. Models, complete histories and predictions stay under `results/invariance/`; compact tables and the four requested figure families are written under `report/tables/invariance/` and `report/figures/invariance/`. [RESULTS_3.md](RESULTS_3.md) is generated from completed measurements and explicitly assesses H_annot, H_shift or neither. Deviations and operational choices are listed above; the five-rotation GUM budget and inclusion of GUM discourse are fixed before fitting.
