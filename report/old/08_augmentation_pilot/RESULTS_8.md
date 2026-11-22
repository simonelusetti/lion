# Experiment 8: original-plus-fill augmentation

All three arms train on the same **2,250 original CoNLL-2003 sentences (32,721 labelled words): T1 ∪ T2**. Both scorer partitions are excluded. Exact sentence IDs are in prepare_training_sentences.csv.

The word-rank artifacts were absent on this machine. Following authorization, the scorer, cross-fitted ranking selectors and cuts were rebuilt once from the current CoNLL-2003 pilot preset (3,000 source training sentences, seed 0), then frozen. This is the matched population of the rebuilt word-rank pilot; the historical HardKuma results in Experiment 7 are not a numerical reference for this run.

| Partition | Sentences | Words |
|---|---:|---:|
| scorer_train | 600 | 8236 |
| scorer_validation | 150 | 2412 |
| g1 | 1125 | 16682 |
| g2 | 1125 | 16039 |

Each augmented arm contains all originals and exactly one cached, focal-labelled variant per training word. Selector masks use the opposite fold and scorer-predicted raw group. Random masks drop the same number of context words at uniformly random positions. Every dropped piece is drawn once from the single all-dropped-masked MLM pass; no draw is selected or rejected by its label score. Focal words and piece counts remain intact.

The three MLPs share initialization within each seed and use the same ordered original-sentence stream. The matched baseline’s 64-sentence batches define a common labelled-token budget per optimizer step. Baseline spends it on originals; augmented arms spend half on originals and half on variants (one-token rounding when odd). Thus all arms have equal steps and labelled-token counts; augmented originals progress through the shared stream at half speed. Frozen whole-sentence features allow an original sentence’s word rows to span adjacent optimizer batches. Ten baseline-budget epochs equal five passes over each original-plus-variant pool. AdamW lr .001, weight decay .01, dropout .1 and per-head clip 1 match the source baseline. Each arm selects its checkpoint on raw CoNLL dev macro-F1: the unweighted mean of token-level F1 across the nine BIO tags, including O (zero-denominator tags contribute zero). This is not span F1. The existing matched-baseline code uses this same token macro-F1 criterion, as do all three current arms; checkpoint selection is unchanged.

## Raw test results

Mean ± sample SD over three seeds; percentages. Span counting and native/shared label mapping call the existing certification evaluation code. CoNLL’s shared result is retained alongside its native result.

| Dataset | Label space | Arm | Span F1 | Precision | Recall |
|---|---|---|---:|---:|---:|
| conll2003 | native | baseline | 80.37 ± 0.39 | 78.00 ± 0.10 | 82.90 ± 0.73 |
| conll2003 | native | selector_augmented | 79.04 ± 0.40 | 76.45 ± 0.27 | 81.80 ± 0.55 |
| conll2003 | native | random_augmented | 79.31 ± 0.32 | 77.20 ± 0.72 | 81.55 ± 0.45 |
| conll2003 | shared_ner | baseline | 82.60 ± 0.59 | 80.30 ± 0.65 | 85.03 ± 0.58 |
| conll2003 | shared_ner | selector_augmented | 81.19 ± 0.37 | 78.84 ± 0.35 | 83.68 ± 0.74 |
| conll2003 | shared_ner | random_augmented | 81.10 ± 0.09 | 78.72 ± 0.49 | 83.62 ± 0.64 |
| wnut | shared_ner | baseline | 41.56 ± 1.06 | 38.60 ± 1.43 | 45.02 ± 0.70 |
| wnut | shared_ner | selector_augmented | 41.93 ± 0.56 | 39.83 ± 1.50 | 44.40 ± 2.43 |
| wnut | shared_ner | random_augmented | 40.79 ± 0.35 | 38.62 ± 1.88 | 43.33 ± 1.82 |
| wikiann | shared_ner | baseline | 37.00 ± 0.45 | 32.30 ± 0.57 | 43.32 ± 0.22 |
| wikiann | shared_ner | selector_augmented | 36.18 ± 1.37 | 31.54 ± 1.11 | 42.42 ± 1.75 |
| wikiann | shared_ner | random_augmented | 36.61 ± 1.08 | 32.01 ± 0.77 | 42.77 ± 1.59 |

## Seed-paired differences

Percentage points, mean ± sample SD of paired differences. Per-arm/per-seed values and each seed’s differences are in the accompanying CSVs.

| Dataset | Label space | Selector minus | Span F1 | Precision | Recall |
|---|---|---|---:|---:|---:|
| conll2003 | native | selector_minus_baseline | -1.33 ± 0.07 | -1.54 ± 0.23 | -1.10 ± 0.19 |
| conll2003 | native | selector_minus_random_augmented | -0.27 ± 0.22 | -0.74 ± 0.62 | 0.25 ± 0.42 |
| conll2003 | shared_ner | selector_minus_baseline | -1.41 ± 0.22 | -1.46 ± 0.56 | -1.35 ± 0.17 |
| conll2003 | shared_ner | selector_minus_random_augmented | 0.09 ± 0.39 | 0.12 ± 0.58 | 0.06 ± 0.18 |
| wikiann | shared_ner | selector_minus_baseline | -0.82 ± 1.61 | -0.76 ± 1.45 | -0.90 ± 1.82 |
| wikiann | shared_ner | selector_minus_random_augmented | -0.44 ± 0.29 | -0.47 ± 0.35 | -0.35 ± 0.21 |
| wnut | shared_ner | selector_minus_baseline | 0.38 ± 1.38 | 1.23 ± 0.18 | -0.62 ± 3.12 |
| wnut | shared_ner | selector_minus_random_augmented | 1.15 ± 0.53 | 1.20 ± 0.42 | 1.07 ± 0.63 |

## Variant diagnostics

Scorer disagreement is a biased proxy for label flips, measured after generation without changing the draw. Unchanged/zero-drop draws still count as variants.

| Arm | Raw group | Variants | Variant token share | Scorer disagreement | Input changed |
|---|---|---:|---:|---:|---:|
| selector_augmented | all | 32721 | 50.0% | 5.20% | 98.11% |
| selector_augmented | O | 27594 | 50.0% | 1.78% | 99.88% |
| selector_augmented | entity | 5127 | 50.0% | 23.62% | 88.57% |
| random_augmented | all | 32721 | 50.0% | 5.17% | 98.13% |
| random_augmented | O | 27594 | 50.0% | 1.80% | 99.87% |
| random_augmented | entity | 5127 | 50.0% | 23.33% | 88.80% |

## Frozen scorer validation

On its own scorer-validation split (150 sentences, 2412 words), the frozen scorer has **71.07% native span F1** and **0.1300 nats per word** of raw, unsmoothed cross-entropy, with full-sentence inputs. The mean sum of word NLLs per sentence is 2.0899 nats. This is the scorer’s A-validation partition, not the native CoNLL dev set used to select the three downstream classifiers.

## Recalibrated kept fractions

| Selector | Applied to | Raw group | rho | Calibration rows |
|---|---|---|---:|---:|
| g1 | g2 | O | 0.00 | 776 |
| g1 | g2 | entity | 0.85 | 619 |
| g2 | g1 | O | 0.25 | 779 |
| g2 | g1 | entity | 0.75 | 614 |

## Held-out bounds for selector fills

These are the existing certificate calculation on the rebuilt pilot’s held-out test rows and calibrated cuts: one-sided 95% normal upper bounds on **smoothed fitted-scorer excess cross-entropy, in nats**. They use the same calibration-style independent fills. They are not guaranteed label-error probabilities. Interpreting them as information-loss bounds requires accurate reference posteriors and the relevant fill conditional; selection-conditioned and calibration-style fills are different. No probability conversion is asserted. All arms run regardless of these diagnostics.

| Selector | Group | Kept fraction | n | Absolute CE mean | Absolute CE SD | Absolute CE UCB | CE increase mean | Increase SD | Increase UCB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| g1 | O | 0.00 | 79 | 0.1493 | 0.7530 | 0.2886 | 0.0414 | 0.2687 | 0.0912 |
| g1 | entity | 0.85 | 65 | 0.8218 | 1.0428 | 1.0346 | 0.0332 | 0.1164 | 0.0570 |
| g2 | O | 0.25 | 78 | 0.0309 | 0.0738 | 0.0446 | 0.0024 | 0.0315 | 0.0082 |
| g2 | entity | 0.75 | 68 | 0.8467 | 1.0359 | 1.0534 | 0.0478 | 0.2899 | 0.1056 |

Absolute CE is −log[(1−gamma) × mean_fill p_scorer(gold) + gamma/C], with gamma=.01. Its UCB is mean + 1.645 × sample SD / sqrt(n). It is reconstructed on the exact existing test rows and fills as the saved CE increase plus raw gamma-smoothed CE; no new fill-ins are sampled. The increase and absolute CE have separate standard deviations and bounds.

## Timing and artifacts

Variant generation: **1.6 minutes**. Three-seed fitting of all three arms: **0.2 minutes**, including dev checkpoint evaluations. Training plus raw test evaluation: 0.2 minutes. End-to-end including rebuilt prerequisites: **0.13 hours**.

Raw run: `/home/simonelusetti/lion/results/augmentation/39d500bd075b`. Frozen source run: `/home/simonelusetti/lion/results/augmentation/source/b5ca455dbea8`. Cached focal features are reused for all seeds/epochs; schedules and predictions are saved per seed. [Implementation](IMPLEMENTATION_8.md), [tables](tables/augmentation/).
