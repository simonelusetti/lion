# Experiment 8: original-plus-fill augmentation

All three arms train on the same **10,531 original CoNLL-2003 sentences (152,068 labelled words): T1 ∪ T2**. Both scorer partitions are excluded. Exact sentence IDs are in prepare_training_sentences.csv.

The scorer, cross-fitted word-ranking selectors and cuts were rebuilt once using full CoNLL-2003 (seed 0), then frozen. Source sentence cap: None; selector-training focal cap: None; certificate sentence cap: None. None means uncapped. The matched classifier population is T1 union T2.

| Partition | Sentences | Words |
|---|---:|---:|
| scorer_train | 2808 | 41073 |
| scorer_validation | 702 | 10480 |
| g1 | 5265 | 75647 |
| g2 | 5266 | 76421 |

Each augmented arm contains all originals and exactly one cached, focal-labelled variant per training word. Selector masks use the opposite fold and scorer-predicted raw group. Random masks drop the same number of context words at uniformly random positions. Every dropped piece is drawn once from the single all-dropped-masked MLM pass; no draw is selected or rejected by its label score. Focal words and piece counts remain intact.

The three MLPs share initialization within each seed and use the same ordered original-sentence stream. The matched baseline’s 64-sentence batches define a common labelled-token budget per optimizer step. Baseline spends it on originals; augmented arms spend half on originals and half on variants (one-token rounding when odd). Thus all arms have equal steps and labelled-token counts; augmented originals progress through the shared stream at half speed. Frozen whole-sentence features allow an original sentence’s word rows to span adjacent optimizer batches. Ten baseline-budget epochs equal five passes over each original-plus-variant pool. AdamW lr .001, weight decay .01, dropout .1 and per-head clip 1 match the source baseline. Each arm selects its checkpoint on raw CoNLL dev macro-F1: the unweighted mean of token-level F1 across the nine BIO tags, including O (zero-denominator tags contribute zero). This is not span F1. The existing matched-baseline code uses this same token macro-F1 criterion, as do all three current arms; checkpoint selection is unchanged.

## Raw test results

Mean ± sample SD over three seeds; percentages. Span counting and native/shared label mapping call the existing certification evaluation code. CoNLL’s shared result is retained alongside its native result.

| Dataset | Label space | Arm | Span F1 | Precision | Recall |
|---|---|---|---:|---:|---:|
| conll2003 | native | baseline | 84.27 ± 0.29 | 82.19 ± 0.20 | 86.45 ± 0.68 |
| conll2003 | native | selector_augmented | 83.79 ± 0.37 | 81.73 ± 0.45 | 85.98 ± 1.01 |
| conll2003 | native | random_augmented | 83.62 ± 0.12 | 81.53 ± 0.68 | 85.82 ± 0.65 |
| conll2003 | shared_ner | baseline | 86.05 ± 0.25 | 84.23 ± 0.32 | 87.96 ± 0.58 |
| conll2003 | shared_ner | selector_augmented | 85.84 ± 0.53 | 84.33 ± 0.80 | 87.41 ± 1.32 |
| conll2003 | shared_ner | random_augmented | 85.86 ± 0.27 | 84.49 ± 1.05 | 87.29 ± 0.91 |
| wnut | shared_ner | baseline | 40.65 ± 0.47 | 37.35 ± 1.54 | 44.69 ± 1.99 |
| wnut | shared_ner | selector_augmented | 41.92 ± 2.16 | 40.75 ± 5.55 | 43.58 ± 1.40 |
| wnut | shared_ner | random_augmented | 41.61 ± 2.43 | 41.28 ± 5.83 | 42.30 ± 1.05 |
| wikiann | shared_ner | baseline | 37.70 ± 0.40 | 33.50 ± 0.60 | 43.12 ± 0.10 |
| wikiann | shared_ner | selector_augmented | 36.42 ± 0.45 | 32.12 ± 0.73 | 42.08 ± 0.97 |
| wikiann | shared_ner | random_augmented | 36.98 ± 0.26 | 32.79 ± 0.87 | 42.43 ± 0.88 |

## Seed-paired differences

Percentage points, mean ± sample SD of paired differences. Per-arm/per-seed values and each seed’s differences are in the accompanying CSVs.

| Dataset | Label space | Selector minus | Span F1 | Precision | Recall |
|---|---|---|---:|---:|---:|
| conll2003 | native | selector_minus_baseline | -0.47 ± 0.61 | -0.46 ± 0.61 | -0.47 ± 1.62 |
| conll2003 | native | selector_minus_random_augmented | 0.18 ± 0.35 | 0.20 ± 0.35 | 0.15 ± 0.35 |
| conll2003 | shared_ner | selector_minus_baseline | -0.22 ± 0.69 | 0.10 ± 0.99 | -0.55 ± 1.72 |
| conll2003 | shared_ner | selector_minus_random_augmented | -0.02 ± 0.42 | -0.16 ± 0.43 | 0.12 ± 0.42 |
| wikiann | shared_ner | selector_minus_baseline | -1.28 ± 0.81 | -1.38 ± 1.31 | -1.03 ± 0.88 |
| wikiann | shared_ner | selector_minus_random_augmented | -0.56 ± 0.33 | -0.68 ± 0.38 | -0.34 ± 0.31 |
| wnut | shared_ner | selector_minus_baseline | 1.28 ± 2.05 | 3.40 ± 6.68 | -1.11 ± 3.33 |
| wnut | shared_ner | selector_minus_random_augmented | 0.32 ± 0.58 | -0.53 ± 0.49 | 1.28 ± 0.82 |

## Variant diagnostics

Scorer disagreement is a biased proxy for label flips, measured after generation without changing the draw. Unchanged/zero-drop draws still count as variants.

| Arm | Raw group | Variants | Variant token share | Scorer disagreement | Input changed |
|---|---|---:|---:|---:|---:|
| selector_augmented | all | 152068 | 50.0% | 3.34% | 97.10% |
| selector_augmented | O | 126979 | 50.0% | 1.34% | 99.89% |
| selector_augmented | entity | 25089 | 50.0% | 13.46% | 83.01% |
| random_augmented | all | 152068 | 50.0% | 3.34% | 97.69% |
| random_augmented | O | 126979 | 50.0% | 1.32% | 99.90% |
| random_augmented | entity | 25089 | 50.0% | 13.56% | 86.53% |

## Frozen scorer validation

On its own scorer-validation split (702 sentences, 10480 words), the frozen scorer has **85.28% native span F1** and **0.0925 nats per word** of raw, unsmoothed cross-entropy, with full-sentence inputs. The mean sum of word NLLs per sentence is 1.3805 nats. This is the scorer’s A-validation partition, not the native CoNLL dev set used to select the three downstream classifiers.

## Recalibrated kept fractions

| Selector | Applied to | Raw group | rho | Calibration rows |
|---|---|---|---:|---:|
| g1 | g2 | O | 0.00 | 3630 |
| g1 | g2 | entity | 0.80 | 2911 |
| g2 | g1 | O | 0.00 | 3637 |
| g2 | g1 | entity | 0.85 | 2880 |

## Held-out bounds for selector fills

These are the existing certificate calculation on the rebuilt pilot’s held-out test rows and calibrated cuts: one-sided 95% normal upper bounds on **smoothed fitted-scorer excess cross-entropy, in nats**. They use the same calibration-style independent fills. They are not guaranteed label-error probabilities. Interpreting them as information-loss bounds requires accurate reference posteriors and the relevant fill conditional; selection-conditioned and calibration-style fills are different. No probability conversion is asserted. All arms run regardless of these diagnostics.

| Selector | Group | Kept fraction | n | Absolute CE mean | Absolute CE SD | Absolute CE UCB | CE increase mean | Increase SD | Increase UCB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| g1 | O | 0.00 | 1560 | 0.0964 | 0.5923 | 0.1210 | 0.0264 | 0.3169 | 0.0396 |
| g1 | entity | 0.80 | 1277 | 0.4005 | 0.7943 | 0.4370 | 0.0305 | 0.3207 | 0.0453 |
| g2 | O | 0.00 | 1551 | 0.0739 | 0.4548 | 0.0929 | 0.0148 | 0.2217 | 0.0240 |
| g2 | entity | 0.85 | 1246 | 0.3971 | 0.8277 | 0.4356 | 0.0256 | 0.2604 | 0.0377 |

Absolute CE is −log[(1−gamma) × mean_fill p_scorer(gold) + gamma/C], with gamma=.01. Its UCB is mean + 1.645 × sample SD / sqrt(n). It is reconstructed on the exact existing test rows and fills as the saved CE increase plus raw gamma-smoothed CE; no new fill-ins are sampled. The increase and absolute CE have separate standard deviations and bounds.

## Timing and artifacts

Variant generation: **7.6 minutes**. Three-seed fitting of all three arms: **0.3 minutes**, including dev checkpoint evaluations. Training plus raw test evaluation: 0.4 minutes. End-to-end including rebuilt prerequisites: **2.52 hours**.

Raw run: `/home/simonelusetti/lion/results/augmentation_full/f3d6dc697455`. Frozen source run: `/home/simonelusetti/lion/results/augmentation_full/source/28b4f1db440a`. Cached focal features are reused for all seeds/epochs; schedules and predictions are saved per seed. [Implementation](IMPLEMENTATION_8.md), [tables](tables/augmentation/).
