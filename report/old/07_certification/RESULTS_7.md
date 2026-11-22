# Experiment 7 — Cross-fitted context selection for NER

Run `a0529c5f9ea3` finished on 19 November 2026 after **14 h 51 min**. Both selectors kept zero context subwords on their 256 certification rows. Both sufficiency tests rejected at the minimum attainable p-value of 0.0196 under the run's MLM sampler. The classifier trained on joint-fill text lost **31.3 points of span F1** on CoNLL-2003, **17.9** on WNUT, and **21.3** on WikiANN relative to the matched raw-text baseline.

## Data and fixed scorer

One seed-0 run used frozen `bert-base-cased` and its MLM head. The source was CoNLL-2003. WNUT and English WikiANN supplied untouched transfer test sets. The raw-text scorer was a 768→256→9 MLP selected by macro-F1 on its own validation partition. At epoch 9 it reached **94.94% accuracy**, **78.89% macro-F1**, and **0.175 nats** of cross-entropy on that partition.

| Source training partition | Sentences | Labeled focal subwords |
|---|---:|---:|
| Scorer fit | 2,808 | 54,788 |
| Scorer validation | 702 | 13,706 |
| T1 | 5,265 | 101,849 |
| T2 | 5,266 | 102,252 |
| Total | 14,041 | 272,595 |

Training assigned each WordPiece the original word's BIO tag. Source validation and all downstream tests used the first WordPiece of each original word. The classifier comparison used only T1 and T2. [Sample counts](tables/certification/counts.csv) · [Scorer history](tables/certification/scorer_history.csv)

## Selector training and masks

The two HardKuma selectors used a three-layer MLP with hidden widths 128 and 64. Each trained for five epochs on 4,096 labeled focals from its own fold, with four precomputed single-position MLM draws per position and a size penalty of 0.05 per expected kept context subword. All nine labels were represented, and about 76% of sampled focals had tag `O`. The held-out mask kept the more probable HardKuma endpoint at each context position. The focal and boundary tokens were protected.

| Selector | Fit fold | Applied to | Sampled sentences | Epoch 1 loss (nats) | Epoch 5 loss (nats) | Expected kept context, epochs 1 → 2 → 5 |
|---|---|---|---:|---:|---:|---:|
| g1 | T1 | T2 | 2,592 | 0.2457 | 0.2223 | 0.476 → 0.0000262 → 0.0000126 |
| g2 | T2 | T1 | 2,585 | 0.2635 | 0.2383 | 0.517 → 0.0000275 → 0.0000092 |

The scorer term alone went from 0.2219 to 0.2223 nats for g1 and from 0.2376 to 0.2383 for g2 as the size penalty approached zero. On the held-out certification rows, g1 dropped 16.34 context subwords per focal on average and g2 dropped 18.20. Neither kept any context beyond the focal. [g1 history](tables/certification/selector_g1_history.csv) · [g2 history](tables/certification/selector_g2_history.csv) · [g1 label counts](tables/certification/selector_g1_sample.csv) · [g2 label counts](tables/certification/selector_g2_sample.csv) · [Mask sizes](tables/certification/mask_sizes.csv)

## Importance tests and certificate

Each fold used one random labeled focal per sampled sentence and 50 accepted draws per nonempty drop set. The sufficiency CRT ranked the sum of the original gold-label log probabilities against sums from joint MLM fills, with ties counted against rejection. The reported effect is the row mean of original log probability minus the log of mean fill probability. The minimality CRT would resample one kept context position. Both folds had zero eligible minimality rows, so there was no p-value or effect for minimality.

| Selector | Sufficiency rows with draws | Pooled rows | Proposals | Same-mask acceptance | p | Effect (nats) | MLM top-1 error | MLM NLL / entropy (nats per token) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| g1 | 255/255 | 256 | 12,755 | 100% | 0.01961 | +0.5676 | 81.06% | 5.735 / 4.459 |
| g2 | 256/256 | 256 | 12,805 | 100% | 0.01961 | +0.4936 | 80.85% | 5.693 / 4.523 |

One g1 row had an empty drop set and contributed exact ties. The 100% acceptance describes whether the same selector mask recurred. The masks were constant on the sampled originals and remained unchanged under the proposed fills. MLM diagnostics used teacher-forced left-to-right regeneration.

| Selector | Rows | Full-context CE | Smoothed fill CE | CE gap | 95% normal UCB | 95% Hoeffding UCB |
|---|---:|---:|---:|---:|---:|---:|
| g1 | 256 | 0.1774 | 0.7325 | +0.5551 | 0.8736 | 1.2528 |
| g2 | 256 | 0.1667 | 0.6613 | +0.4946 | 0.7894 | 1.1816 |

All certificate values are in nats. The fill CE averages gold-label probabilities over the 50 draws and mixes in 1% uniform mass before taking the negative log. The gap compares this result to the fixed scorer on the raw version of the same rows. [CRT statistics](tables/certification/crt.csv) · [Certificate statistics](tables/certification/certificate.csv)

## Cleaning and matched classifiers

Cleaning used one accepted joint fill from the opposite-fold selector per focal with a nonempty drop set. It reused 256 certification records per fold, including the g1 identity record. It wrote 204,101 examples and preserved the focal token ID, position, and label on every example. Across both folds, **5,392,248 of 5,521,566** context positions changed. At least one sibling WordPiece changed in **76,267 of 77,874** examples whose focal belonged to a multi-piece word.

| Selector | Cleaned fold | Examples | Reused certification records | New proposals |
|---|---|---:|---:|---:|
| g1 | T2 | 102,252 | 256 | 101,983 |
| g2 | T1 | 101,849 | 256 | 101,571 |

One saved example started with `Rwanda says [Z] ##aire ex ##pel ##s …`. The replacement turned the word fragment into `[Z] ##on ##ging`, while the retained `Z` kept its original `B-LOC` label. In another example, `Scott [Eric] ##ks ##on …` became `GM [Eric] ##uer …` with `I-PER` still attached to the focal. Brackets mark the fixed focal position.

Two fresh MLP heads trained on the same 204,101 examples with either the filled or raw features. Both selected their checkpoints on raw CoNLL-2003 validation and evaluated raw input. Cleaned selected epoch 4 with 60.36% validation macro-F1. The matched baseline selected epoch 9 with 89.15%.

| Test set and label space | Test words | Accuracy, cleaned / baseline | Macro-F1, cleaned / baseline | Span F1, cleaned / baseline | Span F1 difference |
|---|---:|---:|---:|---:|---:|
| CoNLL-2003, native | 46,435 | 91.41% / 97.19% | 59.53% / 84.78% | 52.02% / 83.32% | −31.30 pp |
| CoNLL-2003, shared | 46,435 | 94.14% / 97.96% | 77.78% / 92.80% | 50.81% / 85.22% | −34.41 pp |
| WNUT, shared | 23,394 | 91.92% / 95.52% | 46.53% / 59.73% | 24.36% / 42.26% | −17.91 pp |
| WikiANN, shared | 80,326 | 68.08% / 75.99% | 53.33% / 68.12% | 15.76% / 37.07% | −21.31 pp |

The shared spaces use O/PER/ORG/LOC. Shared span F1 retains BIO boundaries when mapping the original tags to those types. The cleaned classifier split *Manchester United* into two organizations and treated *New Jersey* in a standings row as two locations. The baseline recovered both gold spans. CoNLL-2003 `I-ORG` recall was 28.7% for cleaned and 83.4% for baseline. On WNUT, cleaned proposed 1,793 spans for 810 gold spans and had 17.7% precision. Baseline proposed 993 with 38.4% precision. Cleaned achieved 40.7% accuracy on entity focal tags on its own cleaned training inputs. [Classifier metrics](tables/certification/performance.csv) · [Transfer coverage](tables/certification/target_coverage.csv)

## Post-run diagnosis

- **The sufficiency CRT detected the failure.** The frozen scorer gave the dropped-context inputs lower gold-label scores in both folds. The CRT ran on held-out data after selector training and supplied no training gradient. Its rejection is evidence against sufficiency under the approximate MLM sampler. Exact calibration depends on the quality of that conditional sampler and on independent rows.
- **The training approximation made extensive dropping cheap.** A replay on 160 held-out focals, including 64 entity focals, found that single-position replacement draws reproduced about half the original context tokens. For the entity focals, mean scorer loss was 0.44 nats on raw text, 0.67 with the four training-style draws, 2.40 after averaging 50 joint draws, and 3.94 for a single joint draw on average. The single-position draws condition on original context that can also be selected for dropping. A gradient replay showed that the scorer loss reached fresh gates. For the fitted selector on the same small batch, all sampled context gates clamped to zero and the scorer gradient through those gates was zero.
- **The CoNLL-2003 split uses sentences as its grouping unit.** The cached data lacks article IDs. There were 290 distinct tokenized sentences present in both selector folds. Three certification rows repeated an exact sentence, focal, and label in that selector's own capped training sample. The normal and Hoeffding bounds reported above assume independent rows.
- **The audit reproduced the downstream measurements.** A direct BERT replay matched the saved cleaned features to fp16 rounding and reproduced both heads' test tags on a sample. An independent BIO span scorer reproduced the reported span F1. The shared token metrics currently sum posterior probability by type, while the pilot mapped the winning BIO tag. Recomputing the latter moved macro-F1 by less than 0.8 points and preserved span F1. The sampler also emitted 75 BERT `[unused…]` tokens in the cleaned corpus.

This run measures the cost of near-total context replacement, including the loss of pieces of the labeled word. The transfer value of a selector that retains useful entity context remains open.

## Measured runtime and provenance

The run used CPU with 16 threads and two eight-core groups for the fold stages. Fold durations overlap. The prior full-suite run was still using CPU during the first few minutes of this run.

| Stage | Elapsed |
|---|---:|
| Prepare | 11 s |
| Fixed scorer and raw feature cache | 5 min 43 s |
| Selectors g1 / g2 | 1 h 7 min 44 s / 1 h 15 min 24 s |
| Certification g1 / g2 | 47 min 50 s / 48 min 48 s |
| Cleaning with g1 / g2 | 12 h 38 min 44 s / 11 h 34 min 40 s |
| Classifier fit and evaluation | 2 min 24 s |
| End to end | 14 h 51 min 26 s |

Raw artifacts and per-stage timing files are under `results/certification/model-x-conll2003/a0529c5f9ea3/`. The command below records the overrides used with the Experiment 7 implementation at commit `29da2a3`.

```bash
USE_TF=0 python3 -B -m lion.certification.run runtime=cpu runtime.threads=16 \
  'runtime.selector_cpu_groups=[0-7,8-15]' certification.source=conll2003 \
  'certification.ner_targets=[wnut,wikiann]' certification.selector_train_focals_per_fold=4096 \
  certification.draws=50 certification.results=results/certification/model-x-conll2003
```
