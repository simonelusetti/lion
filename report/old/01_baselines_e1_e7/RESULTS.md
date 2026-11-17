# Results: over-contextualization diagnostics

This is the completed experimental record. All eight full-data baselines and all stages E1–E7 have finished. The diagnostic cohort is the original six available native datasets plus the derived shared NER condition; WikiANN and WNUT were repaired later and appear only in the baseline table. No additional experiments were run to produce this report.

Accuracy, F1, balanced accuracy and flip rates below are percentages; differences are percentage points (pp). Diagnostic entries show mean ± sample SD over classifier seeds 0/1/2 on one fixed sample. These SDs do not measure data-sampling uncertainty. KL, MMD and JS use their unscaled values. Full implementation details are in [IMPLEMENTATION.md](IMPLEMENTATION.md).

## Full-data baselines

All use frozen cased BERT block 12 and the MLP, seed 0, with validation-selected checkpoints. These fits use complete native training data. They are not the matched-sample full-view models used in E1–E7.

| Dataset | Labels | Train / val / test sentences | Accuracy | Macro F1 | Span F1 |
|---|---|---|---|---|---|
| Few-NERD | 9 | 131,767 / 18,824 / 37,648 | 93.63 | 78.09 | 66.21 |
| WikiANN | 7 | 20,000 / 10,000 / 10,000 | 86.82 | 79.29 | 56.37 |
| CoNLL-2003 | 9 | 14,041 / 3,250 / 3,453 | 95.81 | 79.71 | 76.10 |
| WNUT-17 | 13 | 3,394 / 1,009 / 1,287 | 93.88 | 28.78 | 33.63 |
| EWT UPOS | 17 | 12,544 / 2,001 / 2,077 | 91.08 | 83.88 | n/a |
| EWT dependency | 51 | 12,544 / 2,001 / 2,077 | 80.82 | 55.76 | n/a |
| GUM discourse | 33 | 11,314 / 1,575 / 1,464 | 26.83 | 22.29 | n/a |
| CoNLL-2000 | 23 | 8,042 / 894 / 2,012 | 94.02 | 49.47 | 90.50 |

WikiANN and WNUT initially failed before training. The loaders were repaired to use the original namespaced WikiANN repository and original WNUT parquet conversion. Both retries completed 10 epochs; their saved predictions reproduce the final reports. Run paths, token counts and unrounded scores are in [baselines.csv](tables/baselines.csv).

## Diagnostic sample and coverage

| Dataset | Train focals | Validation focals | Test focals | Z |
|---|---|---|---|---|
| Few-NERD | 4,096 | 1,023 | 2,047 | no native Z |
| CoNLL-2003 | 4,061 | 1,022 | 2,043 | no native Z |
| EWT UPOS | 4,017 | 998 | 1,971 | 5 EWT genres |
| EWT dependency | 4,017 | 998 | 1,971 | 5 EWT genres |
| GUM discourse | 16,309 | 3,138 | 2,916 | 15 GUM genres |
| CoNLL-2000 | 4,094 | 1,024 | 2,047 | no native Z |
| Shared NER | 7,762 | 1,951 | 3,922 | 2 dataset identities |

Native splits stay separate. Shared NER reuses Few-NERD and CoNLL-2003 examples after mapping to O/PER/ORG/LOC; it is not an independent corpus. UPOS and dependency relations also share EWT sentences. [sample_sizes.csv](tables/sample_sizes.csv) includes sentence counts.

## E1 — does context help?

**Implementation:** train one MLP per view on identical sampled focal words. The table reports test accuracy; the next table reports macro F1.

| Dataset | Full accuracy | Word-only accuracy | Context-only accuracy |
|---|---|---|---|
| Few-NERD | 90.46 ± 0.20 | 85.96 ± 0.12 | 87.27 ± 0.50 |
| CoNLL-2003 | 89.57 ± 0.13 | 85.17 ± 0.08 | 87.57 ± 0.10 |
| EWT UPOS | 74.63 ± 0.18 | 76.19 ± 0.26 | 54.66 ± 0.86 |
| EWT dependency | 64.82 ± 0.53 | 60.71 ± 0.16 | 51.80 ± 0.32 |
| GUM discourse | 24.66 ± 0.15 | 15.71 ± 0.69 | 21.94 ± 0.36 |
| CoNLL-2000 | 84.68 ± 0.40 | 75.54 ± 0.57 | 80.10 ± 0.10 |
| Shared NER | 93.01 ± 0.17 | 90.48 ± 0.12 | 89.99 ± 0.37 |

| Dataset | Full macro F1 | Word-only macro F1 | Context-only macro F1 |
|---|---|---|---|
| Few-NERD | 59.85 ± 0.92 | 31.61 ± 0.57 | 53.85 ± 1.83 |
| CoNLL-2003 | 57.95 ± 1.21 | 41.18 ± 0.26 | 50.22 ± 1.09 |
| EWT UPOS | 65.49 ± 0.48 | 69.55 ± 0.85 | 46.37 ± 0.75 |
| EWT dependency | 32.49 ± 1.33 | 27.13 ± 0.56 | 25.85 ± 0.47 |
| GUM discourse | 19.89 ± 0.64 | 6.66 ± 0.30 | 16.74 ± 0.60 |
| CoNLL-2000 | 29.21 ± 0.09 | 23.25 ± 0.88 | 27.12 ± 0.34 |
| Shared NER | 79.29 ± 0.67 | 67.84 ± 0.41 | 73.65 ± 0.49 |

**Measurement:** full context beats word-only on five of the six native tasks. UPOS is the exception: word-only accuracy is 76.19%, versus 74.63% for full context. Full context also has the highest macro F1 on those five tasks; word-only has the highest UPOS macro F1.

Paired full-minus-word-only accuracy gains by training lexical entropy are:

| Dataset | Zero entropy: n / gain pp | High entropy: n / gain pp |
|---|---|---|
| Few-NERD | 426 / 5.79 ± 0.49 | 240 / 18.75 ± 0.72 |
| CoNLL-2003 | 1,064 / 2.98 ± 0.24 | 53 / 23.27 ± 2.88 |
| EWT UPOS | 774 / -3.66 ± 0.30 | 92 / 28.26 ± 1.09 |
| EWT dependency | 326 / 0.61 ± 1.23 | 755 / 11.66 ± 0.26 |
| GUM discourse | 114 / 12.57 ± 1.83 | 2,470 / 8.43 ± 0.89 |
| CoNLL-2000 | 288 / 8.68 ± 1.59 | 342 / 27.49 ± 1.75 |
| Shared NER | 1,238 / 3.02 ± 0.31 | 205 / 17.72 ± 2.03 |

**Interpretation:** context gains concentrate more on high-entropy forms than unambiguous forms in five native tasks and shared NER. GUM differs: high-entropy gain is 8.43 pp, versus 12.57 pp for zero entropy. The high-entropy CoNLL-2003 bin has only 53 test focal words. These are lexical-ambiguity associations; they do not establish that harmful context is spurious. A full-data baseline versus a small-sample diagnostic score is not a controlled context effect.

[Overall scores](tables/e1_scores.csv), [per-label scores](tables/e1_per_label.csv), [frequency/entropy gains and supports](tables/e1_gains_with_support.csv), [figure](figures/e1_scores.png).

## E2 — masking as conditioning on less

**Implementation:** compare word-only-trained and full-trained classifiers on the same word-only vectors. Targets are empirical full-training label distributions for types with frequency ≥20 and multiple labels. KL is empirical-distribution-to-mean-prediction, averaged equally across types. ECE is occurrence-weighted, using 10 confidence bins.

| Dataset | Types / test tokens | KL: mask-trained | KL: full-trained | ECE %: mask-trained | ECE %: full-trained |
|---|---|---|---|---|---|
| Few-NERD | 557 / 1485 | 0.272 ± 0.006 | 0.879 ± 0.010 | 2.27 ± 0.42 | 14.32 ± 2.07 |
| CoNLL-2003 | 102 / 525 | 0.846 ± 0.032 | 2.881 ± 0.163 | 3.91 ± 0.43 | 11.57 ± 1.09 |
| EWT UPOS | 221 / 835 | 0.488 ± 0.005 | 2.021 ± 0.065 | 3.58 ± 0.70 | 25.67 ± 5.02 |
| EWT dependency | 355 / 1096 | 0.892 ± 0.006 | 2.925 ± 0.159 | 3.14 ± 0.16 | 20.99 ± 2.84 |
| GUM discourse | 475 / 2058 | 0.673 ± 0.031 | 1.786 ± 0.057 | 3.31 ± 1.31 | 22.24 ± 2.00 |
| CoNLL-2000 | 414 / 1322 | 0.336 ± 0.012 | 2.084 ± 0.083 | 2.73 ± 0.05 | 31.30 ± 3.44 |

**Measurement:** matched masked training lowers both mean type KL and ECE on every native task. For example, dependency KL falls from 2.925 to 0.892; CoNLL-2000 ECE falls from 31.30% to 2.73%.

Illustrative types were selected by training entropy ≥0.2 nats and highest held-out support (at least two tokens), taking up to two per dataset. Selection did not depend on which classifier performed better:

| Dataset | Type | Train / test count | KL: mask-trained | KL: full-trained |
|---|---|---|---|---|
| Few-NERD | of | 93,933 / 53 | 0.031 | 0.109 |
| Few-NERD | and | 83,862 / 48 | 0.009 | 0.148 |
| CoNLL-2003 | DIVISION | 94 / 6 | 1.462 | 0.654 |
| CoNLL-2003 | Group | 25 / 6 | 4.338 | 6.189 |
| EWT UPOS | to | 5,076 / 31 | 0.177 | 1.594 |
| EWT UPOS | in | 2,911 / 20 | 0.061 | 2.249 |
| EWT dependency | to | 5,076 / 31 | 0.221 | 1.330 |
| EWT dependency | is | 2,152 / 22 | 0.124 | 1.846 |
| GUM discourse | . | 9,201 / 215 | 0.142 | 1.261 |
| GUM discourse | , | 10,720 / 119 | 0.091 | 0.954 |
| CoNLL-2000 | , | 9,679 / 91 | 0.007 | 0.761 |
| CoNLL-2000 | to | 4,525 / 49 | 0.137 | 3.739 |

**Interpretation:** the aggregate result supports training under the masking distribution rather than applying a full-trained readout to masked inputs. It does not show exact posterior recovery, and the benefit is not uniform over types: CoNLL-2003 `DIVISION` is a counterexample in the illustration table. Finite sampled training, mask length and position remain alternative sources of mismatch.

[Scores and seed spread](tables/e2_summary.csv), [example empirical/predicted distributions](tables/e2_examples.csv), [occurrence-level KL and Brier scores](tables/e2_seed_scores.csv), [figure](figures/e2_kl.png).

## E3 — is Z linearly readable within Y?

**Implementation:** balanced linear probes fitted separately within each gold label. Only label/domain cells meeting 32/8/8 train/validation/test support are used. Each dataset/view entry averages eligible labels equally within a seed.

| Dataset | Eligible labels | Mean chance | Full BA | Word-only BA | Context-only BA |
|---|---|---|---|---|---|
| EWT UPOS | 9 | 25.93 | 35.48 ± 0.36 | 28.95 ± 1.65 | 33.72 ± 0.68 |
| EWT dependency | 8 | 29.79 | 39.19 ± 1.94 | 32.07 ± 2.00 | 37.04 ± 0.34 |
| GUM discourse | 11 | 20.78 | 47.26 ± 1.21 | 25.40 ± 1.08 | 48.22 ± 1.30 |
| Shared NER | 4 | 50.00 | 83.25 ± 0.84 | 66.98 ± 0.32 | 84.88 ± 1.56 |

**Measurement:** context-only is above mean chance for every nuisance dataset. GUM reaches 48.22% against 20.78% chance; shared NER reaches 84.88% against 50%. Word-only is also above chance, especially for dataset identity (66.98%).

**Interpretation:** this supports the claim that masked context representations retain source/genre information beyond label frequencies. It does not establish a calibrated posterior over latent state and Z, nor show that word-only is nuisance-free. Chance varies by label because eligibility can remove some domains.

[Aggregate probes](tables/e3_aggregate.csv), [per-label probes](tables/e3_per_label.csv), [figure](figures/e3_readability.png).

## E4 — can linear erasure preserve Y?

**Implementation:** fit a single training-only affine conditional-covariance erasure to full vectors, then retrain both the Y MLP and conditional Z probes. A positive F1 drop means erasure hurt the Y classifier.

| Dataset | Estimated removed rank | Y macro-F1 drop pp | Post-erasure Z BA | Mean chance | Labels above chance in all seeds |
|---|---|---|---|---|---|
| EWT UPOS | 65 | 0.14 ± 1.48 | 25.00 ± 1.10 | 25.93 | 1 / 9 |
| EWT dependency | 149 | 2.22 ± 0.92 | 29.30 ± 1.71 | 29.79 | 1 / 8 |
| GUM discourse | 433 | 9.95 ± 1.14 | 19.10 ± 1.02 | 20.78 | 3 / 11 |
| Shared NER | 4 | 0.58 ± 0.10 | 50.55 ± 0.63 | 50.00 | 1 / 4 |

**Measurement:** average conditional Z probing falls to approximately chance. Y loss is small for UPOS (0.14 pp) and shared NER (0.58 pp), moderate for dependency labels (2.22 pp), and large for GUM (9.95 pp). GUM macro F1 falls from 19.89% to 9.94%. The fitted removed subspace is also much larger for GUM (433 dimensions) than shared NER (4). Training cross-covariance norms fall from 0.15–0.65 to approximately 1.3–2.1 × 10⁻⁶.

**Interpretation:** the results are consistent with different costs of separating the measured domain covariance from label information. They do not prove a universal additive/entangled dichotomy. Average near-chance probing hides residual positive results for individual labels; the final column counts labels above chance in all three seeds, without a significance claim. In particular, the GUM loss alone is insufficient to prove causal entanglement.

[Y and aggregate Z results](tables/e4_aggregate.csv), [per-label post-erasure probes](tables/e4_z_per_label.csv), [projection diagnostics](tables/e4_erasure.csv), [figure](figures/e4_label_drop.png).

## E5 — within-label dependence and ambiguity

**Implementation:** unbiased RBF MMD between prediction distributions within a label and domain pair; document-block permutations, 199 permutations per cell, and BH correction within each seed across datasets/views/strata. Word-distribution JS uses the same sampled groups. The following descriptive MMD averages use unstratified eligible label/domain cells:

| Dataset | Full MMD² | Word-only MMD² | Context-only MMD² | Word-distribution JS (nats) |
|---|---|---|---|---|
| EWT UPOS | 0.0104 ± 0.0017 | 0.0293 ± 0.0005 | 0.0183 ± 0.0003 | 0.478 |
| EWT dependency | 0.0303 ± 0.0032 | 0.0701 ± 0.0013 | 0.0428 ± 0.0026 | 0.421 |
| Shared NER | 0.0111 ± 0.0007 | 0.0440 ± 0.0019 | 0.0107 ± 0.0013 | 0.640 |

| Dataset | Tests across seeds/views/strata | Raw p ≤ .05 | BH q ≤ .05 |
|---|---|---|---|
| Shared NER | 166 | 54 | 0 |
| EWT dependency | 636 | 223 | 0 |
| EWT UPOS | 963 | 174 | 0 |

**Measurement:** none of the 1,765 tests survives BH correction at 5%; the smallest adjusted q is 0.05193. Word-only has larger average MMD than full and context-only in all three measured dataset conditions, opposite to the proposed descriptive ordering under this kernel normalization. Lexical distributions also differ across domains.

**Missing coverage:** GUM has two test documents per genre and the protocol requires three, so it has no eligible E5 measurements. Some high-entropy strata were measured, but no matched high-versus-zero lexical-entropy contrasts within the same label/domain pair were produced; the concentration-on-ambiguity claim is therefore not resolved.

**Interpretation:** the current analysis does not provide corrected statistical support for the proposed dependence claim. Non-rejection does not establish invariance: permutation resolution is only 0.005, support is sparse, and the correction family is large. Cross-view MMD ordering is descriptive because bandwidths are fitted separately. Shared NER uses sentence rather than document blocks, so its p values are exploratory. Word JS is an empirical lexical difference, not direct observation of latent states.

[Unstratified effects](tables/e5_unstratified.csv), [stratified effects and cell counts](tables/e5_strata.csv), [test counts](tables/e5_test_counts.csv), [individual tests](tables/e5_seed_scores.csv), [figure](figures/e5_dependence.png).

## E6 — cross-domain degradation

**Implementation:** train and validate on one domain only, then evaluate source and other-domain native test samples. The table averages ordered source–target pairs equally within each seed. Positive degradation is in-domain accuracy minus out-of-domain accuracy, in pp.

| Dataset | Full drop pp | Word-only drop pp | Context-only drop pp | Pairs following context > full > word |
|---|---|---|---|---|
| EWT UPOS | 8.57 ± 0.13 | 5.88 ± 0.33 | 5.21 ± 0.52 | 11.67 ± 2.89% |
| EWT dependency | 5.82 ± 0.65 | 4.41 ± 0.24 | 5.06 ± 0.31 | 23.33 ± 10.41% |
| GUM discourse | 9.95 ± 0.27 | 5.97 ± 0.26 | 9.75 ± 0.23 | 31.27 ± 2.25% |
| Shared NER | 5.36 ± 0.07 | 3.53 ± 0.06 | 5.92 ± 0.06 | 50.00 ± 0.00% |

**Measurement:** only shared NER follows the predicted ordering in aggregate: context-only loses 5.92 pp, full loses 5.36 pp, and word-only loses 3.53 pp. The three native UD tasks instead have larger full-view than context-only drops. The strict ordering holds in only about 12%, 23%, 31%, and 50% of ordered domain pairs for UPOS, dependencies, GUM and shared NER respectively, averaging over seeds.

Absolute out-of-domain accuracies are:

| Dataset | Full OOD accuracy | Word-only OOD accuracy | Context-only OOD accuracy |
|---|---|---|---|
| EWT UPOS | 59.24 ± 0.39 | 64.74 ± 0.10 | 46.29 ± 0.45 |
| EWT dependency | 47.46 ± 0.39 | 48.56 ± 0.17 | 41.89 ± 0.16 |
| GUM discourse | 13.11 ± 0.18 | 11.54 ± 0.04 | 12.87 ± 0.10 |
| Shared NER | 87.95 ± 0.07 | 86.85 ± 0.12 | 84.59 ± 0.39 |

**Interpretation:** the simple universal degradation ordering is not supported. A weaker context-only classifier can have less accuracy left to lose, so degradation must be read alongside its absolute accuracy. Genre label priors, lexical coverage and annotation differences also remain possible explanations. Per-label recall/MMD correlations were recorded for 288 supported conditions; they are descriptive links between different fitted classifiers, not evidence that measured dependence caused a shift failure.

[OOD aggregates](tables/e6_ood.csv), [all source–target pairs](tables/e6_domain_pairs.csv), [ordering frequencies](tables/e6_ordering.csv), [per-label results](tables/e6_per_label.csv), [E5 correlations](tables/e6_e5_correlations.csv), [figure](figures/e6_degradation.png).

## E7 — localized mention substitutions

**Implementation:** replace a neighboring or focal entity span with a distinct training mention of the same type and matching per-word WordPiece lengths. Evaluate the already-trained view classifiers. Tables combine single- and multiword focal mentions; the supporting CSV separates them.

### Neighbor replacement

| Dataset | Paired interventions | Full flips % | Word-only flips % | Context-only flips % |
|---|---|---|---|---|
| Few-NERD | 99 | 11.78 ± 0.58 | 0.00 ± 0.00 | 13.13 ± 1.01 |
| CoNLL-2003 | 84 | 17.46 ± 2.48 | 0.00 ± 0.00 | 16.27 ± 3.64 |


### Focal replacement

| Dataset | Paired interventions | Full flips % | Word-only flips % | Context-only flips % |
|---|---|---|---|---|
| Few-NERD | 121 | 42.42 ± 0.48 | 59.50 ± 1.43 | 30.85 ± 2.08 |
| CoNLL-2003 | 128 | 41.41 ± 1.56 | 52.08 ± 1.19 | 9.90 ± 1.97 |


**Measurement:** neighbor substitutions change full-view predictions for 11.78% of Few-NERD and 17.46% of CoNLL-2003 focal tokens, while word-only has zero flips. Context-only also changes (13.13% and 16.27%). Focal replacement changes word-only predictions more often than full-view predictions. The single-word focal/context-only control has zero flips for all seeds (26 Few-NERD and 72 CoNLL-2003 examples).

**Interpretation:** classifiers using context are sensitive to changes in nearby entity identity. This supports the localized sensitivity prediction, but not the stronger assertion that every substitution is an exact nuisance-only intervention: coherence and semantic label preservation were not independently judged. Focal substitution changes entity identity; multiword replacements can change other visible words even when the focal token is masked.

The neighbor word-only control has small probability L1 changes despite zero argmax flips: about 0.0067 for Few-NERD and 0.0055 for CoNLL-2003. This is an empirical control floor; its numerical/batching causes were not separately isolated. Changes at that scale should not be interpreted as semantic effects.

[Flip rates, L1 changes, before/after accuracy and length strata](tables/e7_aggregate.csv), [individual intervention records](tables/e7_interventions.csv), [figure](figures/e7_flips.png).

## What the experiment establishes

1. Training under the masked input distribution is substantially better than applying a full-trained classifier to masked vectors, for aggregate lexical-distribution matching and calibration.
2. Context-only frozen representations retain linearly readable genre/source information conditional on the recorded label. Word-only is not nuisance-free.
3. Removing conditional domain covariance has task-dependent label costs: little average loss for UPOS/shared NER and much more for GUM, with incomplete evidence of erasure in individual labels.
4. Neither corrected E5 tests nor the universal domain-shift ordering establish the complete proposed causal framework. E5 ambiguity concentration remains unresolved.
5. Nearby entity substitutions measurably affect context-using predictions, while the relevant argmax controls remain unchanged.

These observations support parts of the diagnostic decomposition, not identification of a latent nuisance/state causal model. The report records unsupported or missing outcomes rather than replacing them with favorable conditions.

## Integrity and provenance

All 702 diagnostic readouts were checked for finite weights and predictions, complete 10-epoch histories, and correct validation checkpoint selection. Cached-view shapes and E1 metrics were checked against tensors; all eight completed baseline reports reproduce from saved predictions. E5 q values reproduce exactly, and all seven PNGs decode. These checks establish artifact consistency, not the truth of causal assumptions.

The [provenance manifest](provenance.json) identifies baseline runs and hashes the numerical sources and implementation files. [Skip counts](tables/skip_counts.csv) summarize unavailable cells; detailed reasons remain under `results/eN/skips.csv`. The original phase inventory is retained as a historical record rather than rewritten to pretend WikiANN/WNUT were part of the diagnostics.
