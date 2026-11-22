# Step 1: sensitivity to selector-droppable context

Generation: **27.4 min**; measurement: **0.2 min**; end-to-end: **0.62 h**. No model was trained or selected. Steps 2 and 3 were not implemented.

All words in each raw test set use both frozen selectors, 4 independent calibration-style fills per mask kind, and the existing full-scale classifiers (three arms × three seeds) plus the frozen scorer. All distributions retain the native nine BIO tags and gamma=.01 uniform mixing; no shared-label projection is applied to these sensitivity measures.

Main tables use the **row-wise mean of g1 and g2**, then average over rows. S is KL(original || filled), in nats; flips are percentages. Classifier ± values are sample SD across three seeds; scorer has one fitted model. Complete per-classifier/per-seed g1, g2 and mean results, including standard errors over rows, are in [per_classifier_seed.csv](tables/sensitivity/per_classifier_seed.csv). The seed summary also reports row SE after seed-averaging; folds and draws are never treated as independent extra rows.

## conll2003 — in-domain reference

| Classifier | Group | Rows | Selector S | Random S | S ratio | Selector flips % | Random flips % | ID label reference |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | O | 38303 | 0.0573 ± 0.0109 | 0.0572 ± 0.0108 | 1.002 | 1.6824 ± 0.3400 | 1.6773 ± 0.3209 | 0.1070 |
| baseline | entity | 8132 | 0.1024 ± 0.0007 | 0.1283 ± 0.0005 | 0.798 | 4.9844 ± 0.0441 | 5.9200 ± 0.1000 | 0.4363 |
| baseline | all | 46435 | 0.0652 ± 0.0090 | 0.0696 ± 0.0090 | 0.935 | 2.2607 ± 0.2867 | 2.4203 ± 0.2483 | 0.1647 |
| selector_augmented | O | 38303 | 0.0205 ± 0.0040 | 0.0201 ± 0.0040 | 1.023 | 0.5961 ± 0.1633 | 0.5920 ± 0.1595 | 0.1070 |
| selector_augmented | entity | 8132 | 0.0731 ± 0.0010 | 0.0977 ± 0.0006 | 0.748 | 4.6908 ± 0.1615 | 5.6433 ± 0.2400 | 0.4363 |
| selector_augmented | all | 46435 | 0.0297 ± 0.0033 | 0.0337 ± 0.0034 | 0.882 | 1.3132 ± 0.1113 | 1.4766 ± 0.0901 | 0.1647 |
| random_augmented | O | 38303 | 0.0201 ± 0.0039 | 0.0196 ± 0.0039 | 1.025 | 0.5920 ± 0.1494 | 0.5881 ± 0.1450 | 0.1070 |
| random_augmented | entity | 8132 | 0.0750 ± 0.0017 | 0.0929 ± 0.0020 | 0.808 | 4.7257 ± 0.2093 | 5.6300 ± 0.1926 | 0.4363 |
| random_augmented | all | 46435 | 0.0297 ± 0.0033 | 0.0325 ± 0.0034 | 0.915 | 1.3159 ± 0.0896 | 1.4711 ± 0.0863 | 0.1647 |
| scorer | O | 38303 | 0.0365 | 0.0365 | 1.000 | 0.7989 | 0.8133 | 0.1070 |
| scorer | entity | 8132 | 0.0861 | 0.1064 | 0.809 | 5.7197 | 6.6235 | 0.4363 |
| scorer | all | 46435 | 0.0452 | 0.0487 | 0.927 | 1.6607 | 1.8308 | 0.1647 |

## wikiann — UNCERTIFIED transfer

| Classifier | Group | Rows | Selector S | Random S | S ratio | Selector flips % | Random flips % | ID label reference |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | O | 49937 | 0.1841 ± 0.0284 | 0.1841 ± 0.0285 | 1.000 | 4.3103 ± 0.7028 | 4.3324 ± 0.7090 | 0.1070 |
| baseline | entity | 30389 | 0.1321 ± 0.0021 | 0.1478 ± 0.0021 | 0.894 | 7.7911 ± 0.1467 | 8.8726 ± 0.0985 | 0.4363 |
| baseline | all | 80326 | 0.1644 ± 0.0184 | 0.1703 ± 0.0184 | 0.965 | 5.6272 ± 0.4707 | 6.0500 ± 0.4555 | 0.2316 |
| selector_augmented | O | 49937 | 0.1138 ± 0.0265 | 0.1135 ± 0.0264 | 1.003 | 2.6256 ± 0.6496 | 2.6413 ± 0.6500 | 0.1070 |
| selector_augmented | entity | 30389 | 0.0984 ± 0.0017 | 0.1147 ± 0.0020 | 0.858 | 7.6220 ± 0.0834 | 8.7277 ± 0.0311 | 0.4363 |
| selector_augmented | all | 80326 | 0.1080 ± 0.0171 | 0.1139 ± 0.0171 | 0.947 | 4.5159 ± 0.3995 | 4.9439 ± 0.4154 | 0.2316 |
| random_augmented | O | 49937 | 0.1090 ± 0.0265 | 0.1086 ± 0.0264 | 1.003 | 2.5200 ± 0.7182 | 2.5278 ± 0.7243 | 0.1070 |
| random_augmented | entity | 30389 | 0.0979 ± 0.0025 | 0.1122 ± 0.0030 | 0.873 | 7.5707 ± 0.0561 | 8.6839 ± 0.0880 | 0.4363 |
| random_augmented | all | 80326 | 0.1048 ± 0.0171 | 0.1100 ± 0.0172 | 0.952 | 4.4308 ± 0.4263 | 4.8568 ± 0.4481 | 0.2316 |
| scorer | O | 49937 | 0.1148 | 0.1147 | 1.000 | 1.7014 | 1.7645 | 0.1070 |
| scorer | entity | 30389 | 0.1015 | 0.1163 | 0.873 | 7.5188 | 8.6573 | 0.4363 |
| scorer | all | 80326 | 0.1098 | 0.1153 | 0.952 | 3.9022 | 4.3722 | 0.2316 |

## wnut — UNCERTIFIED transfer

| Classifier | Group | Rows | Selector S | Random S | S ratio | Selector flips % | Random flips % | ID label reference |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | O | 21877 | 0.1041 ± 0.0201 | 0.1025 ± 0.0197 | 1.015 | 2.8788 ± 0.6828 | 2.8268 ± 0.6704 | 0.1070 |
| baseline | entity | 1517 | 0.2491 ± 0.0094 | 0.2694 ± 0.0073 | 0.925 | 18.5646 ± 0.5152 | 18.6909 ± 0.4411 | 0.4363 |
| baseline | all | 23394 | 0.1135 ± 0.0189 | 0.1133 ± 0.0183 | 1.001 | 3.8959 ± 0.6308 | 3.8555 ± 0.5988 | 0.1283 |
| selector_augmented | O | 21877 | 0.0432 ± 0.0110 | 0.0432 ± 0.0109 | 1.000 | 0.9378 ± 0.2658 | 0.9262 ± 0.2445 | 0.1070 |
| selector_augmented | entity | 1517 | 0.1730 ± 0.0130 | 0.1794 ± 0.0108 | 0.964 | 18.6635 ± 2.4976 | 17.7406 ± 0.9210 | 0.4363 |
| selector_augmented | all | 23394 | 0.0517 ± 0.0111 | 0.0520 ± 0.0109 | 0.992 | 2.0873 ± 0.3978 | 2.0165 ± 0.2842 | 0.1283 |
| random_augmented | O | 21877 | 0.0406 ± 0.0102 | 0.0405 ± 0.0101 | 1.002 | 0.8563 ± 0.2395 | 0.8618 ± 0.2331 | 0.1070 |
| random_augmented | entity | 1517 | 0.1691 ± 0.0165 | 0.1775 ± 0.0118 | 0.952 | 18.3861 ± 2.4357 | 17.8532 ± 1.6595 | 0.4363 |
| random_augmented | all | 23394 | 0.0489 ± 0.0104 | 0.0494 ± 0.0102 | 0.990 | 1.9930 ± 0.3767 | 1.9636 ± 0.3234 | 0.1283 |
| scorer | O | 21877 | 0.0648 | 0.0635 | 1.020 | 1.4010 | 1.3353 | 0.1070 |
| scorer | entity | 1517 | 0.2393 | 0.2313 | 1.035 | 22.3303 | 20.0560 | 0.4363 |
| scorer | all | 23394 | 0.0761 | 0.0744 | 1.023 | 2.7582 | 2.5493 | 0.1283 |

## Existing label references and cuts

The reference is the existing held-out **absolute gamma-smoothed CE UCB**, not the CE increase. Under deterministic labels, appropriate oracle posteriors and selection-conditioned fills, such a loss can bound label flips. The actual fitted scorer and calibration-style independent fills do not guarantee those assumptions. Values below 1 are numerically non-vacuous references, not proof of a valid label-error certificate. Transfer results are uncertified; their numbers in this column are ID references only. “All” uses the target group-frequency-weighted references; the mean-fold column averages the fold references. These derived references are not new joint 95% certificates.

| Selector | Group | rho | Reference UCB | Below 1 |
|---|---|---:|---:|---|
| g1 | O | 0.00 | 0.1210 | True |
| g1 | entity | 0.80 | 0.4370 | True |
| g2 | O | 0.00 | 0.0929 | True |
| g2 | entity | 0.85 | 0.4356 | True |

## Dropped-set diagnostics

Coincidence means exact equality of the selector and random masks, including zero-drop and accidental matches. The all-context-dropped share is listed separately (excluding rows with no context). Random and selector masks receive independent fills even when their sets coincide.

| Dataset | Fold | Group | Mean dropped words | Same-set % | All context dropped % |
|---|---|---|---:|---:|---:|
| conll2003 | g1 | O | 22.78 | 100.00 | 99.94 |
| conll2003 | g1 | entity | 4.08 | 10.87 | 0.00 |
| conll2003 | g1 | all | 19.51 | 84.39 | 82.44 |
| conll2003 | g2 | O | 22.78 | 100.00 | 99.94 |
| conll2003 | g2 | entity | 3.04 | 14.81 | 0.00 |
| conll2003 | g2 | all | 19.32 | 85.08 | 82.44 |
| wikiann | g1 | O | 12.01 | 100.00 | 100.00 |
| wikiann | g1 | entity | 2.02 | 19.34 | 0.00 |
| wikiann | g1 | all | 8.23 | 69.49 | 62.17 |
| wikiann | g2 | O | 12.01 | 100.00 | 100.00 |
| wikiann | g2 | entity | 1.56 | 25.52 | 0.00 |
| wikiann | g2 | all | 8.06 | 71.82 | 62.17 |
| wnut | g1 | O | 29.07 | 100.00 | 99.87 |
| wnut | g1 | entity | 4.18 | 3.30 | 0.00 |
| wnut | g1 | all | 27.45 | 93.73 | 93.40 |
| wnut | g2 | O | 29.07 | 100.00 | 99.87 |
| wnut | g2 | entity | 3.16 | 4.22 | 0.00 |
| wnut | g2 | all | 27.39 | 93.79 | 93.40 |

Per-row draw-averaged measurements and sentence/focal keys are saved under `results/sensitivity/measure/*_rows.pt`; all four filled focal vectors per mask type are cached under `results/sensitivity/generate/`. Full per-seed ratios and row SEs are in ratios.csv. [Implementation](IMPLEMENTATION_9.md).
