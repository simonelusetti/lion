# Conflict-insertion feasibility: coverage, supply, room

Full native CoNLL-2003 and EWT train/dev/test splits. CoNLL labels collapse B/I to five entity/O classes; eligible slots are O words and singleton gold entities. EWT uses every word and base dependency relations. Only one new context-only MLP per task was fitted; existing layer-run classifiers and the frozen BERT/MLM are reused. No augmentation training.

Context input removes all focal WordPieces and inserts one [MASK]. Its layer-8 MLP selects a checkpoint using a fixed 10% document holdout inside native train, leaving native dev for calibration. Selection uses max-label confidence ≥τ, regardless of whether the top label equals gold. Confident errors remain in every analysis.

Supply uses the original MLM softmax probabilities on whole single-piece alphanumeric tokens (no ##, punctuation or special tokens); it does not renormalize them. Natural draws alone renormalize over this vocabulary. A word needs ≥5 native-training occurrences and a unique most frequent label; tied maxima are excluded. Conflict additionally requires this label differ from gold and gold share ≤10%. Counts include all native-training occurrences, including CoNLL multiword entity words.

The priced quantity (1−p_ctx(y_c))/p_MLM(w*) is a plug-in estimate, not a certificate. A missing conflict candidate gives price ∞. Quartiles use empirical nearest-rank order statistics, including ∞. Dev CE upper bounds are pointwise one-sided 95% CLT bounds (z=1.645), not simultaneous bounds across thresholds/groups and not guarantees for individual contexts.

Pre-run conservative estimate: 6.1 min, assuming all test contexts selected and all eleven insertion slots available. Actual wall-clock: 3.2 min; feature extraction 1.1, context fitting 0.3, supply 0.2, room 1.1, analysis 0.2 min.

Classifier estimates are mean ± sample seed SD over the three saved heads, plus word-row SE. Insertions are averaged within each context before averaging contexts. Bins below 50 supporting contexts have *. Layer 8 appears below; layer 12 and per-seed results are in CSVs. Existing BIO-head probabilities are summed over B/I before the CoNLL type argmax.

## Eligible slots

| Dataset | Split | Slots | All words | Entity words kept | Entity retention % |
| --- | --- | --- | --- | --- | --- |
| conll2003 | train | 184409 | 203621 | 14831 | 43.5655 |
| conll2003 | validation | 46526 | 51362 | 3767 | 43.7871 |
| conll2003 | id | 41897 | 46435 | 3574 | 44.0582 |
| en_ewt_deprel | train | 204578 | 204578 | 0 | — |
| en_ewt_deprel | validation | 25148 | 25148 | 0 | — |
| en_ewt_deprel | id | 25094 | 25094 | 0 | — |

Entity retention counts original gold entity words; singleton eligibility is applied consistently to fitting, calibration and measurement.

## Context-only predictor

| Dataset | Split | Group | Rows | Accuracy % (SE) | CE (SE) |
| --- | --- | --- | --- | --- | --- |
| conll2003 | validation | all | 46526 | 96.0753 (0.0900) | 0.1335 (0.0032) |
| conll2003 | validation | O | 42759 | 98.9757 (0.0487) | 0.0329 (0.0010) |
| conll2003 | validation | entity | 3767 | 63.1537 (0.7861) | 1.2744 (0.0329) |
| conll2003 | id | all | 41897 | 95.7061 (0.0990) | 0.1644 (0.0042) |
| conll2003 | id | O | 38323 | 99.0528 (0.0495) | 0.0337 (0.0011) |
| conll2003 | id | entity | 3574 | 59.8209 (0.8202) | 1.5660 (0.0404) |
| en_ewt_deprel | validation | all | 25148 | 77.3024 (0.2641) | 0.7427 (0.0081) |
| en_ewt_deprel | id | all | 25094 | 76.8471 (0.2663) | 0.7343 (0.0079) |

Dev and test are evaluated after the checkpoint is fixed using training data only.

## 1: dev coverage and held-out CE bound

| Dataset | τ | Group | Selected / population | Coverage % | Error % (SE) | Mean CE | CE upper 95% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| conll2003 | 0.9 | all | 42662 / 46526 | 91.6950 | 1.1415 (0.0514) | 0.0647 | 0.0696 |
| conll2003 | 0.9 | O | 40639 / 42759 | 95.0420 | 0.1033 (0.0159) | 0.0082 | 0.0094 |
| conll2003 | 0.9 | entity | 2023 / 3767 | 53.7032 | 21.9970 (0.9212) | 1.1994 | 1.2900 |
| conll2003 | 0.95 | all | 41411 / 46526 | 89.0061 | 0.8138 (0.0441) | 0.0513 | 0.0560 |
| conll2003 | 0.95 | O | 39701 / 42759 | 92.8483 | 0.0730 (0.0136) | 0.0057 | 0.0068 |
| conll2003 | 0.95 | entity | 1710 / 3767 | 45.3942 | 18.0117 (0.9296) | 1.1102 | 1.2112 |
| conll2003 | 0.99 | all | 38152 / 46526 | 82.0015 | 0.4010 (0.0324) | 0.0305 | 0.0346 |
| conll2003 | 0.99 | O | 36979 / 42759 | 86.4824 | 0.0216 (0.0076) | 0.0020 | 0.0028 |
| conll2003 | 0.99 | entity | 1173 / 3767 | 31.1388 | 12.3615 (0.9614) | 0.9290 | 1.0531 |
| en_ewt_deprel | 0.9 | all | 11349 / 25148 | 45.1288 | 3.8946 (0.1816) | 0.2194 | 0.2350 |
| en_ewt_deprel | 0.95 | all | 8745 / 25148 | 34.7741 | 2.4128 (0.1641) | 0.1538 | 0.1700 |
| en_ewt_deprel | 0.99 | all | 4070 / 25148 | 16.1842 | 0.9582 (0.1527) | 0.0727 | 0.0912 |

Cross-entropy upper-bounds conditional entropy under the stated sampling assumptions. The reported uncertainty concerns mean CE on selected rows, not whether each row fixes a unique label.

## 1: test coverage

| Dataset | τ | Group | Subset | Selected / population | Coverage % (SE) |
| --- | --- | --- | --- | --- | --- |
| conll2003 | 0.9 | all | all | 38438 / 41897 | 91.7440 (0.1345) |
| conll2003 | 0.9 | all | conflict_prone | 12334 / 13040 | 94.5859 (0.1982) |
| conll2003 | 0.9 | O | all | 36357 / 38323 | 94.8699 (0.1127) |
| conll2003 | 0.9 | O | conflict_prone | 11527 / 11746 | 98.1355 (0.1248) |
| conll2003 | 0.9 | entity | all | 2081 / 3574 | 58.2261 (0.8250) |
| conll2003 | 0.9 | entity | conflict_prone | 807 / 1294 | 62.3648 (1.3468) |
| conll2003 | 0.95 | all | all | 37262 / 41897 | 88.9372 (0.1532) |
| conll2003 | 0.95 | all | conflict_prone | 12108 / 13040 | 92.8528 (0.2256) |
| conll2003 | 0.95 | O | all | 35496 / 38323 | 92.6232 (0.1335) |
| conll2003 | 0.95 | O | conflict_prone | 11422 / 11746 | 97.2416 (0.1511) |
| conll2003 | 0.95 | entity | all | 1766 / 3574 | 49.4124 (0.8363) |
| conll2003 | 0.95 | entity | conflict_prone | 686 / 1294 | 53.0139 (1.3874) |
| conll2003 | 0.99 | all | all | 34100 / 41897 | 81.3901 (0.1901) |
| conll2003 | 0.99 | all | conflict_prone | 11453 / 13040 | 87.8298 (0.2863) |
| conll2003 | 0.99 | O | all | 33006 / 38323 | 86.1258 (0.1766) |
| conll2003 | 0.99 | O | conflict_prone | 11018 / 11746 | 93.8021 (0.2225) |
| conll2003 | 0.99 | entity | all | 1094 / 3574 | 30.6100 (0.7709) |
| conll2003 | 0.99 | entity | conflict_prone | 435 / 1294 | 33.6167 (1.3132) |
| en_ewt_deprel | 0.9 | all | all | 11333 / 25094 | 45.1622 (0.3142) |
| en_ewt_deprel | 0.9 | all | conflict_prone | 8896 / 20010 | 44.4578 (0.3513) |
| en_ewt_deprel | 0.95 | all | all | 8854 / 25094 | 35.2833 (0.3017) |
| en_ewt_deprel | 0.95 | all | conflict_prone | 6810 / 20010 | 34.0330 (0.3350) |
| en_ewt_deprel | 0.99 | all | all | 4208 / 25094 | 16.7689 (0.2358) |
| en_ewt_deprel | 0.99 | all | conflict_prone | 3058 / 20010 | 15.2824 (0.2544) |

Conflict-prone means the original word has ≥2 training labels, or has training occurrences but none with the current gold label. Unseen words are not included in that subset.

## 2: lexical supply

| Dataset | τ | Group | Rows | Conflict mass mean [Q1, median, Q3] | Non-conflict mass mean [Q1, median, Q3] | Any conflict p≥.01 % | Any conflict p≥.05 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| conll2003 | 0.9 | all | 38438 | 0.0261 [0.0000, 0.0002, 0.0032] | 0.6073 [0.1467, 0.7746, 0.9855] | 7.1388 | 2.7941 |
| conll2003 | 0.9 | O | 36357 | 0.0094 [0.0000, 0.0002, 0.0021] | 0.6311 [0.2314, 0.8087, 0.9883] | 4.1065 | 1.1965 |
| conll2003 | 0.9 | entity | 2081 | 0.3168 [0.0161, 0.1859, 0.5713] | 0.1906 [0.0012, 0.0280, 0.2532] | 60.1153 | 30.7064 |
| conll2003 | 0.95 | all | 37262 | 0.0223 [0.0000, 0.0002, 0.0027] | 0.6116 [0.1532, 0.7868, 0.9869] | 6.2235 | 2.2999 |
| conll2003 | 0.95 | O | 35496 | 0.0085 [0.0000, 0.0001, 0.0019] | 0.6328 [0.2298, 0.8145, 0.9891] | 3.6765 | 1.0593 |
| conll2003 | 0.95 | entity | 1766 | 0.2998 [0.0112, 0.1699, 0.5171] | 0.1870 [0.0011, 0.0261, 0.2436] | 57.4179 | 27.2367 |
| conll2003 | 0.99 | all | 34100 | 0.0128 [0.0000, 0.0001, 0.0017] | 0.6208 [0.1604, 0.8099, 0.9898] | 4.4076 | 1.3372 |
| conll2003 | 0.99 | O | 33006 | 0.0070 [0.0000, 0.0001, 0.0014] | 0.6347 [0.2159, 0.8287, 0.9911] | 3.0358 | 0.8392 |
| conll2003 | 0.99 | entity | 1094 | 0.1900 [0.0064, 0.0747, 0.2499] | 0.2039 [0.0009, 0.0314, 0.2878] | 45.7952 | 16.3620 |
| en_ewt_deprel | 0.9 | all | 11333 | 0.0969 [0.0007, 0.0096, 0.0828] | 0.4489 [0.0021, 0.2492, 0.9744] | 34.0598 | 16.4564 |
| en_ewt_deprel | 0.95 | all | 8854 | 0.0800 [0.0004, 0.0055, 0.0538] | 0.4531 [0.0008, 0.2488, 0.9818] | 28.8344 | 13.1127 |
| en_ewt_deprel | 0.99 | all | 4208 | 0.0478 [0.0001, 0.0017, 0.0235] | 0.4614 [0.0000, 0.2428, 0.9914] | 17.7281 | 8.0086 |

Masses and thresholds use raw full-vocabulary MLM probabilities. CSVs also include retained vocabulary mass, row SEs and masses conditional on the allowed vocabulary.

## 2: price of the most plausible conflict

| Dataset | τ | Group | Rows | Mean price | Q1 / median / Q3 | Price ≤.1 % | Price ≤.2 % | No conflict % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conll2003 | 0.9 | all | 38438 | 2700.9991 | 0.1908 / 2.4131 / 28.8720 | 20.1233 | 25.4150 | 0.0000 |
| conll2003 | 0.9 | O | 36357 | 2841.2686 | 0.1861 / 2.5278 / 31.4622 | 20.3262 | 25.5687 | 0.0000 |
| conll2003 | 0.9 | entity | 2081 | 250.3621 | 0.2378 / 1.5181 / 6.4927 | 16.5786 | 22.7295 | 0.0000 |
| conll2003 | 0.95 | all | 37262 | 1474.8814 | 0.1746 / 2.2193 / 26.1041 | 20.7343 | 26.1553 | 0.0000 |
| conll2003 | 0.95 | O | 35496 | 1534.6558 | 0.1738 / 2.3157 / 28.5772 | 20.8136 | 26.1663 | 0.0000 |
| conll2003 | 0.95 | entity | 1766 | 273.4366 | 0.1862 / 1.1620 / 4.7400 | 19.1393 | 25.9343 | 0.0000 |
| conll2003 | 0.99 | all | 34100 | 1020.5359 | 0.1357 / 1.7233 / 21.3672 | 22.5367 | 28.3109 | 0.0000 |
| conll2003 | 0.99 | O | 33006 | 1040.7348 | 0.1409 / 1.7620 / 22.4636 | 22.3141 | 27.9858 | 0.0000 |
| conll2003 | 0.99 | entity | 1094 | 411.1367 | 0.0612 / 0.7184 / 3.8187 | 29.2505 | 38.1170 | 0.0000 |
| en_ewt_deprel | 0.9 | all | 11333 | 2238.3765 | 1.1831 / 7.2373 / 85.9088 | 5.8061 | 9.3356 | 0.0000 |
| en_ewt_deprel | 0.95 | all | 8854 | 2417.7265 | 1.0801 / 7.2505 / 110.5398 | 6.9686 | 10.7748 | 0.0000 |
| en_ewt_deprel | 0.99 | all | 4208 | 3073.6375 | 1.0426 / 8.1500 / 194.4864 | 8.9354 | 12.9990 | 0.0000 |

The price numerator uses probability assigned to gold y_c, rather than top-label confidence. Contexts without a conflict candidate remain in the denominator at infinite price.

## 3: layer-8 room

| Dataset | τ | Group | Draw scheme | Kind | Contexts | Predict y_c % | Predict word label % (different) | Original accuracy % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conll2003 | 0.9 | all | natural | conflict | 4082 | 22.3314 ± 1.6214 (SE 0.6054) | 64.7673 ± 1.0230 (SE 0.6928) | 94.0634 ± 0.2084 (SE 0.3497) |
| conll2003 | 0.9 | all | natural | nonconflict | 35832 | 99.3059 ± 0.0986 (SE 0.0351) | — | 99.3925 ± 0.0883 (SE 0.0383) |
| conll2003 | 0.9 | all | top | conflict | 38438 | 31.9663 ± 2.1348 (SE 0.1762) | 53.2783 ± 1.2803 (SE 0.1789) | 98.8544 ± 0.0794 (SE 0.0511) |
| conll2003 | 0.9 | all | top | nonconflict | 38438 | 98.4671 ± 0.1231 (SE 0.0451) | — | 98.8544 ± 0.0794 (SE 0.0511) |
| conll2003 | 0.9 | O | natural | conflict | 2739 | 30.0890 ± 2.7278 (SE 0.8170) | 52.3431 ± 1.9397 (SE 0.8820) | 97.3713 ± 0.1591 (SE 0.2924) |
| conll2003 | 0.9 | O | natural | nonconflict | 35069 | 99.3966 ± 0.1113 (SE 0.0321) | — | 99.4886 ± 0.0886 (SE 0.0354) |
| conll2003 | 0.9 | O | top | conflict | 36357 | 33.1833 ± 2.2888 (SE 0.1819) | 51.3830 ± 1.3935 (SE 0.1815) | 99.3261 ± 0.0941 (SE 0.0403) |
| conll2003 | 0.9 | O | top | nonconflict | 36357 | 99.1220 ± 0.1358 (SE 0.0330) | — | 99.3261 ± 0.0941 (SE 0.0403) |
| conll2003 | 0.9 | entity | natural | conflict | 1343 | 6.5099 ± 0.6759 (SE 0.5765) | 90.1059 ± 0.8742 (SE 0.6970) | 87.3170 ± 0.7750 (SE 0.8509) |
| conll2003 | 0.9 | entity | natural | nonconflict | 763 | 95.1376 ± 0.5148 (SE 0.7177) | — | 94.9760 ± 0.0757 (SE 0.7466) |
| conll2003 | 0.9 | entity | top | conflict | 2081 | 10.7053 ± 0.8178 (SE 0.5111) | 86.3901 ± 0.7391 (SE 0.5560) | 90.6135 ± 0.5733 (SE 0.6010) |
| conll2003 | 0.9 | entity | top | nonconflict | 2081 | 87.0255 ± 0.6449 (SE 0.5431) | — | 90.6135 ± 0.5733 (SE 0.6010) |
| conll2003 | 0.95 | all | natural | conflict | 3713 | 23.9555 ± 1.6853 (SE 0.6507) | 62.5741 ± 1.0702 (SE 0.7364) | 94.4878 ± 0.1892 (SE 0.3551) |
| conll2003 | 0.95 | all | natural | nonconflict | 34928 | 99.3496 ± 0.0982 (SE 0.0342) | — | 99.4551 ± 0.0774 (SE 0.0368) |
| conll2003 | 0.95 | all | top | conflict | 37262 | 32.5249 ± 2.1699 (SE 0.1795) | 52.5352 ± 1.3102 (SE 0.1812) | 98.9730 ± 0.0733 (SE 0.0493) |
| conll2003 | 0.95 | all | top | nonconflict | 37262 | 98.5887 ± 0.1269 (SE 0.0438) | — | 98.9730 ± 0.0733 (SE 0.0493) |
| conll2003 | 0.95 | O | natural | conflict | 2594 | 31.3276 ± 2.7633 (SE 0.8492) | 50.8969 ± 1.9379 (SE 0.9074) | 97.3786 ± 0.1335 (SE 0.3009) |
| conll2003 | 0.95 | O | natural | nonconflict | 34287 | 99.4121 ± 0.1094 (SE 0.0321) | — | 99.5295 ± 0.0783 (SE 0.0345) |
| conll2003 | 0.95 | O | top | conflict | 35496 | 33.5844 ± 2.3105 (SE 0.1845) | 50.8762 ± 1.4131 (SE 0.1833) | 99.3661 ± 0.0830 (SE 0.0397) |
| conll2003 | 0.95 | O | top | nonconflict | 35496 | 99.1204 ± 0.1362 (SE 0.0334) | — | 99.3661 ± 0.0830 (SE 0.0397) |
| conll2003 | 0.95 | entity | natural | conflict | 1119 | 6.8658 ± 0.8678 (SE 0.6429) | 89.6435 ± 0.9415 (SE 0.7809) | 87.7867 ± 0.7223 (SE 0.9194) |
| conll2003 | 0.95 | entity | natural | nonconflict | 641 | 96.0062 ± 0.5823 (SE 0.7113) | — | 95.4758 ± 0.1560 (SE 0.7670) |
| conll2003 | 0.95 | entity | top | conflict | 1766 | 11.2307 ± 0.9545 (SE 0.5648) | 85.8815 ± 0.8208 (SE 0.6133) | 91.0721 ± 0.5587 (SE 0.6392) |
| conll2003 | 0.95 | entity | top | nonconflict | 1766 | 87.9011 ± 0.6392 (SE 0.5778) | — | 91.0721 ± 0.5587 (SE 0.6392) |
| conll2003 | 0.99 | all | natural | conflict | 2907 | 28.3146 ± 2.1635 (SE 0.7761) | 56.5199 ± 1.4133 (SE 0.8493) | 95.8491 ± 0.1697 (SE 0.3510) |
| conll2003 | 0.99 | all | natural | nonconflict | 32359 | 99.4459 ± 0.0942 (SE 0.0320) | — | 99.5931 ± 0.0653 (SE 0.0331) |
| conll2003 | 0.99 | all | top | conflict | 34100 | 34.0420 ± 2.2729 (SE 0.1892) | 50.5937 ± 1.3607 (SE 0.1877) | 99.2747 ± 0.0633 (SE 0.0436) |
| conll2003 | 0.99 | all | top | nonconflict | 34100 | 98.8811 ± 0.1207 (SE 0.0406) | — | 99.2747 ± 0.0633 (SE 0.0436) |
| conll2003 | 0.99 | O | natural | conflict | 2308 | 33.1709 ± 2.9530 (SE 0.9138) | 48.6840 ± 2.0504 (SE 0.9593) | 97.7036 ± 0.1562 (SE 0.2975) |
| conll2003 | 0.99 | O | natural | nonconflict | 31942 | 99.4831 ± 0.0981 (SE 0.0306) | — | 99.6410 ± 0.0660 (SE 0.0311) |
| conll2003 | 0.99 | O | top | conflict | 33006 | 34.7341 ± 2.3705 (SE 0.1926) | 49.4947 ± 1.4294 (SE 0.1890) | 99.4890 ± 0.0667 (SE 0.0371) |
| conll2003 | 0.99 | O | top | nonconflict | 33006 | 99.1651 ± 0.1301 (SE 0.0334) | — | 99.4890 ± 0.0667 (SE 0.0371) |
| conll2003 | 0.99 | entity | natural | conflict | 599 | 9.6030 ± 1.0939 (SE 1.0269) | 86.7121 ± 1.2105 (SE 1.1891) | 88.7034 ± 0.8403 (SE 1.2179) |
| conll2003 | 0.99 | entity | natural | nonconflict | 417 | 96.5987 ± 0.2095 (SE 0.8272) | — | 95.9233 ± 0.2398 (SE 0.9428) |
| conll2003 | 0.99 | entity | top | conflict | 1094 | 13.1627 ± 1.0360 (SE 0.7821) | 83.7497 ± 0.9598 (SE 0.8321) | 92.8093 ± 0.4601 (SE 0.7436) |
| conll2003 | 0.99 | entity | top | nonconflict | 1094 | 90.3108 ± 0.3047 (SE 0.7188) | — | 92.8093 ± 0.4601 (SE 0.7436) |
| en_ewt_deprel | 0.9 | all | natural | conflict | 4912 | 44.7201 ± 2.0124 (SE 0.6259) | 33.2416 ± 0.5286 (SE 0.5400) | 95.3855 ± 0.1193 (SE 0.2758) |
| en_ewt_deprel | 0.9 | all | natural | nonconflict | 6413 | 98.7609 ± 0.0495 (SE 0.1221) | — | 98.3835 ± 0.0737 (SE 0.1429) |
| en_ewt_deprel | 0.9 | all | top | conflict | 11333 | 39.6472 ± 0.5697 (SE 0.3876) | 45.8190 ± 0.3698 (SE 0.3637) | 96.8323 ± 0.0551 (SE 0.1503) |
| en_ewt_deprel | 0.9 | all | top | nonconflict | 9255 | 95.6360 ± 0.1262 (SE 0.1578) | — | 96.1426 ± 0.0675 (SE 0.1826) |
| en_ewt_deprel | 0.95 | all | natural | conflict | 3779 | 45.1780 ± 2.3951 (SE 0.7007) | 32.4454 ± 0.5481 (SE 0.5888) | 96.8775 ± 0.0794 (SE 0.2573) |
| en_ewt_deprel | 0.95 | all | natural | nonconflict | 4955 | 99.3378 ± 0.0315 (SE 0.1014) | — | 99.0313 ± 0.0925 (SE 0.1244) |
| en_ewt_deprel | 0.95 | all | top | conflict | 8854 | 40.1137 ± 0.6427 (SE 0.4408) | 46.1097 ± 0.4075 (SE 0.4097) | 97.9105 ± 0.0000 (SE 0.1376) |
| en_ewt_deprel | 0.95 | all | top | nonconflict | 6988 | 96.5902 ± 0.1446 (SE 0.1544) | — | 97.3669 ± 0.0000 (SE 0.1732) |
| en_ewt_deprel | 0.99 | all | natural | conflict | 1844 | 44.0148 ± 3.4155 (SE 0.9437) | 32.8169 ± 0.8363 (SE 0.7804) | 98.4273 ± 0.1435 (SE 0.2563) |
| en_ewt_deprel | 0.99 | all | natural | nonconflict | 2313 | 99.7682 ± 0.0110 (SE 0.0841) | — | 99.6974 ± 0.1144 (SE 0.0943) |
| en_ewt_deprel | 0.99 | all | top | conflict | 4208 | 38.4242 ± 1.0706 (SE 0.6190) | 49.3716 ± 0.6452 (SE 0.5819) | 99.0811 ± 0.0960 (SE 0.1279) |
| en_ewt_deprel | 0.99 | all | top | nonconflict | 3057 | 97.5284 ± 0.1637 (SE 0.1837) | — | 98.7351 ± 0.1322 (SE 0.1757) |

Natural uses five independent draws with replacement; unclassified draws are logged and never relabelled or retried. Top uses up to three candidates per class. The original reference uses precisely the contexts supporting that row. Predict-word-label is undefined when word label equals y_c.

## 3: main paired gap, non-conflict minus conflict

| Dataset | τ | Group | Scheme | Paired contexts | Gap percentage points ± SD (SE) | Original accuracy % |
| --- | --- | --- | --- | --- | --- | --- |
| conll2003 | 0.9 | all | natural | 2675 | 67.8692 ± 2.4302 (SE 0.8471) | 97.6449 |
| conll2003 | 0.9 | all | top | 38438 | 66.5007 ± 2.0267 (SE 0.1775) | 98.8544 |
| conll2003 | 0.9 | O | natural | 2361 | 65.9831 ± 2.5300 (SE 0.9160) | 98.3199 |
| conll2003 | 0.9 | O | top | 36357 | 65.9387 ± 2.1567 (SE 0.1834) | 99.3261 |
| conll2003 | 0.9 | entity | natural | 314 | 82.0506 ± 1.7184 (SE 1.9828) | 92.5690 |
| conll2003 | 0.9 | entity | top | 2081 | 76.3201 ± 0.2452 (SE 0.6637) | 90.6135 |
| conll2003 | 0.95 | all | natural | 2516 | 66.9239 ± 2.3944 (SE 0.8815) | 97.8537 |
| conll2003 | 0.95 | all | top | 37262 | 66.0637 ± 2.0557 (SE 0.1808) | 98.9730 |
| conll2003 | 0.95 | O | natural | 2252 | 64.9793 ± 2.4986 (SE 0.9455) | 98.3866 |
| conll2003 | 0.95 | O | top | 35496 | 65.5360 ± 2.1777 (SE 0.1860) | 99.3661 |
| conll2003 | 0.95 | entity | natural | 264 | 83.5122 ± 1.5113 (SE 2.0900) | 93.3081 |
| conll2003 | 0.95 | entity | top | 1766 | 76.6704 ± 0.4096 (SE 0.7184) | 91.0721 |
| conll2003 | 0.99 | all | natural | 2176 | 64.5463 ± 2.5522 (SE 0.9638) | 98.2996 |
| conll2003 | 0.99 | all | top | 34100 | 64.8390 ± 2.1541 (SE 0.1905) | 99.2747 |
| conll2003 | 0.99 | O | natural | 2012 | 63.1157 ± 2.6822 (SE 1.0118) | 98.7078 |
| conll2003 | 0.99 | O | top | 33006 | 64.4310 ± 2.2414 (SE 0.1938) | 99.4890 |
| conll2003 | 0.99 | entity | natural | 164 | 82.0969 ± 1.1366 (SE 2.7314) | 93.2927 |
| conll2003 | 0.99 | entity | top | 1094 | 77.1481 ± 0.7313 (SE 0.9537) | 92.8093 |
| en_ewt_deprel | 0.9 | all | natural | 1062 | 44.9100 ± 0.2911 (SE 1.5047) | 95.7941 |
| en_ewt_deprel | 0.9 | all | top | 9255 | 49.1446 ± 0.0740 (SE 0.4423) | 96.1426 |
| en_ewt_deprel | 0.95 | all | natural | 685 | 38.4144 ± 0.6574 (SE 1.8242) | 96.3504 |
| en_ewt_deprel | 0.95 | all | top | 6988 | 48.4155 ± 0.1089 (SE 0.5070) | 97.3669 |
| en_ewt_deprel | 0.99 | all | natural | 247 | 27.3392 ± 0.6123 (SE 2.6489) | 97.8408 |
| en_ewt_deprel | 0.99 | all | top | 3057 | 49.5402 ± 0.2672 (SE 0.7402) | 98.7351 |

Each paired gap uses contexts with at least one insertion from both classes under that scheme; each context receives equal weight. Marginal class supports above can differ, so their raw difference is not the main gap.

Manual audit: [100 random most-plausible conflict insertions](tables/conflict/audit.csv) from τ=0.9. It includes original/inserted sentences, slot, gold label, full training-label counts/distributions, raw MLM probability and all seed predictions. If fewer than 100 eligible contexts exist, all are included.

[Implementation](IMPLEMENTATION_17.md) · [All CSVs](tables/conflict/). Raw arrays: `results/conflict/`.
