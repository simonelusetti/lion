# EWT dependency relations: context sensitivity

Predict each word’s base UD dependency relation from sentence text. No head positions are supplied or predicted; this is relation tagging, not dependency parsing.

Pinned `universal-dependencies/universal_dependencies`, `en_ewt`, revision `c3fed78df6b9f8cd27ed24cc7142e9eccf1e6703`. Strip colon subtypes, matching the existing certification deprel task. Include punctuation and root. No BIO/span metric. File hashes and source IDs: [tables](tables/deprel/).

Source train: 10531 sentences. Scorer fit/validation: 1920/683; classifier T1/T2: 4414/3514. Roles are document-disjoint. Native dev/test: 2001/2077 sentences, 25094 test words. Test not halved under the inherited six-hour budget rule.

Part 1 estimate/actual: 2.08/1.88 h. Part 2 estimate/actual: 6.7/5.0 min. Wall-clock: 1.96 h.

One all-token calibration group; both folds: gate_passed. Calibration γ=0.179837 comes from the scorer's own validation error rate. Measurement S is unchanged KL(original || filled), γ=.01 over 36 native relations. Each test word has four independent fills per mask and fold, saved once with token IDs. Baseline results are mean ± seed SD (three seeds), with word-row SEs. Folds/draws are not additional independent observations. * marks bins below fifty paired rows; these cells do not enter ratios.

## Raw test performance

| Model | Metric | Mean % | Seed SD |
| --- | --- | --- | --- |
| baseline | accuracy | 85.5915 | 0.1163 |
| baseline | macro_f1 | 65.7162 | 0.3491 |
| scorer | accuracy | 81.0273 | — |
| scorer | macro_f1 | 54.3007 | — |

Checkpoints use raw native-dev token macro-F1, with the same ten-epoch recipe as the previous baseline. No head annotations are input to the classifier and no span score is reported.

## Cuts and held-out reference

| Fold | Retained ρ | Gate | n | ΔCE | Absolute CE mean | SD | Absolute CE UCB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| g1 | 0.75 | gate_passed | 1054 | 0.0323 | 0.8385 | 1.0822 | 0.8934 |
| g2 | 0.65 | gate_passed | 1324 | 0.0279 | 0.7747 | 0.9755 | 0.8188 |

The unchanged gate rejects if the CE-increase lower bound exceeds ε=.05; ρ=1 passes by identity. Absolute CE UCB is mean + 1.645 SD/√n. It is an exact label-error reference only for fills conditioned on selection; these use the calibration-style procedure.

## All-draw sensitivity

| Model | Rows | S sel / rand | Flips % sel / rand | Raw ratio |
| --- | --- | --- | --- | --- |
| baseline | 25094 | 0.2114 ± 0.0012 (SE 0.0029) / 0.2402 ± 0.0013 (SE 0.0034) | 10.9367 ± 0.0690 (SE 0.1274) / 11.7090 ± 0.0558 (SE 0.1334) | 0.8801 ± 0.0023 (SE 0.0133) |
| scorer | 25094 | 0.1609 (SE 0.0021) / 0.1818 (SE 0.0025) | 11.1984 (SE 0.1378) / 12.0497 (SE 0.1460) | 0.8847 (SE 0.0128) |

Each statistic averages draws within fold and folds within word. The masks have the same dropped-word count for each row and fold.

## A: restoration

| Mask | Mean dropped | Mean changed | Restored % | Null swaps % |
| --- | --- | --- | --- | --- |
| selector | 6.1103 | 4.4174 | 27.7068 | 6.9897 |
| random | 6.1103 | 4.4260 | 27.5648 | 5.5885 |

Restoration is case-sensitive at word level; null swaps include zero-drop draws.

## A: non-null sensitivity

| Model | Paired rows | Excluded sel / rand | Non-null S sel / rand | Flips % sel / rand |
| --- | --- | --- | --- | --- |
| baseline | 24499 | 527 / 517 | 0.2265 ± 0.0013 (SE 0.0031) / 0.2544 ± 0.0014 (SE 0.0036) | 11.6937 ± 0.0728 (SE 0.1356) / 12.4069 ± 0.0484 (SE 0.1404) |
| scorer | 24499 | 527 / 517 | 0.1723 (SE 0.0023) / 0.1924 (SE 0.0026) | 11.9685 (SE 0.1467) / 12.7328 (SE 0.1535) |

Non-null estimates exclude all-null words, average eligible draws within each fold and use common support across masks.

## B: changed-word count

| Model | Bin | Rows | S sel / rand | Flips % sel / rand |
| --- | --- | --- | --- | --- |
| baseline | 0 | 2010 | 0.0000 ± 0.0000 (SE 0.0000) / 0.0000 ± 0.0000 (SE 0.0000) | 0.0000 ± 0.0000 (SE 0.0000) / 0.0000 ± 0.0000 (SE 0.0000) |
| baseline | 1 | 5276 | 0.1818 ± 0.0024 (SE 0.0074) / 0.1900 ± 0.0017 (SE 0.0075) | 9.8975 ± 0.0296 (SE 0.3083) / 10.4517 ± 0.0819 (SE 0.3192) |
| baseline | 2 | 7268 | 0.2064 ± 0.0013 (SE 0.0069) / 0.2365 ± 0.0011 (SE 0.0076) | 11.4387 ± 0.2452 (SE 0.2858) / 12.2516 ± 0.2631 (SE 0.2957) |
| baseline | 3 | 7189 | 0.2017 ± 0.0015 (SE 0.0073) / 0.2384 ± 0.0013 (SE 0.0083) | 10.7338 ± 0.2268 (SE 0.2882) / 11.5931 ± 0.1136 (SE 0.3007) |
| baseline | 4 | 6490 | 0.2115 ± 0.0009 (SE 0.0082) / 0.2499 ± 0.0014 (SE 0.0090) | 11.0595 ± 0.0434 (SE 0.3182) / 11.9108 ± 0.4250 (SE 0.3248) |
| baseline | 5–9 | 12230 | 0.2377 ± 0.0013 (SE 0.0057) / 0.2639 ± 0.0013 (SE 0.0061) | 12.1535 ± 0.1443 (SE 0.2210) / 12.7468 ± 0.1783 (SE 0.2238) |
| baseline | 10+ | 3711 | 0.2990 ± 0.0032 (SE 0.0118) / 0.3156 ± 0.0046 (SE 0.0127) | 14.2239 ± 0.2100 (SE 0.4367) / 14.4765 ± 0.2384 (SE 0.4433) |
| scorer | 0 | 2010 | 0.0000 (SE 0.0000) / 0.0000 (SE 0.0000) | 0.0000 (SE 0.0000) / 0.0000 (SE 0.0000) |
| scorer | 1 | 5276 | 0.1333 (SE 0.0050) / 0.1406 (SE 0.0052) | 9.8481 (SE 0.3409) / 10.6228 (SE 0.3565) |
| scorer | 2 | 7268 | 0.1538 (SE 0.0048) / 0.1756 (SE 0.0055) | 11.2302 (SE 0.3129) / 12.2827 (SE 0.3292) |
| scorer | 3 | 7189 | 0.1557 (SE 0.0053) / 0.1824 (SE 0.0062) | 11.0504 (SE 0.3254) / 12.2282 (SE 0.3420) |
| scorer | 4 | 6490 | 0.1673 (SE 0.0062) / 0.1997 (SE 0.0068) | 11.4959 (SE 0.3541) / 12.6932 (SE 0.3686) |
| scorer | 5–9 | 12230 | 0.1823 (SE 0.0042) / 0.1983 (SE 0.0044) | 12.4516 (SE 0.2426) / 13.3412 (SE 0.2490) |
| scorer | 10+ | 3711 | 0.2282 (SE 0.0087) / 0.2372 (SE 0.0092) | 15.0611 (SE 0.4828) / 14.7534 (SE 0.4788) |

Both masks use the same words within each bin. Familiarity and ambiguity use exact case-sensitive words and base dependency labels in T1∪T2 only; entity-span familiarity is inapplicable.

## B: nearest-change distance

| Model | Bin | Rows | S sel / rand | Flips % sel / rand |
| --- | --- | --- | --- | --- |
| baseline | 1 | 11418 | 0.4222 ± 0.0018 (SE 0.0084) / 0.4663 ± 0.0040 (SE 0.0090) | 17.1060 ± 0.0872 (SE 0.2774) / 17.9798 ± 0.0691 (SE 0.2802) |
| baseline | 2 | 5926 | 0.1667 ± 0.0006 (SE 0.0054) / 0.1798 ± 0.0025 (SE 0.0059) | 11.5213 ± 0.1494 (SE 0.3191) / 11.8841 ± 0.1997 (SE 0.3204) |
| baseline | 3 | 2892 | 0.0903 ± 0.0014 (SE 0.0042) / 0.0999 ± 0.0022 (SE 0.0050) | 7.9467 ± 0.2957 (SE 0.3702) / 8.2037 ± 0.2449 (SE 0.3715) |
| baseline | 4–5 | 3586 | 0.0641 ± 0.0004 (SE 0.0040) / 0.0737 ± 0.0003 (SE 0.0038) | 6.2159 ± 0.0790 (SE 0.2969) / 6.8348 ± 0.1619 (SE 0.3063) |
| baseline | 6–10 | 2513 | 0.0350 ± 0.0015 (SE 0.0019) / 0.0387 ± 0.0011 (SE 0.0024) | 4.9288 ± 0.3086 (SE 0.3082) / 4.9095 ± 0.1596 (SE 0.2993) |
| baseline | >10 | 443 | 0.0162 ± 0.0003 (SE 0.0020) / 0.0220 ± 0.0017 (SE 0.0043) | 3.0443 ± 1.1179 (SE 0.5677) / 3.4330 ± 0.5820 (SE 0.5679) |
| scorer | 1 | 11418 | 0.3140 (SE 0.0061) / 0.3471 (SE 0.0066) | 16.9911 (SE 0.2917) / 17.9902 (SE 0.2967) |
| scorer | 2 | 5926 | 0.1331 (SE 0.0039) / 0.1385 (SE 0.0043) | 11.7153 (SE 0.3552) / 12.0823 (SE 0.3565) |
| scorer | 3 | 2892 | 0.0772 (SE 0.0031) / 0.0821 (SE 0.0037) | 9.0811 (SE 0.4568) / 9.3664 (SE 0.4644) |
| scorer | 4–5 | 3586 | 0.0540 (SE 0.0028) / 0.0602 (SE 0.0029) | 7.3666 (SE 0.3743) / 7.7512 (SE 0.3825) |
| scorer | 6–10 | 2513 | 0.0304 (SE 0.0014) / 0.0318 (SE 0.0016) | 5.8745 (SE 0.4083) / 5.1864 (SE 0.3778) |
| scorer | >10 | 443 | 0.0192 (SE 0.0027) / 0.0203 (SE 0.0029) | 4.1196 (SE 0.8537) / 3.4236 (SE 0.7781) |

Both masks use the same words within each bin. Familiarity and ambiguity use exact case-sensitive words and base dependency labels in T1∪T2 only; entity-span familiarity is inapplicable.

## C: word and label familiarity

| Model | Bin | Rows | S sel / rand | Flips % sel / rand | Original error % |
| --- | --- | --- | --- | --- | --- |
| baseline | unseen_word | 2696 | 0.3215 ± 0.0039 (SE 0.0091) / 0.3429 ± 0.0053 (SE 0.0103) | 22.6462 ± 0.2289 (SE 0.4865) / 23.2103 ± 0.4089 (SE 0.4981) | 33.3828 ± 0.6072 (SE 0.8472) |
| baseline | seen_other_label | 2138 | 0.4268 ± 0.0032 (SE 0.0132) / 0.4876 ± 0.0072 (SE 0.0152) | 24.3335 ± 0.1168 (SE 0.5619) / 25.8166 ± 0.4105 (SE 0.5848) | 39.4917 ± 0.6767 (SE 1.0015) |
| baseline | seen_same_label | 20260 | 0.1741 ± 0.0010 (SE 0.0031) / 0.2005 ± 0.0007 (SE 0.0035) | 7.9648 ± 0.0970 (SE 0.1222) / 8.6897 ± 0.1086 (SE 0.1293) | 9.2366 ± 0.0812 (SE 0.1901) |
| scorer | unseen_word | 2696 | 0.1988 (SE 0.0058) / 0.2156 (SE 0.0064) | 21.4021 (SE 0.5286) / 22.3201 (SE 0.5495) | 40.7641 (SE 0.9466) |
| scorer | seen_other_label | 2138 | 0.2941 (SE 0.0092) / 0.3288 (SE 0.0103) | 23.7547 (SE 0.6099) / 25.0760 (SE 0.6297) | 47.8017 (SE 1.0806) |
| scorer | seen_same_label | 20260 | 0.1417 (SE 0.0023) / 0.1618 (SE 0.0027) | 8.5155 (SE 0.1348) / 9.3084 (SE 0.1450) | 13.0306 (SE 0.2365) |

Both masks use the same words within each bin. Familiarity and ambiguity use exact case-sensitive words and base dependency labels in T1∪T2 only; entity-span familiarity is inapplicable.

## D: training word-form ambiguity

| Model | Bin | Rows | S sel / rand | Flips % sel / rand | Share % |
| --- | --- | --- | --- | --- | --- |
| baseline | unseen_word | 2696 | 0.3215 ± 0.0039 (SE 0.0091) / 0.3429 ± 0.0053 (SE 0.0103) | 22.6462 ± 0.2289 (SE 0.4865) / 23.2103 ± 0.4089 (SE 0.4981) | 10.7436 |
| baseline | one_type | 3996 | 0.1376 ± 0.0026 (SE 0.0059) / 0.1501 ± 0.0020 (SE 0.0067) | 7.0247 ± 0.0720 (SE 0.2698) / 7.4001 ± 0.0376 (SE 0.2806) | 15.9241 |
| baseline | two_or_more_types | 18402 | 0.2114 ± 0.0009 (SE 0.0035) / 0.2448 ± 0.0006 (SE 0.0041) | 10.0707 ± 0.1104 (SE 0.1427) / 10.9596 ± 0.0830 (SE 0.1508) | 73.3323 |
| scorer | unseen_word | 2696 | 0.1988 (SE 0.0058) / 0.2156 (SE 0.0064) | 21.4021 (SE 0.5286) / 22.3201 (SE 0.5495) | 10.7436 |
| scorer | one_type | 3996 | 0.1125 (SE 0.0044) / 0.1237 (SE 0.0051) | 7.5732 (SE 0.2969) / 7.9486 (SE 0.3075) | 15.9241 |
| scorer | two_or_more_types | 18402 | 0.1658 (SE 0.0026) / 0.1895 (SE 0.0030) | 10.4907 (SE 0.1554) / 11.4356 (SE 0.1663) | 73.3323 |

Both masks use the same words within each bin. Familiarity and ambiguity use exact case-sensitive words and base dependency labels in T1∪T2 only; entity-span familiarity is inapplicable.

## B: dropped-position distances

| Mask | Mean distance | Median | Within two words % |
| --- | --- | --- | --- |
| selector | 10.8737 | 8.0000 | 16.6520 |
| random | 10.6710 | 8.0000 | 17.2661 |

Every dropped position is weighted equally, pooling both folds.

## B: size/distance-matched selector/random ratio

| Model | Ratio ± SD (row SE) | Supported cells | Retained mass % |
| --- | --- | --- | --- |
| baseline | 0.9300 ± 0.0045 (SE 0.0186) | 32 | 98.4158 |
| scorer | 0.9718 (SE 0.0174) | 32 | 98.4158 |

Compute ratios within common changed-count × nearest-distance cells, then weight by selector cell frequencies. Omit cells with fewer than fifty rows or zero random S; renormalize remaining weights. Row SE fixes cell weights and accounts for repeated words across cells.

## D: baseline flip concentration

| Ranking | Top % | Rows | Selector flip share % | Random flip share % |
| --- | --- | --- | --- | --- |
| selector | 1 | 251 | 5.5932 ± 0.0564 (SE 0.3323) | 3.1611 ± 0.0990 (SE 0.2394) |
| selector | 5 | 1255 | 25.1714 ± 0.2419 (SE 0.5952) | 15.2254 ± 0.3028 (SE 0.4785) |
| selector | 10 | 2510 | 44.6304 ± 0.6130 (SE 0.6405) | 28.6536 ± 0.4802 (SE 0.5896) |
| random | 1 | 251 | 2.9395 ± 0.0720 (SE 0.2277) | 6.0499 ± 0.1264 (SE 0.3575) |
| random | 5 | 1255 | 14.2768 ± 0.3244 (SE 0.4736) | 25.2474 ± 0.0628 (SE 0.5978) |
| random | 10 | 2510 | 28.2081 ± 0.6837 (SE 0.5963) | 44.4420 ± 0.3465 (SE 0.6349) |

Rank within seed by draw-averaged S. Each ranking evaluates both masks on the same selected words; row SE conditions on the ranking.

## D: frequent forms in top 1%

| Rank | Selector ranking | Random ranking |
| --- | --- | --- |
| 1 | to (20) | to (15) |
| 2 | like (8) | that (6) |
| 3 | have (7) | have (5) |
| 4 | that (5) | be (4) |
| 5 | - (5) | Legal (3) |
| 6 | has (3) | this (3) |
| 7 | it (3) | is (3) |
| 8 | in (3) | in (3) |
| 9 | are (3) | one (3) |
| 10 | of (2) | had (3) |
| 11 | all (2) | are (3) |
| 12 | had (2) | system (2) |
| 13 | do (2) | i (2) |
| 14 | this (2) | for (2) |
| 15 | service (2) | better (2) |
| 16 | was (2) | food (2) |
| 17 | fixed (2) | has (2) |
| 18 | time (2) | on (2) |
| 19 | you (2) | like (2) |
| 20 | call (2) | did (2) |

Counts use seed-averaged S, once per focal-word row. Seed-specific lists and sensitivity by gold dependency relation are in CSVs.

Differences from NER: relation labels and token metrics; one calibration group; document-disjoint source roles; native EWT dev/test sizes; word-label familiarity replaces entity-span familiarity. The scorer, selector, fill and sensitivity algorithms are reused.

[Implementation](IMPLEMENTATION_12.md) · [CSV tables](tables/deprel/) · [Previous Few-NERD results](old/11_fewnerd_replication/RESULTS_11.md).
