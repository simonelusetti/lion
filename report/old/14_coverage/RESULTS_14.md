# Is the layer gap a coverage effect?

Frozen bert-base-cased; heads read the mean WordPiece vector at one layer or the learned scalar mixture, with the existing head recipe (MLP 768→256, AdamW, 64-sentence batches, raw-dev token macro-F1 checkpoint). No scorer, selector, fills or certificates. Native dev selects checkpoints and native test evaluates every run.

Estimate 0.61 h; under six hours, so all five fractions were run. Actual: prepare 0.2 min, estimate 0.1 min, train 27.4 min, analyse 0.0 min; wall-clock 0.46 h.

Training frequency = occurrences of the test word’s exact case-sensitive form in the head’s training set. Accuracy is token accuracy over native tags. Gaps are computed per test row (correct − correct), then averaged. Values: mean ± seed SD (row SE) over three seeds, in percentage points. "Monotonically decreasing" means non-increasing along the stated order.

## Part A: gaps by training frequency (layer-run heads, trained on T1∪T2)

### conll2003 (best single layer in the layer run: layer8)

| Group | Training frequency | Rows | Acc. layer 12 % | Acc. layer8 % | Acc. mixture % | layer8 − layer 12 | Mixture − layer 12 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gold_O | 0 | 3445 | 96.20 ± 0.38 | 96.06 ± 0.53 | 95.76 ± 1.03 | -0.14 ± 0.62 (0.18) | -0.44 ± 0.80 (0.17) |
| gold_O | 1–2 | 2043 | 98.42 ± 0.10 | 98.78 ± 0.25 | 98.56 ± 0.37 | +0.36 ± 0.25 (0.17) | +0.15 ± 0.35 (0.17) |
| gold_O | 3–9 | 3170 | 98.70 ± 0.19 | 98.76 ± 0.16 | 98.70 ± 0.28 | +0.06 ± 0.06 (0.12) | -0.00 ± 0.11 (0.12) |
| gold_O | 10–99 | 9862 | 99.19 ± 0.06 | 99.24 ± 0.05 | 99.17 ± 0.07 | +0.05 ± 0.07 (0.04) | -0.02 ± 0.01 (0.04) |
| gold_O | 100+ | 19803 | 99.90 ± 0.01 | 99.91 ± 0.03 | 99.90 ± 0.02 | +0.01 ± 0.02 (0.02) | +0.00 ± 0.02 (0.02) |
| gold_entity | 0 | 2708 | 84.18 ± 0.94 | 85.23 ± 0.42 | 85.73 ± 0.65 | +1.05 ± 0.72 (0.43) | +1.55 ± 0.59 (0.41) |
| gold_entity | 1–2 | 1093 | 87.74 ± 0.66 | 88.44 ± 0.14 | 87.56 ± 0.42 | +0.70 ± 0.78 (0.56) | -0.18 ± 0.60 (0.56) |
| gold_entity | 3–9 | 1496 | 91.56 ± 0.66 | 91.84 ± 0.29 | 92.13 ± 0.21 | +0.29 ± 0.54 (0.48) | +0.58 ± 0.76 (0.47) |
| gold_entity | 10–99 | 2539 | 92.35 ± 0.06 | 93.40 ± 0.53 | 93.80 ± 0.42 | +1.05 ± 0.59 (0.33) | +1.46 ± 0.48 (0.33) |
| gold_entity | 100+ | 276 | 82.85 ± 1.11 | 87.20 ± 1.05 | 87.44 ± 0.75 | +4.35 ± 0.36 (1.43) | +4.59 ± 1.86 (1.28) |

**conll2003: gap decreases monotonically with training frequency (0 → 100+)?** gold_O, best minus layer12: no; gold_O, mixture minus layer12: no; gold_entity, best minus layer12: no; gold_entity, mixture minus layer12: no.

### en_ewt_deprel (best single layer in the layer run: layer8)

| Group | Training frequency | Rows | Acc. layer 12 % | Acc. layer8 % | Acc. mixture % | layer8 − layer 12 | Mixture − layer 12 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 0 | 2695 | 66.84 ± 0.30 | 71.65 ± 0.38 | 71.63 ± 0.98 | +4.81 ± 0.44 (0.68) | +4.79 ± 1.06 (0.64) |
| all | 1–2 | 1390 | 72.64 ± 0.41 | 77.89 ± 0.11 | 76.67 ± 0.61 | +5.25 ± 0.31 (0.88) | +4.03 ± 0.36 (0.87) |
| all | 3–9 | 2320 | 74.83 ± 0.63 | 79.44 ± 0.27 | 78.95 ± 0.11 | +4.61 ± 0.49 (0.67) | +4.12 ± 0.52 (0.64) |
| all | 10–99 | 5522 | 80.19 ± 0.17 | 84.75 ± 0.16 | 85.22 ± 0.34 | +4.56 ± 0.01 (0.39) | +5.03 ± 0.45 (0.37) |
| all | 100+ | 13167 | 95.05 ± 0.02 | 96.36 ± 0.07 | 96.48 ± 0.11 | +1.31 ± 0.06 (0.13) | +1.44 ± 0.11 (0.13) |

**en_ewt_deprel: gap decreases monotonically with training frequency (0 → 100+)?** all, best minus layer12: no; all, mixture minus layer12: no.

Per-bin row counts and each head’s error on words seen in training only with other labels: [frequency_crosstab.csv](tables/coverage/frequency_crosstab.csv).

## Part B: gaps by training fraction

Nested random sentence subsets of the full native training set (seed 0). Every run gets at least as many optimizer steps as ten epochs on the full set: the epoch cap is scaled up for small fractions and the best dev checkpoint is kept. The best single layer is chosen by mean dev macro-F1 at each fraction.

### conll2003

| Fraction | Sentences | Best layer | Best-layer selected step / steps | L12 span F1 | Best span F1 | Mixture span F1 | Best − L12 span F1 | Mixture − L12 span F1 | L12 acc. | Best acc. | Mixture acc. | Best − L12 acc. | Mixture − L12 acc. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0625 | 878 | layer11 | 327 / 2212 | 77.29 ± 0.50 | 79.60 ± 0.52 | 79.34 ± 0.20 | +2.31 ± 0.65 | +2.06 ± 0.58 | 96.18 ± 0.03 | 96.53 ± 0.04 | 96.54 ± 0.04 | +0.35 ± 0.04 | +0.36 ± 0.07 |
| 0.125 | 1756 | layer9 | 2007 / 2212 | 80.15 ± 0.20 | 81.73 ± 0.16 | 80.89 ± 0.15 | +1.58 ± 0.17 | +0.74 ± 0.32 | 96.66 ± 0.02 | 96.99 ± 0.03 | 96.85 ± 0.00 | +0.33 ± 0.04 | +0.19 ± 0.02 |
| 0.25 | 3511 | layer9 | 1650 / 2200 | 82.30 ± 0.65 | 83.72 ± 0.28 | 82.48 ± 0.26 | +1.42 ± 0.46 | +0.18 ± 0.65 | 97.04 ± 0.05 | 97.27 ± 0.02 | 97.04 ± 0.05 | +0.23 ± 0.06 | +0.00 ± 0.06 |
| 0.5 | 7021 | layer8 | 2127 / 2200 | 84.20 ± 0.46 | 85.52 ± 0.28 | 84.87 ± 0.40 | +1.33 ± 0.72 | +0.68 ± 0.85 | 97.30 ± 0.06 | 97.53 ± 0.03 | 97.42 ± 0.03 | +0.23 ± 0.09 | +0.12 ± 0.09 |
| 1 | 14041 | layer8 | 2053 / 2200 | 85.35 ± 0.27 | 86.30 ± 0.29 | 85.87 ± 0.35 | +0.96 ± 0.51 | +0.53 ± 0.19 | 97.52 ± 0.01 | 97.70 ± 0.01 | 97.63 ± 0.07 | +0.18 ± 0.01 | +0.12 ± 0.08 |

**conll2003: gap decreases monotonically with training size (smallest → full)?** best minus layer12, span_f1: yes; best minus layer12, accuracy: yes; mixture minus layer12, span_f1: no; mixture minus layer12, accuracy: no.

Test span f1 % per head and fraction (all metrics and selected steps per head in [curve_performance.csv](tables/coverage/curve_performance.csv)):

| Head | 0.0625 | 0.125 | 0.25 | 0.5 | 1 |
| --- | --- | --- | --- | --- | --- |
| layer0 | 48.36 ± 0.74 | 51.66 ± 1.20 | 54.19 ± 0.12 | 57.36 ± 0.84 | 61.71 ± 0.86 |
| layer1 | 55.35 ± 0.82 | 61.11 ± 0.34 | 64.90 ± 0.74 | 67.66 ± 0.46 | 71.44 ± 0.14 |
| layer2 | 62.98 ± 0.67 | 67.96 ± 0.66 | 71.27 ± 0.50 | 74.33 ± 0.78 | 77.15 ± 0.20 |
| layer3 | 67.87 ± 0.55 | 71.52 ± 0.08 | 73.92 ± 0.47 | 76.93 ± 0.25 | 78.76 ± 0.73 |
| layer4 | 74.64 ± 0.20 | 76.82 ± 0.15 | 78.55 ± 0.16 | 81.36 ± 0.28 | 83.47 ± 0.29 |
| layer5 | 76.67 ± 0.32 | 78.67 ± 0.18 | 80.32 ± 0.04 | 82.68 ± 0.23 | 83.70 ± 0.22 |
| layer6 | 79.22 ± 0.41 | 80.90 ± 0.12 | 82.37 ± 0.65 | 83.64 ± 0.36 | 85.49 ± 0.32 |
| layer7 | 79.53 ± 0.40 | 81.09 ± 0.04 | 82.52 ± 0.34 | 84.44 ± 0.09 | 85.62 ± 0.31 |
| layer8 | 80.47 ± 0.44 | 81.59 ± 0.21 | 83.49 ± 0.16 | 85.52 ± 0.28 | 86.30 ± 0.29 |
| layer9 | 79.67 ± 0.28 | 81.73 ± 0.16 | 83.72 ± 0.28 | 85.16 ± 0.35 | 86.51 ± 0.52 |
| layer10 | 79.57 ± 0.34 | 81.90 ± 0.11 | 83.44 ± 0.28 | 85.07 ± 0.28 | 86.33 ± 0.18 |
| layer11 | 79.60 ± 0.52 | 81.72 ± 0.38 | 83.55 ± 0.15 | 85.08 ± 0.32 | 86.47 ± 0.16 |
| layer12 | 77.29 ± 0.50 | 80.15 ± 0.20 | 82.30 ± 0.65 | 84.20 ± 0.46 | 85.35 ± 0.27 |
| mixture | 79.34 ± 0.20 | 80.89 ± 0.15 | 82.48 ± 0.26 | 84.87 ± 0.40 | 85.87 ± 0.35 |

Learned mixture weights (softmax), mean over seeds:

| Fraction | Scale | w0 | w1 | w2 | w3 | w4 | w5 | w6 | w7 | w8 | w9 | w10 | w11 | w12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0625 | 1.156 | 0.063 | 0.060 | 0.063 | 0.063 | 0.073 | 0.083 | 0.088 | 0.091 | 0.095 | 0.093 | 0.088 | 0.078 | 0.061 |
| 0.125 | 1.251 | 0.057 | 0.054 | 0.056 | 0.057 | 0.067 | 0.082 | 0.098 | 0.100 | 0.103 | 0.100 | 0.094 | 0.078 | 0.054 |
| 0.25 | 1.347 | 0.051 | 0.048 | 0.051 | 0.052 | 0.061 | 0.075 | 0.104 | 0.107 | 0.111 | 0.108 | 0.103 | 0.081 | 0.048 |
| 0.5 | 1.364 | 0.051 | 0.049 | 0.051 | 0.051 | 0.057 | 0.069 | 0.102 | 0.108 | 0.114 | 0.110 | 0.107 | 0.084 | 0.048 |
| 1 | 1.315 | 0.053 | 0.050 | 0.051 | 0.051 | 0.059 | 0.067 | 0.100 | 0.102 | 0.111 | 0.108 | 0.111 | 0.086 | 0.051 |

### en_ewt_deprel

| Fraction | Sentences | Best layer | Best-layer selected step / steps | L12 acc. | Best acc. | Mixture acc. | Best − L12 acc. | Mixture − L12 acc. | L12 macro-F1 | Best macro-F1 | Mixture macro-F1 | Best − L12 macro-F1 | Mixture − L12 macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0625 | 784 | layer8 | 1205 / 1963 | 79.18 ± 0.09 | 83.11 ± 0.25 | 83.00 ± 0.11 | +3.92 ± 0.23 | +3.82 ± 0.19 | 56.04 ± 0.14 | 61.15 ± 0.33 | 60.76 ± 0.35 | +5.12 ± 0.32 | +4.72 ± 0.22 |
| 0.125 | 1568 | layer9 | 1167 / 1975 | 81.85 ± 0.05 | 84.76 ± 0.20 | 85.21 ± 0.07 | +2.91 ± 0.15 | +3.36 ± 0.07 | 60.92 ± 0.28 | 64.59 ± 0.59 | 64.63 ± 0.39 | +3.67 ± 0.35 | +3.70 ± 0.21 |
| 0.25 | 3136 | layer8 | 964 / 1960 | 84.00 ± 0.16 | 86.85 ± 0.08 | 86.88 ± 0.18 | +2.85 ± 0.18 | +2.88 ± 0.08 | 64.06 ± 0.22 | 68.03 ± 0.30 | 67.07 ± 0.38 | +3.97 ± 0.52 | +3.01 ± 0.18 |
| 0.5 | 6272 | layer8 | 1437 / 1960 | 85.55 ± 0.05 | 88.20 ± 0.11 | 88.37 ± 0.17 | +2.65 ± 0.13 | +2.81 ± 0.16 | 66.49 ± 0.08 | 69.91 ± 0.73 | 69.61 ± 0.32 | +3.42 ± 0.75 | +3.12 ± 0.38 |
| 1 | 12544 | layer9 | 1633 / 1960 | 86.55 ± 0.11 | 89.35 ± 0.16 | 89.44 ± 0.06 | +2.80 ± 0.15 | +2.89 ± 0.04 | 67.62 ± 0.48 | 71.50 ± 0.30 | 71.43 ± 0.33 | +3.89 ± 0.47 | +3.81 ± 0.19 |

**en_ewt_deprel: gap decreases monotonically with training size (smallest → full)?** best minus layer12, accuracy: no; best minus layer12, macro_f1: no; mixture minus layer12, accuracy: no; mixture minus layer12, macro_f1: no.

Test macro f1 % per head and fraction (all metrics and selected steps per head in [curve_performance.csv](tables/coverage/curve_performance.csv)):

| Head | 0.0625 | 0.125 | 0.25 | 0.5 | 1 |
| --- | --- | --- | --- | --- | --- |
| layer0 | 37.14 ± 0.49 | 39.84 ± 0.29 | 40.67 ± 0.37 | 41.99 ± 0.29 | 42.51 ± 0.09 |
| layer1 | 44.17 ± 0.33 | 47.52 ± 0.40 | 49.32 ± 0.32 | 50.62 ± 0.63 | 52.20 ± 0.93 |
| layer2 | 52.35 ± 0.33 | 55.61 ± 0.19 | 57.95 ± 0.34 | 59.33 ± 0.27 | 60.71 ± 0.09 |
| layer3 | 53.57 ± 0.31 | 57.07 ± 0.38 | 59.10 ± 0.05 | 60.81 ± 0.07 | 62.00 ± 0.34 |
| layer4 | 56.05 ± 0.07 | 59.75 ± 0.48 | 61.38 ± 0.29 | 62.78 ± 0.50 | 64.34 ± 0.58 |
| layer5 | 58.18 ± 0.26 | 61.73 ± 0.09 | 63.89 ± 0.28 | 65.49 ± 0.08 | 67.42 ± 0.21 |
| layer6 | 59.79 ± 0.12 | 63.90 ± 0.26 | 66.05 ± 0.28 | 67.55 ± 0.09 | 69.33 ± 0.35 |
| layer7 | 60.09 ± 0.08 | 63.71 ± 0.03 | 66.30 ± 0.19 | 67.81 ± 0.10 | 69.67 ± 0.14 |
| layer8 | 61.15 ± 0.33 | 64.75 ± 0.48 | 68.03 ± 0.30 | 69.91 ± 0.73 | 71.24 ± 0.36 |
| layer9 | 59.92 ± 0.12 | 64.59 ± 0.59 | 67.50 ± 0.12 | 70.40 ± 0.56 | 71.50 ± 0.30 |
| layer10 | 58.94 ± 0.05 | 63.37 ± 0.23 | 66.62 ± 0.23 | 68.46 ± 0.31 | 70.56 ± 0.34 |
| layer11 | 58.12 ± 0.34 | 62.01 ± 0.51 | 65.95 ± 0.60 | 68.23 ± 0.30 | 69.62 ± 0.46 |
| layer12 | 56.04 ± 0.14 | 60.92 ± 0.28 | 64.06 ± 0.22 | 66.49 ± 0.08 | 67.62 ± 0.48 |
| mixture | 60.76 ± 0.35 | 64.63 ± 0.39 | 67.07 ± 0.38 | 69.61 ± 0.32 | 71.43 ± 0.33 |

Learned mixture weights (softmax), mean over seeds:

| Fraction | Scale | w0 | w1 | w2 | w3 | w4 | w5 | w6 | w7 | w8 | w9 | w10 | w11 | w12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0625 | 1.419 | 0.044 | 0.049 | 0.052 | 0.054 | 0.062 | 0.083 | 0.104 | 0.111 | 0.121 | 0.118 | 0.095 | 0.062 | 0.046 |
| 0.125 | 1.483 | 0.041 | 0.044 | 0.047 | 0.047 | 0.054 | 0.078 | 0.109 | 0.113 | 0.132 | 0.126 | 0.102 | 0.063 | 0.044 |
| 0.25 | 1.329 | 0.044 | 0.047 | 0.048 | 0.048 | 0.055 | 0.073 | 0.103 | 0.109 | 0.124 | 0.122 | 0.105 | 0.073 | 0.051 |
| 0.5 | 1.395 | 0.037 | 0.038 | 0.040 | 0.040 | 0.046 | 0.065 | 0.113 | 0.116 | 0.144 | 0.139 | 0.109 | 0.068 | 0.046 |
| 1 | 1.517 | 0.028 | 0.029 | 0.032 | 0.031 | 0.036 | 0.057 | 0.126 | 0.114 | 0.171 | 0.163 | 0.114 | 0.061 | 0.038 |

Subset sizes: conll2003 0.0625: 878 sentences / 12672 words; conll2003 0.125: 1756 sentences / 25209 words; conll2003 0.25: 3511 sentences / 50646 words; conll2003 0.5: 7021 sentences / 101296 words; conll2003 1.0: 14041 sentences / 203621 words; conll2003 T1∪T2 (part A): 10531 sentences / 152068 words; en_ewt_deprel 0.0625: 784 sentences / 12899 words; en_ewt_deprel 0.125: 1568 sentences / 25831 words; en_ewt_deprel 0.25: 3136 sentences / 51375 words; en_ewt_deprel 0.5: 6272 sentences / 102273 words; en_ewt_deprel 1.0: 12544 sentences / 204578 words; en_ewt_deprel T1∪T2 (part A): 7928 sentences / 129066 words.

Per-fraction frequency breakdowns (frequencies counted in that subset) are in [frequency_gaps.csv](tables/coverage/frequency_gaps.csv) and [frequency_accuracy.csv](tables/coverage/frequency_accuracy.csv).

[Implementation](IMPLEMENTATION_14.md) · [CSV tables](tables/coverage/) · [Previous: layer sensitivity](old/13_layer_sensitivity/RESULTS_13.md)
