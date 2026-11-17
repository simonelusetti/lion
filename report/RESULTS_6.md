# Experiment 6: conditional randomization tests

Wall-clock: **0.31 hours**. **3885/3885 sentence rows**, K=200; 0 row-budget halvings. The measured 1% benchmark and projection are in benchmark.json and budget.json.

One fp16 bert-base-cased backbone and its own pretrained MLM head supply both representations and samples. Three independent seed-0 MLPs use every eligible training word’s ±16-subword window, with CLS/SEP. Training uses a fixed ten epochs; dev accuracy covers all eligible dev words and does not select the scorer, since dev rows enter the CRT.

| Task | Dev accuracy | Dev macro-F1 | Epoch |
|---|---:|---:|---:|
| synthetic | 81.55% | 67.75% | 10 |
| upos | 97.00% | 91.84% | 10 |
| deprel | 87.61% | 64.00% | 10 |

## Arm A: synthetic labels

−2 and +1 are non-null; the other single offsets are null by the specified label rule. Rejection fractions have binomial SEs treating sentence rows as independent. Ties use exact equality and count against rejection.

| d | Rows | Null | p≤.05 ± SE | p≤.01 ± SE | Ties | Aggregate p | z | Δ |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| -5 | 1897 | True | 0.044 ± 0.005 | 0.014 ± 0.003 | 0.455 | 0.07463 | 1.49 | +0.0030 |
| -4 | 2140 | True | 0.050 ± 0.005 | 0.013 ± 0.002 | 0.423 | 0.04975 | 1.71 | +0.0039 |
| -3 | 2469 | True | 0.043 ± 0.004 | 0.012 ± 0.002 | 0.439 | 0.58209 | -0.14 | -0.0003 |
| -2 | 2851 | False | 0.044 ± 0.004 | 0.013 ± 0.002 | 0.424 | 0.00498 | 5.40 | +0.0137 |
| -1 | 3265 | True | 0.043 ± 0.004 | 0.015 ± 0.002 | 0.370 | 0.92537 | -1.44 | -0.0050 |
| 1 | 3227 | False | 0.056 ± 0.004 | 0.020 ± 0.002 | 0.405 | 0.00498 | 7.10 | +0.0322 |
| 2 | 2850 | True | 0.049 ± 0.004 | 0.019 ± 0.003 | 0.427 | 0.98507 | -2.21 | -0.0051 |
| 3 | 2575 | True | 0.057 ± 0.005 | 0.018 ± 0.003 | 0.439 | 0.68657 | -0.39 | -0.0008 |
| 4 | 2287 | True | 0.046 ± 0.004 | 0.019 ± 0.003 | 0.460 | 0.09950 | 1.37 | +0.0022 |
| 5 | 1990 | True | 0.051 ± 0.005 | 0.015 ± 0.003 | 0.465 | 1.00000 | -2.62 | -0.0050 |

[Ten-bin per-row p histograms for each null offset](figures/crt/synthetic_null_histograms.png). Histogram counts are also in null_histograms.csv.

## Arm B: upos

| d | Rows | Δ | z | Aggregate p | Masking effect | Per-row p≤.05 |
|---|---:|---:|---:|---:|---:|---:|
| -5 | 1897 | +0.0020 | 1.00 | 0.11940 | +0.0082 | 0.038 |
| -4 | 2140 | +0.0073 | 2.29 | 0.01990 | +0.0041 | 0.039 |
| -3 | 2469 | +0.0087 | 3.26 | 0.00498 | +0.0014 | 0.040 |
| -2 | 2851 | +0.0114 | 3.14 | 0.00498 | -0.0014 | 0.052 |
| -1 | 3265 | +0.0521 | 7.03 | 0.00498 | +0.0242 | 0.047 |
| 1 | 3227 | +0.0717 | 9.40 | 0.00498 | +0.0473 | 0.070 |
| 2 | 2850 | +0.0105 | 3.27 | 0.00498 | +0.0057 | 0.061 |
| 3 | 2575 | +0.0093 | 3.15 | 0.00498 | -0.0031 | 0.052 |
| 4 | 2287 | +0.0077 | 2.85 | 0.00995 | +0.0112 | 0.043 |
| 5 | 1990 | -0.0003 | -0.11 | 0.53234 | +0.0109 | 0.046 |

## Arm B: deprel

| d | Rows | Δ | z | Aggregate p | Masking effect | Per-row p≤.05 |
|---|---:|---:|---:|---:|---:|---:|
| -5 | 1897 | +0.0105 | 2.49 | 0.00498 | +0.0064 | 0.045 |
| -4 | 2140 | +0.0368 | 7.78 | 0.00498 | +0.0219 | 0.049 |
| -3 | 2469 | +0.0280 | 6.29 | 0.00498 | +0.0052 | 0.036 |
| -2 | 2851 | +0.0500 | 7.89 | 0.00498 | +0.0304 | 0.057 |
| -1 | 3265 | +0.1624 | 15.28 | 0.00498 | +0.1295 | 0.057 |
| 1 | 3227 | +0.2490 | 21.45 | 0.00498 | +0.1951 | 0.081 |
| 2 | 2850 | +0.0349 | 6.22 | 0.00498 | +0.0418 | 0.060 |
| 3 | 2575 | +0.0166 | 4.05 | 0.00498 | +0.0094 | 0.056 |
| 4 | 2287 | +0.0089 | 2.61 | 0.00995 | +0.0068 | 0.048 |
| 5 | 1990 | +0.0061 | 2.08 | 0.01990 | +0.0086 | 0.050 |

## Arm C: adjacent pairs

| Task | Pair | Rows | Δ | z | Pair p | Single p (first / second) | Masking effect | Per-row p≤.05 |
|---|---|---:|---:|---:|---:|---|---:|---:|
| synthetic | 1,2 | 2850 | +0.0652 | 9.09 | 0.00498 | 0.00498 / 0.98507 | +0.0343 | 0.085 |
| upos | 1,2 | 2850 | +0.1773 | 13.32 | 0.00498 | 0.00498 / 0.00498 | +0.1066 | 0.095 |
| deprel | 1,2 | 2850 | +0.4570 | 22.72 | 0.00498 | 0.00498 / 0.00498 | +0.2952 | 0.111 |
| synthetic | 2,3 | 2575 | -0.0035 | -0.88 | 0.81095 | 0.98507 / 0.68657 | -0.0134 | 0.070 |
| upos | 2,3 | 2575 | +0.0250 | 4.47 | 0.00498 | 0.00498 / 0.00498 | +0.0102 | 0.069 |
| deprel | 2,3 | 2575 | +0.0789 | 8.41 | 0.00498 | 0.00498 / 0.00498 | +0.0668 | 0.079 |
| synthetic | -2,-1 | 2851 | +0.0379 | 6.26 | 0.00498 | 0.00498 / 0.92537 | +0.0229 | 0.073 |
| upos | -2,-1 | 2851 | +0.1010 | 9.46 | 0.00498 | 0.00498 / 0.00498 | +0.0460 | 0.074 |
| deprel | -2,-1 | 2851 | +0.4682 | 24.77 | 0.00498 | 0.00498 / 0.00498 | +0.3344 | 0.123 |
| synthetic | -3,-2 | 2469 | +0.0182 | 3.87 | 0.00498 | 0.58209 / 0.00498 | +0.0095 | 0.064 |
| upos | -3,-2 | 2469 | +0.0280 | 4.69 | 0.00498 | 0.00498 / 0.00498 | +0.0065 | 0.079 |
| deprel | -3,-2 | 2469 | +0.1556 | 12.32 | 0.00498 | 0.00498 / 0.00498 | +0.0998 | 0.100 |

Synthetic null pair (+2,+3): per-row rejection 0.070 ± 0.005 at .05 and 0.021 ± 0.003 at .01; tie fraction 0.105.

## Effects by sampler entropy

Entropy is the single-position conditional entropy; for pairs it is first-position entropy plus the draw-average second-position conditional entropy. Tercile cut ties remain together; empty groups are shown as nan.

| Task | Offset/pair | Low Δ (n) | Middle Δ (n) | High Δ (n) |
|---|---|---:|---:|---:|
| synthetic | 1 | +0.0117 (1076) | +0.0509 (1075) | +0.0339 (1076) |
| upos | 1 | +0.0051 (1076) | +0.0783 (1075) | +0.1318 (1076) |
| deprel | 1 | +0.0803 (1076) | +0.2694 (1075) | +0.3975 (1076) |
| synthetic | 2 | +0.0022 (950) | -0.0044 (950) | -0.0132 (950) |
| upos | 2 | -0.0001 (950) | +0.0065 (950) | +0.0251 (950) |
| deprel | 2 | -0.0030 (950) | +0.0355 (950) | +0.0722 (950) |
| synthetic | 3 | -0.0022 (859) | -0.0013 (858) | +0.0011 (858) |
| upos | 3 | +0.0024 (859) | +0.0200 (858) | +0.0056 (858) |
| deprel | 3 | +0.0036 (859) | +0.0188 (858) | +0.0275 (858) |
| synthetic | 4 | +0.0049 (763) | -0.0014 (762) | +0.0033 (762) |
| upos | 4 | -0.0038 (763) | +0.0062 (762) | +0.0208 (762) |
| deprel | 4 | -0.0005 (763) | -0.0019 (762) | +0.0291 (762) |
| synthetic | 5 | +0.0001 (664) | -0.0082 (663) | -0.0070 (663) |
| upos | 5 | +0.0035 (664) | +0.0009 (663) | -0.0052 (663) |
| deprel | 5 | +0.0021 (664) | -0.0003 (663) | +0.0165 (663) |
| synthetic | -1 | -0.0035 (1089) | -0.0080 (1088) | -0.0036 (1088) |
| upos | -1 | +0.0097 (1089) | +0.0341 (1088) | +0.1125 (1088) |
| deprel | -1 | +0.0634 (1089) | +0.2020 (1088) | +0.2219 (1088) |
| synthetic | -2 | +0.0107 (951) | +0.0212 (950) | +0.0091 (950) |
| upos | -2 | +0.0011 (951) | +0.0132 (950) | +0.0198 (950) |
| deprel | -2 | +0.0078 (951) | +0.0532 (950) | +0.0890 (950) |
| synthetic | -3 | +0.0030 (823) | -0.0055 (823) | +0.0015 (823) |
| upos | -3 | +0.0002 (823) | +0.0058 (823) | +0.0201 (823) |
| deprel | -3 | +0.0126 (823) | +0.0131 (823) | +0.0582 (823) |
| synthetic | -4 | +0.0040 (714) | +0.0062 (713) | +0.0016 (713) |
| upos | -4 | -0.0006 (714) | +0.0001 (713) | +0.0224 (713) |
| deprel | -4 | +0.0113 (714) | +0.0431 (713) | +0.0560 (713) |
| synthetic | -5 | -0.0015 (633) | +0.0035 (632) | +0.0069 (632) |
| upos | -5 | +0.0027 (633) | -0.0039 (632) | +0.0074 (632) |
| deprel | -5 | -0.0004 (633) | +0.0099 (632) | +0.0220 (632) |
| synthetic | 1,2 | +0.0403 (950) | +0.0977 (950) | +0.0575 (950) |
| upos | 1,2 | +0.1533 (950) | +0.1489 (950) | +0.2298 (950) |
| deprel | 1,2 | +0.2967 (950) | +0.5137 (950) | +0.5606 (950) |
| synthetic | 2,3 | -0.0020 (859) | +0.0017 (858) | -0.0103 (858) |
| upos | 2,3 | +0.0102 (859) | +0.0209 (858) | +0.0440 (858) |
| deprel | 2,3 | +0.0433 (859) | +0.0618 (858) | +0.1318 (858) |
| synthetic | -2,-1 | +0.0332 (951) | +0.0407 (950) | +0.0399 (950) |
| upos | -2,-1 | +0.0421 (951) | +0.1579 (950) | +0.1031 (950) |
| deprel | -2,-1 | +0.3408 (951) | +0.6156 (950) | +0.4483 (950) |
| synthetic | -3,-2 | +0.0292 (823) | +0.0159 (823) | +0.0094 (823) |
| upos | -3,-2 | +0.0138 (823) | +0.0272 (823) | +0.0430 (823) |
| deprel | -3,-2 | +0.0839 (823) | +0.1843 (823) | +0.1986 (823) |

## Factual summary

Across the eight synthetic-null single offsets, per-row .05 rejection fractions range from 0.043 to 0.057; aggregate p-values range from 0.04975 to 1.00000.

upos: the largest measured single-offset mean CRT effect is +0.0717 at d=1; 8/10 aggregate tests have p≤.05.

deprel: the largest measured single-offset mean CRT effect is +0.2490 at d=1; 10/10 aggregate tests have p≤.05.

## Implementation and raw data

[Implementation](IMPLEMENTATION_6.md). Raw arrays are under `results/crt/test/all/*.pt`: original/resampled/masked statistics for all three tasks, token draws, conditional entropy, sentence-row IDs and original replacement tokens. `results/crt/test/baseline.pt` retains the original rows and their sentence/document metadata. Report-stage files retain per-row p-values and effects.

The three arms share identical encoder work and randomizations. Identical sampled windows are scored once and expanded back to all K draws; unchanged replacements copy the original score exactly to preserve ties. No top-k, top-p or temperature change is used. Joint sampling follows the requested two-mask/first-fill chain and batches each distinct first value once per row. No additional arms, datasets, encoders, sweeps or seed repetitions were added. Binomial SEs use sentence rows; source sentences may share documents. Finite-sample CRT validity requires the sampling conditional to match the data conditional; the reported calibration numbers do not assume this equality. No multiplicity correction was applied to these requested diagnostic p-values.
