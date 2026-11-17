# Mask-rule pilot grid: results

Wall-clock: **1.43 hours**. Four tasks × 11 rules × 2 regularizers × five rotations × three seeds. Scorers, λ pilots and the 20-subset validity readouts are additional fits.

Accuracy/F1/dependence/leakage values are percentages; gains are percentage points. Brackets on ID/OOD accuracy are paired document-bootstrap 95% intervals; other columns give rotation-equal means. All metrics with seed SD, intervals, contributing rotations and valid-bootstrap counts are in summary.csv. “p” is a descriptive mean of cell-level Bonferroni p-values, not an aggregate significance test. H1–H5 use the unregularized grid; H6 compares regularizers. The best-fixed comparator is the per-rotation/seed OOD oracle maximum, not a test-selected deployed model.

## ud_upos

| Rule × penalty | ID accuracy / F1 | OOD accuracy / F1 | Gain ID / OOD / shrink | Invariance F / p / reject % | Genre BA | Mask BA |
|---|---|---|---|---|---:|---:|
| full × none | 67.86 ± 0.22 [66.27, 69.08] / 50.80 | 67.70 ± 0.24 [66.02, 69.45] / 50.68 | -5.83 / -6.98 / +1.15 | 3.99 / 0.551 / 33.3 | 35.20 | 36.75 |
| full × cond | 67.34 ± 0.40 [65.97, 68.52] / 50.19 | 68.15 ± 0.40 [66.64, 69.57] / 51.03 | -6.76 / -6.39 / -0.37 | 3.86 / 0.568 / 13.3 | 34.96 | 36.75 |
| word × none | 73.70 ± 0.20 [72.53, 74.74] / 58.40 | 74.68 ± 0.59 [73.18, 76.23] / 58.22 | +0.00 / +0.00 / +0.00 | 2.75 / 0.797 / 0.0 | 35.77 | 34.73 |
| word × cond | 74.10 ± 0.26 [73.01, 75.25] / 58.70 | 74.54 ± 0.14 [73.09, 76.12] / 58.10 | +0.00 / +0.00 / +0.00 | 3.43 / 0.638 / 0.0 | 35.83 | 34.73 |
| lin1 × none | 70.91 ± 0.07 [69.36, 72.15] / 55.67 | 73.39 ± 0.23 [72.08, 74.91] / 57.65 | -2.79 / -1.28 / -1.50 | 3.57 / 0.520 / 0.0 | 35.26 | 36.28 |
| lin1 × cond | 70.56 ± 0.25 [69.24, 71.75] / 55.49 | 72.95 ± 0.75 [71.72, 74.47] / 57.35 | -3.54 / -1.59 / -1.96 | 4.07 / 0.519 / 20.0 | 34.84 | 36.28 |
| lin2 × none | 69.06 ± 0.03 [67.68, 70.25] / 53.57 | 71.12 ± 0.29 [69.59, 72.61] / 54.40 | -4.64 / -3.55 / -1.08 | 3.83 / 0.480 / 13.3 | 35.96 | 34.30 |
| lin2 × cond | 68.92 ± 0.63 [67.63, 70.10] / 53.73 | 71.24 ± 0.42 [69.71, 72.73] / 54.17 | -5.17 / -3.30 / -1.88 | 4.33 / 0.360 / 0.0 | 35.64 | 34.30 |
| tree1 × none | 70.83 ± 0.27 [69.40, 72.03] / 56.18 | 73.45 ± 0.26 [71.94, 75.07] / 57.77 | -2.86 / -1.23 / -1.64 | 4.48 / 0.388 / 13.3 | 35.39 | 33.96 |
| tree1 × cond | 70.15 ± 0.36 [68.78, 71.29] / 55.24 | 72.96 ± 0.61 [71.54, 74.53] / 57.28 | -3.95 / -1.57 / -2.38 | 5.40 / 0.374 / 26.7 | 35.87 | 33.96 |
| tree2 × none | 68.51 ± 0.08 [67.06, 69.68] / 52.77 | 69.53 ± 0.32 [68.04, 71.19] / 52.88 | -5.19 / -5.15 / -0.04 | 4.90 / 0.342 / 40.0 | 34.47 | 38.08 |
| tree2 × cond | 68.19 ± 0.26 [66.74, 69.38] / 52.26 | 68.92 ± 0.39 [67.54, 70.48] / 52.14 | -5.91 / -5.61 / -0.30 | 4.99 / 0.354 / 20.0 | 33.89 | 38.08 |
| rand_tree1 × none | 69.58 ± 0.34 [68.12, 70.72] / 53.56 | 69.47 ± 0.37 [68.02, 71.20] / 53.25 | -4.11 / -5.21 / +1.09 | 4.72 / 0.318 / 13.3 | 35.65 | 36.29 |
| rand_tree1 × cond | 69.73 ± 0.42 [68.31, 70.88] / 53.66 | 69.99 ± 0.45 [68.64, 71.64] / 54.03 | -4.37 / -4.55 / +0.18 | 5.38 / 0.248 / 20.0 | 35.26 | 36.29 |
| full_prev × none | 72.76 ± 0.41 [71.08, 73.96] / 56.71 | 73.66 ± 0.70 [72.13, 75.02] / 57.16 | -0.94 / -1.02 / +0.08 | 3.77 / 0.587 / 6.7 | 35.97 | 37.24 |
| full_prev × cond | 72.75 ± 0.40 [71.28, 74.00] / 55.43 | 73.14 ± 0.28 [71.74, 74.79] / 56.30 | -1.35 / -1.40 / +0.05 | 3.72 / 0.583 / 0.0 | 35.55 | 37.24 |
| sel_suff × none | 71.65 ± 0.08 [70.30, 72.81] / 55.71 | 71.03 ± 0.43 [69.28, 72.63] / 55.72 | -2.05 / -3.65 / +1.60 | 4.46 / 0.515 / 33.3 | 36.04 | 39.16 |
| sel_suff × cond | 71.94 ± 0.20 [70.67, 73.12] / 56.33 | 71.74 ± 0.39 [70.29, 73.08] / 55.96 | -2.16 / -2.80 / +0.64 | 4.89 / 0.433 / 20.0 | 36.21 | 39.16 |
| sel_inv_genre × none | 70.99 ± 0.08 [69.48, 72.20] / 55.06 | 71.09 ± 0.43 [69.49, 72.62] / 55.12 | -2.71 / -3.59 / +0.88 | 4.98 / 0.340 / 33.3 | 35.52 | 38.41 |
| sel_inv_genre × cond | 71.20 ± 0.27 [69.89, 72.51] / 55.86 | 71.52 ± 0.46 [70.07, 72.94] / 55.53 | -2.90 / -3.02 / +0.12 | 5.14 / 0.247 / 20.0 | 35.58 | 38.41 |
| sel_inv_free × none | 71.02 ± 0.08 [69.34, 72.32] / 55.78 | 71.70 ± 0.28 [70.11, 73.43] / 55.07 | -2.68 / -2.97 / +0.29 | 4.81 / 0.385 / 33.3 | 35.28 | 37.98 |
| sel_inv_free × cond | 71.43 ± 0.61 [70.10, 72.59] / 56.35 | 72.04 ± 0.26 [70.50, 73.66] / 55.74 | -2.66 / -2.50 / -0.16 | 5.19 / 0.186 / 13.3 | 36.21 | 37.98 |

- H1 locality/syntax: **supported**.
- H2 covariate shift vs non-invariance: **inconclusive as a causal distinction**; fitted-residual pass/reject patterns are in H2_patterns.csv.
- H3 previous-sentence context: **inconclusive**.
- H4 invariance-based selection: **inconclusive**.
- H5 annotation-free environments: **inconclusive**, equivalence margin ±0.5 pp.
- H6 conditional penalty: **supported**.

Scorer-validity mean Spearman: **0.340** (poor if <.5); random-document calibration rejection rate: **3.1%** (α=5%, conservative Bonferroni). Selection evidence is discounted if mean ranking <.5 or null rejection >10%.
Fallback to word, genre/free: 0/5 and 1/5. Chosen subsets and descriptive ICP intersections are in selections.json.

## ud_deprel

| Rule × penalty | ID accuracy / F1 | OOD accuracy / F1 | Gain ID / OOD / shrink | Invariance F / p / reject % | Genre BA | Mask BA |
|---|---|---|---|---|---:|---:|
| full × none | 52.94 ± 0.32 [51.54, 54.07] / 19.71 | 52.80 ± 0.30 [51.08, 54.37] / 18.32 | -1.00 / -0.42 / -0.58 | 12.83 / 0.018 / 86.7 | 43.95 | 42.93 |
| full × cond | 52.94 ± 0.32 [51.54, 54.07] / 19.71 | 52.80 ± 0.30 [51.08, 54.37] / 18.32 | -1.00 / -0.42 / -0.58 | 12.83 / 0.018 / 86.7 | 43.95 | 42.93 |
| word × none | 53.94 ± 0.19 [52.49, 55.25] / 21.19 | 53.22 ± 0.47 [51.69, 54.96] / 19.26 | +0.00 / +0.00 / +0.00 | 14.78 / 0.008 / 93.3 | 43.58 | 41.76 |
| word × cond | 53.94 ± 0.19 [52.49, 55.25] / 21.19 | 53.22 ± 0.47 [51.69, 54.96] / 19.26 | +0.00 / +0.00 / +0.00 | 14.78 / 0.008 / 93.3 | 43.58 | 41.76 |
| lin1 × none | 54.75 ± 0.28 [52.79, 56.13] / 22.03 | 53.69 ± 0.32 [52.14, 55.21] / 20.42 | +0.81 / +0.47 / +0.33 | 28.19 / 0.001 / 100.0 | 41.29 | 42.17 |
| lin1 × cond | 54.75 ± 0.28 [52.79, 56.13] / 22.03 | 53.69 ± 0.32 [52.14, 55.21] / 20.42 | +0.81 / +0.47 / +0.33 | 28.19 / 0.001 / 100.0 | 41.29 | 42.17 |
| lin2 × none | 54.58 ± 0.05 [53.14, 55.72] / 21.99 | 53.78 ± 0.24 [51.93, 55.66] / 18.97 | +0.64 / +0.57 / +0.08 | 35.67 / 0.000 / 100.0 | 42.69 | 42.77 |
| lin2 × cond | 54.58 ± 0.05 [53.14, 55.72] / 21.99 | 53.78 ± 0.24 [51.93, 55.66] / 18.97 | +0.64 / +0.57 / +0.08 | 35.67 / 0.000 / 100.0 | 42.69 | 42.77 |
| tree1 × none | 58.16 ± 0.40 [56.64, 59.32] / 23.17 | 59.02 ± 0.42 [57.28, 60.95] / 22.65 | +4.22 / +5.80 / -1.58 | 14.16 / 0.008 / 100.0 | 43.63 | 41.69 |
| tree1 × cond | 58.16 ± 0.40 [56.64, 59.32] / 23.17 | 59.02 ± 0.42 [57.28, 60.95] / 22.65 | +4.22 / +5.80 / -1.58 | 14.16 / 0.008 / 100.0 | 43.63 | 41.69 |
| tree2 × none | 54.06 ± 0.54 [52.50, 55.54] / 21.30 | 53.36 ± 0.42 [51.77, 55.38] / 18.79 | +0.12 / +0.14 / -0.02 | 24.21 / 0.002 / 100.0 | 42.16 | 42.52 |
| tree2 × cond | 54.06 ± 0.54 [52.50, 55.54] / 21.30 | 53.36 ± 0.42 [51.77, 55.38] / 18.79 | +0.12 / +0.14 / -0.02 | 24.21 / 0.002 / 100.0 | 42.16 | 42.52 |
| rand_tree1 × none | 52.18 ± 0.19 [50.83, 53.71] / 20.21 | 52.87 ± 0.34 [51.38, 54.66] / 18.38 | -1.76 / -0.35 / -1.41 | 23.63 / 0.003 / 100.0 | 41.27 | 42.21 |
| rand_tree1 × cond | 52.18 ± 0.19 [50.83, 53.71] / 20.21 | 52.87 ± 0.34 [51.38, 54.66] / 18.38 | -1.76 / -0.35 / -1.41 | 23.63 / 0.003 / 100.0 | 41.27 | 42.21 |
| full_prev × none | 55.38 ± 0.22 [54.04, 56.60] / 20.84 | 56.28 ± 0.18 [54.60, 58.05] / 20.14 | +1.44 / +3.07 / -1.63 | 16.36 / 0.002 / 100.0 | 43.05 | 42.34 |
| full_prev × cond | 55.38 ± 0.22 [54.04, 56.60] / 20.84 | 56.28 ± 0.18 [54.60, 58.05] / 20.14 | +1.44 / +3.07 / -1.63 | 16.36 / 0.002 / 100.0 | 43.05 | 42.34 |
| sel_suff × none | 56.84 ± 0.22 [55.41, 58.17] / 23.63 | 55.84 ± 0.38 [53.97, 57.70] / 20.47 | +2.90 / +2.63 / +0.27 | 24.10 / 0.001 / 100.0 | 41.39 | 42.50 |
| sel_suff × cond | 56.84 ± 0.22 [55.41, 58.17] / 23.63 | 55.84 ± 0.38 [53.97, 57.70] / 20.47 | +2.90 / +2.63 / +0.27 | 24.10 / 0.001 / 100.0 | 41.39 | 42.50 |
| sel_inv_genre × none | 56.71 ± 0.29 [55.12, 58.15] / 23.76 | 56.59 ± 0.17 [54.94, 58.34] / 20.38 | +2.77 / +3.37 / -0.60 | 23.33 / 0.000 / 100.0 | 42.68 | 43.06 |
| sel_inv_genre × cond | 56.71 ± 0.29 [55.12, 58.15] / 23.76 | 56.59 ± 0.17 [54.94, 58.34] / 20.38 | +2.77 / +3.37 / -0.60 | 23.33 / 0.000 / 100.0 | 42.68 | 43.06 |
| sel_inv_free × none | 56.40 ± 0.17 [54.80, 57.75] / 22.85 | 55.75 ± 0.07 [54.11, 57.50] / 20.32 | +2.46 / +2.53 / -0.08 | 25.57 / 0.000 / 100.0 | 41.06 | 43.31 |
| sel_inv_free × cond | 56.40 ± 0.17 [54.80, 57.75] / 22.85 | 55.75 ± 0.07 [54.11, 57.50] / 20.32 | +2.46 / +2.53 / -0.08 | 25.57 / 0.000 / 100.0 | 41.06 | 43.31 |

- H1 locality/syntax: **inconclusive**.
- H2 covariate shift vs non-invariance: **inconclusive as a causal distinction**; fitted-residual pass/reject patterns are in H2_patterns.csv.
- H3 previous-sentence context: **inconclusive**.
- H4 invariance-based selection: **inconclusive**.
- H5 annotation-free environments: **inconclusive**, equivalence margin ±0.5 pp.
- H6 conditional penalty: **inconclusive**.

Scorer-validity mean Spearman: **0.308** (poor if <.5); random-document calibration rejection rate: **1.5%** (α=5%, conservative Bonferroni). Selection evidence is discounted if mean ranking <.5 or null rejection >10%.
Fallback to word, genre/free: 0/5 and 0/5. Chosen subsets and descriptive ICP intersections are in selections.json.

## gum_upos

| Rule × penalty | ID accuracy / F1 | OOD accuracy / F1 | Gain ID / OOD / shrink | Invariance F / p / reject % | Genre BA | Mask BA |
|---|---|---|---|---|---:|---:|
| full × none | 82.51 ± 0.00 [82.02, 83.04] / 72.47 | 80.08 ± 0.70 [78.93, 81.29] / 61.47 | -1.31 / -2.37 / +1.06 | 4.28 / 0.231 / 0.0 | 25.59 | 26.18 |
| full × cond | 83.09 ± 0.15 [82.51, 83.66] / 69.77 | 80.50 ± 0.96 [78.99, 82.12] / 62.19 | -1.29 / -1.69 / +0.40 | 7.28 / 0.015 / 100.0 | 25.89 | 26.18 |
| word × none | 83.83 ± 0.65 [83.24, 84.37] / 68.27 | 82.46 ± 0.56 [80.97, 83.68] / 63.64 | +0.00 / +0.00 / +0.00 | 11.03 / 0.003 / 100.0 | 26.34 | 26.83 |
| word × cond | 84.39 ± 0.21 [83.87, 84.88] / 68.38 | 82.20 ± 0.36 [80.45, 83.33] / 62.92 | +0.00 / +0.00 / +0.00 | 11.57 / 0.002 / 100.0 | 26.21 | 26.83 |
| lin1 × none | 84.98 ± 0.18 [84.43, 85.58] / 72.15 | 82.28 ± 0.39 [80.56, 83.71] / 64.16 | +1.15 / -0.18 / +1.33 | 10.08 / 0.011 / 86.7 | 26.10 | 26.86 |
| lin1 × cond | 85.31 ± 0.26 [84.75, 85.89] / 71.68 | 82.21 ± 0.70 [80.81, 83.67] / 63.13 | +0.92 / +0.01 / +0.91 | 13.43 / 0.002 / 100.0 | 26.49 | 26.86 |
| lin2 × none | 82.82 ± 0.19 [82.24, 83.29] / 73.04 | 80.70 ± 0.45 [79.29, 82.39] / 62.50 | -1.01 / -1.75 / +0.75 | 6.32 / 0.033 / 73.3 | 25.36 | 26.14 |
| lin2 × cond | 83.91 ± 0.18 [83.39, 84.47] / 70.49 | 80.95 ± 0.37 [79.69, 82.56] / 62.86 | -0.48 / -1.25 / +0.77 | 9.81 / 0.006 / 100.0 | 26.27 | 26.14 |
| tree1 × none | 83.68 ± 0.39 [83.23, 84.33] / 73.24 | 81.75 ± 0.25 [80.37, 83.32] / 63.35 | -0.15 / -0.70 / +0.55 | 8.43 / 0.013 / 93.3 | 26.36 | 26.82 |
| tree1 × cond | 83.81 ± 0.34 [83.35, 84.40] / 69.74 | 81.63 ± 0.48 [80.32, 83.09] / 61.90 | -0.57 / -0.56 / -0.01 | 9.57 / 0.004 / 100.0 | 25.69 | 26.82 |
| tree2 × none | 82.98 ± 0.23 [82.55, 83.37] / 72.87 | 81.52 ± 0.67 [80.05, 83.10] / 64.04 | -0.85 / -0.93 / +0.08 | 8.63 / 0.012 / 93.3 | 26.37 | 25.98 |
| tree2 × cond | 83.73 ± 0.39 [83.29, 84.23] / 69.88 | 82.04 ± 0.42 [80.82, 83.37] / 63.45 | -0.65 / -0.15 / -0.50 | 11.50 / 0.002 / 100.0 | 25.23 | 25.98 |
| rand_tree1 × none | 81.77 ± 0.27 [81.26, 82.37] / 66.96 | 80.27 ± 0.31 [78.72, 81.96] / 61.77 | -2.06 / -2.18 / +0.12 | 9.96 / 0.003 / 100.0 | 26.14 | 27.12 |
| rand_tree1 × cond | 82.47 ± 0.17 [81.95, 83.02] / 67.12 | 80.28 ± 0.41 [78.70, 81.82] / 61.11 | -1.91 / -1.91 / +0.00 | 10.64 / 0.002 / 100.0 | 25.93 | 27.12 |
| full_prev × none | 87.51 ± 0.08 [87.06, 88.07] / 77.70 | 85.75 ± 0.12 [84.67, 86.90] / 68.19 | +3.69 / +3.29 / +0.39 | 4.48 / 0.185 / 6.7 | 26.00 | 25.97 |
| full_prev × cond | 88.20 ± 0.10 [87.80, 88.71] / 75.16 | 86.39 ± 0.47 [85.34, 87.41] / 68.94 | +3.81 / +4.20 / -0.39 | 6.14 / 0.034 / 66.7 | 26.13 | 25.97 |
| sel_suff × none | 84.04 ± 0.10 [83.50, 84.67] / 73.87 | 82.35 ± 0.22 [80.76, 84.17] / 64.13 | +0.21 / -0.10 / +0.31 | 7.56 / 0.021 / 93.3 | 25.97 | 26.25 |
| sel_suff × cond | 84.34 ± 0.25 [83.76, 84.99] / 71.01 | 82.50 ± 0.60 [80.81, 84.27] / 63.57 | -0.04 / +0.30 / -0.34 | 9.80 / 0.004 / 100.0 | 25.47 | 26.25 |
| sel_inv_genre × none | 83.68 ± 0.04 [83.08, 84.30] / 73.53 | 81.86 ± 0.38 [80.39, 83.52] / 63.93 | -0.15 / -0.59 / +0.45 | 8.65 / 0.019 / 93.3 | 25.54 | 25.81 |
| sel_inv_genre × cond | 84.03 ± 0.07 [83.41, 84.66] / 70.76 | 82.35 ± 0.19 [80.67, 84.01] / 63.49 | -0.35 / +0.16 / -0.51 | 10.61 / 0.004 / 100.0 | 25.22 | 25.81 |
| sel_inv_free × none | 83.82 ± 0.08 [83.20, 84.32] / 72.54 | 82.12 ± 0.36 [80.79, 83.65] / 64.14 | -0.01 / -0.34 / +0.33 | 8.86 / 0.017 / 93.3 | 25.52 | 25.88 |
| sel_inv_free × cond | 84.30 ± 0.18 [83.71, 84.87] / 70.32 | 82.50 ± 0.65 [80.93, 84.06] / 62.88 | -0.08 / +0.31 / -0.39 | 10.45 / 0.005 / 100.0 | 25.33 | 25.88 |

- H1 locality/syntax: **supported**.
- H2 covariate shift vs non-invariance: **inconclusive as a causal distinction**; fitted-residual pass/reject patterns are in H2_patterns.csv.
- H3 previous-sentence context: **inconclusive**.
- H4 invariance-based selection: **inconclusive**.
- H5 annotation-free environments: **inconclusive**, equivalence margin ±0.5 pp.
- H6 conditional penalty: **inconclusive**.

Scorer-validity mean Spearman: **0.473** (poor if <.5); random-document calibration rejection rate: **9.3%** (α=5%, conservative Bonferroni). Selection evidence is discounted if mean ranking <.5 or null rejection >10%.
Fallback to word, genre/free: 0/5 and 1/5. Chosen subsets and descriptive ICP intersections are in selections.json.

## ud_discourse

| Rule × penalty | ID accuracy / F1 | OOD accuracy / F1 | Gain ID / OOD / shrink | Invariance F / p / reject % | Genre BA | Mask BA |
|---|---|---|---|---|---:|---:|
| full × none | 20.89 ± 0.28 [20.32, 21.41] / 14.67 | 18.42 ± 0.54 [16.81, 19.76] / 8.90 | +6.52 / +3.36 / +3.16 | 11.03 / 0.004 / 100.0 | 36.65 | 36.99 |
| full × cond | 20.89 ± 0.28 [20.32, 21.41] / 14.67 | 18.42 ± 0.54 [16.81, 19.76] / 8.90 | +6.52 / +3.36 / +3.16 | 11.03 / 0.004 / 100.0 | 36.65 | 36.99 |
| word × none | 14.36 ± 0.13 [13.89, 14.75] / 7.52 | 15.06 ± 0.23 [13.44, 16.55] / 5.62 | +0.00 / +0.00 / +0.00 | 7.68 / 0.036 / 73.3 | 35.00 | 35.27 |
| word × cond | 14.36 ± 0.13 [13.89, 14.75] / 7.52 | 15.06 ± 0.23 [13.44, 16.55] / 5.62 | +0.00 / +0.00 / +0.00 | 7.68 / 0.036 / 73.3 | 35.00 | 35.27 |
| lin1 × none | 15.26 ± 0.20 [14.85, 15.69] / 8.59 | 15.13 ± 0.92 [13.65, 16.64] / 6.24 | +0.90 / +0.07 / +0.83 | 7.33 / 0.066 / 53.3 | 36.47 | 36.99 |
| lin1 × cond | 15.26 ± 0.20 [14.85, 15.69] / 8.59 | 15.13 ± 0.92 [13.65, 16.64] / 6.24 | +0.90 / +0.07 / +0.83 | 7.33 / 0.066 / 53.3 | 36.47 | 36.99 |
| lin2 × none | 16.88 ± 0.07 [16.34, 17.33] / 10.05 | 14.40 ± 0.62 [12.78, 15.78] / 6.11 | +2.52 / -0.66 / +3.17 | 6.27 / 0.167 / 33.3 | 37.51 | 38.06 |
| lin2 × cond | 16.88 ± 0.07 [16.34, 17.33] / 10.05 | 14.40 ± 0.62 [12.78, 15.78] / 6.11 | +2.52 / -0.66 / +3.17 | 6.27 / 0.167 / 33.3 | 37.51 | 38.06 |
| tree1 × none | 14.62 ± 0.38 [14.21, 15.05] / 8.37 | 13.30 ± 0.18 [12.14, 14.51] / 5.54 | +0.26 / -1.76 / +2.02 | 8.52 / 0.033 / 73.3 | 36.24 | 35.16 |
| tree1 × cond | 14.62 ± 0.38 [14.21, 15.05] / 8.37 | 13.30 ± 0.18 [12.14, 14.51] / 5.54 | +0.26 / -1.76 / +2.02 | 8.52 / 0.033 / 73.3 | 36.24 | 35.16 |
| tree2 × none | 17.38 ± 0.25 [16.89, 17.82] / 10.98 | 15.66 ± 0.44 [14.19, 17.12] / 7.77 | +3.02 / +0.61 / +2.41 | 7.94 / 0.036 / 86.7 | 36.93 | 36.38 |
| tree2 × cond | 17.38 ± 0.25 [16.89, 17.82] / 10.98 | 15.66 ± 0.44 [14.19, 17.12] / 7.77 | +3.02 / +0.61 / +2.41 | 7.94 / 0.036 / 86.7 | 36.93 | 36.38 |
| rand_tree1 × none | 14.10 ± 0.35 [13.70, 14.51] / 7.59 | 13.11 ± 0.60 [11.69, 14.26] / 5.36 | -0.26 / -1.94 / +1.69 | 9.30 / 0.023 / 86.7 | 36.55 | 36.37 |
| rand_tree1 × cond | 14.10 ± 0.35 [13.70, 14.51] / 7.59 | 13.11 ± 0.60 [11.69, 14.26] / 5.36 | -0.26 / -1.94 / +1.69 | 9.30 / 0.023 / 86.7 | 36.55 | 36.37 |
| full_prev × none | 22.53 ± 0.32 [21.97, 23.09] / 15.53 | 20.37 ± 0.59 [18.59, 21.81] / 10.10 | +8.17 / +5.32 / +2.86 | 7.90 / 0.027 / 80.0 | 37.94 | 34.98 |
| full_prev × cond | 22.53 ± 0.32 [21.97, 23.09] / 15.53 | 20.37 ± 0.59 [18.59, 21.81] / 10.10 | +8.17 / +5.32 / +2.86 | 7.90 / 0.027 / 80.0 | 37.94 | 34.98 |
| sel_suff × none | 16.35 ± 0.53 [15.75, 16.78] / 9.55 | 13.78 ± 0.58 [12.36, 15.13] / 6.08 | +1.98 / -1.28 / +3.26 | 6.36 / 0.125 / 40.0 | 37.13 | 37.33 |
| sel_suff × cond | 16.35 ± 0.53 [15.75, 16.78] / 9.55 | 13.78 ± 0.58 [12.36, 15.13] / 6.08 | +1.98 / -1.28 / +3.26 | 6.36 / 0.125 / 40.0 | 37.13 | 37.33 |
| sel_inv_genre × none | 14.26 ± 0.12 [13.77, 14.69] / 7.54 | 14.96 ± 0.27 [13.53, 16.52] / 5.85 | -0.10 / -0.10 / -0.01 | 7.70 / 0.037 / 73.3 | 34.58 | 35.27 |
| sel_inv_genre × cond | 14.26 ± 0.12 [13.77, 14.69] / 7.54 | 14.96 ± 0.27 [13.53, 16.52] / 5.85 | -0.10 / -0.10 / -0.01 | 7.70 / 0.037 / 73.3 | 34.58 | 35.27 |
| sel_inv_free × none | 14.95 ± 0.16 [14.46, 15.42] / 8.24 | 15.57 ± 0.17 [14.32, 16.98] / 6.25 | +0.59 / +0.51 / +0.08 | 7.49 / 0.061 / 60.0 | 35.16 | 36.22 |
| sel_inv_free × cond | 14.95 ± 0.16 [14.46, 15.42] / 8.24 | 15.57 ± 0.17 [14.32, 16.98] / 6.25 | +0.59 / +0.51 / +0.08 | 7.49 / 0.061 / 60.0 | 35.16 | 36.22 |

- H1 locality/syntax: **contradicted**.
- H2 covariate shift vs non-invariance: **inconclusive as a causal distinction**; fitted-residual pass/reject patterns are in H2_patterns.csv.
- H3 previous-sentence context: **inconclusive**.
- H4 invariance-based selection: **inconclusive**.
- H5 annotation-free environments: **inconclusive**, equivalence margin ±0.5 pp.
- H6 conditional penalty: **inconclusive**.

Scorer-validity mean Spearman: **0.136** (poor if <.5); random-document calibration rejection rate: **15.5%** (α=5%, conservative Bonferroni). Selection evidence is discounted if mean ranking <.5 or null rejection >10%.
Fallback to word, genre/free: 5/5 and 4/5. Chosen subsets and descriptive ICP intersections are in selections.json.

## Scope and limitations

All 11 mandatory rules and both regularizers were run; no cost-driven rule drops. The optional per-word-class selector was omitted. The prompt’s 12-rule total includes that optional rule. This pilot uses fixed document-spread focal samples, at most two per sentence, with caps 256/64/256 per genre for native train/validation/test; actual counts are in samples.csv. Probe minimum supports are reduced to 8/4/4 for this pilot and coverage is explicit. Native validation documents select E3 probe checkpoints; held-document folds use five EWT/two GUM folds. All masks are re-encoded, with identical masked strings deduplicated. Scorers use eight cached uniformly sampled subsets per focal as their training mask pool. Full-prev prepends as much of the previous sentence as fits in 512 WordPieces, preserving the complete current cached sentence; changed absolute positions and unavailable/truncated previous context limit H3 interpretation.

A residual-mean rejection can arise from estimation or calibration error, not only a changing true conditional. Non-rejection does not establish invariance, and ICP intersections are descriptive, not proven necessary slots. Bonferroni controls across classes, not the search over 256 subsets. Genre-free clustering uses no genre annotation as its environment labels, but the shared rotation and balanced sampling design still uses genres. Gold syntax itself can reveal annotation-related structure; the mask-shape probe measures some of this. No test labels select subsets or λ. Intervals condition on fitted readouts, frozen scorers and selections; models/probes are not refitted in bootstrap. Nominal intervals accompany only the declared H1–H6 contrast family; other measurements are descriptive.

[Implementation](IMPLEMENTATION_5.md); [tables](tables/pilot/).
