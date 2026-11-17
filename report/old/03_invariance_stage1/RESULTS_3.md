# Stage 1: fixed invariance modes

Stage-1 wall-clock: **1.60 hours** (45 fits; includes pilots, training and analysis). Kernel: **rbf**, cap 256 tokens per genre/cell. Superseded full-grid time is excluded.

Only R0/R1/R2 were trained. Both entropy thresholds below are post-hoc cuts of the same predictions. Main λ is selected once per task/mode on the pilot rotation’s mean validation macro-F1 across three seeds; all other rotations keep it fixed. The neighbouring λ is a sensitivity check, never a second test-based selection.

| Task | Pilot test / validation | R1 λ / neighbour | R2 λ / neighbour |
|---|---|---:|---:|
| ud_upos | answers / email | 0.1 / 0.3 | 0.3 / 1 |
| ud_deprel | answers / email | 0 / 0.1 | 0 / 0.1 |
| gum_upos | academic / bio | 0 / 0.1 | 0 / 0.1 |

Values are accuracy percentage-point differences: mean ± seed SD [95% document-block bootstrap interval]. Rotations receive equal weight; intervals condition on the three fitted seeds and pilot choices. Unsupported rotations are excluded per contrast; support is retained in the CSVs.

| Task | Policy | H | Comparison | Shift tertile | Difference | Rotations |
|---|---|---:|---|---|---:|---:|
| ud_upos | main | 0.2 | 1: R1−R2 overall | -1 | +0.01 ± 0.04 [-0.02, +0.03] | 5 |
| ud_upos | main | 0.2 | 2: unambiguous − ambiguous/rare | -1 | +0.06 ± 0.08 [+0.01, +0.11] | 5 |
| ud_upos | main | 0.2 | 1: R1−R2 overall | 0 | -0.05 ± 0.06 [-0.09, -0.02] | 5 |
| ud_upos | main | 0.2 | 2: unambiguous − ambiguous/rare | 0 | +0.18 ± 0.19 [+0.10, +0.26] | 5 |
| ud_upos | main | 0.2 | 1: R1−R2 overall | 1 | +0.02 ± 0.05 [-0.00, +0.05] | 5 |
| ud_upos | main | 0.2 | 2: unambiguous − ambiguous/rare | 1 | -0.00 ± 0.01 [-0.08, +0.07] | 5 |
| ud_upos | main | 0.2 | 1: R1−R2 overall | 2 | +0.07 ± 0.07 [+0.02, +0.12] | 5 |
| ud_upos | main | 0.2 | 2: unambiguous − ambiguous/rare | 2 | -0.03 ± 0.10 [-0.13, +0.07] | 5 |
| ud_upos | main | 0.5 | 1: R1−R2 overall | -1 | +0.01 ± 0.04 [-0.02, +0.03] | 5 |
| ud_upos | main | 0.5 | 2: unambiguous − ambiguous/rare | -1 | +0.05 ± 0.09 [-0.01, +0.10] | 5 |
| ud_upos | main | 0.5 | 1: R1−R2 overall | 0 | -0.05 ± 0.06 [-0.09, -0.02] | 5 |
| ud_upos | main | 0.5 | 2: unambiguous − ambiguous/rare | 0 | +0.15 ± 0.27 [+0.06, +0.23] | 5 |
| ud_upos | main | 0.5 | 1: R1−R2 overall | 1 | +0.02 ± 0.05 [-0.00, +0.05] | 5 |
| ud_upos | main | 0.5 | 2: unambiguous − ambiguous/rare | 1 | -0.06 ± 0.08 [-0.14, +0.04] | 5 |
| ud_upos | main | 0.5 | 1: R1−R2 overall | 2 | +0.07 ± 0.07 [+0.02, +0.12] | 5 |
| ud_upos | main | 0.5 | 2: unambiguous − ambiguous/rare | 2 | -0.04 ± 0.08 [-0.14, +0.08] | 5 |
| ud_upos | neighbour | 0.2 | 1: R1−R2 overall | -1 | +0.16 ± 0.10 [+0.12, +0.19] | 5 |
| ud_upos | neighbour | 0.2 | 2: unambiguous − ambiguous/rare | -1 | -0.11 ± 0.06 [-0.16, -0.06] | 5 |
| ud_upos | neighbour | 0.2 | 1: R1−R2 overall | 0 | +0.06 ± 0.30 [-0.00, +0.12] | 5 |
| ud_upos | neighbour | 0.2 | 2: unambiguous − ambiguous/rare | 0 | -0.06 ± 0.16 [-0.16, +0.04] | 5 |
| ud_upos | neighbour | 0.2 | 1: R1−R2 overall | 1 | +0.17 ± 0.06 [+0.13, +0.20] | 5 |
| ud_upos | neighbour | 0.2 | 2: unambiguous − ambiguous/rare | 1 | -0.07 ± 0.00 [-0.15, +0.02] | 5 |
| ud_upos | neighbour | 0.2 | 1: R1−R2 overall | 2 | +0.28 ± 0.11 [+0.19, +0.35] | 5 |
| ud_upos | neighbour | 0.2 | 2: unambiguous − ambiguous/rare | 2 | -0.29 ± 0.12 [-0.41, -0.15] | 5 |
| ud_upos | neighbour | 0.5 | 1: R1−R2 overall | -1 | +0.16 ± 0.10 [+0.12, +0.19] | 5 |
| ud_upos | neighbour | 0.5 | 2: unambiguous − ambiguous/rare | -1 | -0.14 ± 0.05 [-0.20, -0.08] | 5 |
| ud_upos | neighbour | 0.5 | 1: R1−R2 overall | 0 | +0.06 ± 0.30 [-0.00, +0.12] | 5 |
| ud_upos | neighbour | 0.5 | 2: unambiguous − ambiguous/rare | 0 | -0.08 ± 0.15 [-0.19, +0.03] | 5 |
| ud_upos | neighbour | 0.5 | 1: R1−R2 overall | 1 | +0.17 ± 0.06 [+0.13, +0.20] | 5 |
| ud_upos | neighbour | 0.5 | 2: unambiguous − ambiguous/rare | 1 | -0.16 ± 0.06 [-0.29, -0.02] | 5 |
| ud_upos | neighbour | 0.5 | 1: R1−R2 overall | 2 | +0.28 ± 0.11 [+0.19, +0.35] | 5 |
| ud_upos | neighbour | 0.5 | 2: unambiguous − ambiguous/rare | 2 | -0.32 ± 0.18 [-0.46, -0.16] | 5 |
| ud_deprel | main | 0.2 | 1: R1−R2 overall | -1 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| ud_deprel | main | 0.2 | 2: unambiguous − ambiguous/rare | -1 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| ud_deprel | main | 0.2 | 1: R1−R2 overall | 0 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| ud_deprel | main | 0.2 | 2: unambiguous − ambiguous/rare | 0 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| ud_deprel | main | 0.2 | 1: R1−R2 overall | 1 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| ud_deprel | main | 0.2 | 2: unambiguous − ambiguous/rare | 1 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| ud_deprel | main | 0.2 | 1: R1−R2 overall | 2 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| ud_deprel | main | 0.2 | 2: unambiguous − ambiguous/rare | 2 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| ud_deprel | main | 0.5 | 1: R1−R2 overall | -1 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| ud_deprel | main | 0.5 | 2: unambiguous − ambiguous/rare | -1 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| ud_deprel | main | 0.5 | 1: R1−R2 overall | 0 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| ud_deprel | main | 0.5 | 2: unambiguous − ambiguous/rare | 0 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| ud_deprel | main | 0.5 | 1: R1−R2 overall | 1 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| ud_deprel | main | 0.5 | 2: unambiguous − ambiguous/rare | 1 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| ud_deprel | main | 0.5 | 1: R1−R2 overall | 2 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| ud_deprel | main | 0.5 | 2: unambiguous − ambiguous/rare | 2 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| ud_deprel | neighbour | 0.2 | 1: R1−R2 overall | -1 | +0.03 ± 0.07 [+0.00, +0.05] | 5 |
| ud_deprel | neighbour | 0.2 | 2: unambiguous − ambiguous/rare | -1 | +0.06 ± 0.07 [+0.01, +0.11] | 5 |
| ud_deprel | neighbour | 0.2 | 1: R1−R2 overall | 0 | +0.09 ± 0.28 [+0.05, +0.14] | 5 |
| ud_deprel | neighbour | 0.2 | 2: unambiguous − ambiguous/rare | 0 | -0.06 ± 0.36 [-0.14, +0.00] | 5 |
| ud_deprel | neighbour | 0.2 | 1: R1−R2 overall | 1 | +0.05 ± 0.19 [+0.01, +0.09] | 5 |
| ud_deprel | neighbour | 0.2 | 2: unambiguous − ambiguous/rare | 1 | +0.05 ± 0.16 [-0.05, +0.12] | 5 |
| ud_deprel | neighbour | 0.2 | 1: R1−R2 overall | 2 | -0.09 ± 0.38 [-0.14, -0.03] | 5 |
| ud_deprel | neighbour | 0.2 | 2: unambiguous − ambiguous/rare | 2 | +0.33 ± 0.58 [+0.15, +0.50] | 5 |
| ud_deprel | neighbour | 0.5 | 1: R1−R2 overall | -1 | +0.03 ± 0.07 [+0.00, +0.05] | 5 |
| ud_deprel | neighbour | 0.5 | 2: unambiguous − ambiguous/rare | -1 | +0.09 ± 0.08 [+0.02, +0.15] | 5 |
| ud_deprel | neighbour | 0.5 | 1: R1−R2 overall | 0 | +0.09 ± 0.28 [+0.05, +0.14] | 5 |
| ud_deprel | neighbour | 0.5 | 2: unambiguous − ambiguous/rare | 0 | -0.06 ± 0.32 [-0.15, +0.01] | 5 |
| ud_deprel | neighbour | 0.5 | 1: R1−R2 overall | 1 | +0.05 ± 0.19 [+0.01, +0.09] | 5 |
| ud_deprel | neighbour | 0.5 | 2: unambiguous − ambiguous/rare | 1 | +0.09 ± 0.19 [-0.01, +0.17] | 5 |
| ud_deprel | neighbour | 0.5 | 1: R1−R2 overall | 2 | -0.09 ± 0.38 [-0.14, -0.03] | 5 |
| ud_deprel | neighbour | 0.5 | 2: unambiguous − ambiguous/rare | 2 | +0.33 ± 0.57 [+0.23, +0.45] | 5 |
| gum_upos | main | 0.2 | 1: R1−R2 overall | -1 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| gum_upos | main | 0.2 | 2: unambiguous − ambiguous/rare | -1 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| gum_upos | main | 0.2 | 1: R1−R2 overall | 0 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| gum_upos | main | 0.2 | 2: unambiguous − ambiguous/rare | 0 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| gum_upos | main | 0.2 | 1: R1−R2 overall | 1 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| gum_upos | main | 0.2 | 2: unambiguous − ambiguous/rare | 1 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| gum_upos | main | 0.2 | 1: R1−R2 overall | 2 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| gum_upos | main | 0.2 | 2: unambiguous − ambiguous/rare | 2 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| gum_upos | main | 0.5 | 1: R1−R2 overall | -1 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| gum_upos | main | 0.5 | 2: unambiguous − ambiguous/rare | -1 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| gum_upos | main | 0.5 | 1: R1−R2 overall | 0 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| gum_upos | main | 0.5 | 2: unambiguous − ambiguous/rare | 0 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| gum_upos | main | 0.5 | 1: R1−R2 overall | 1 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| gum_upos | main | 0.5 | 2: unambiguous − ambiguous/rare | 1 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| gum_upos | main | 0.5 | 1: R1−R2 overall | 2 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| gum_upos | main | 0.5 | 2: unambiguous − ambiguous/rare | 2 | +0.00 ± 0.00 [+0.00, +0.00] | 5 |
| gum_upos | neighbour | 0.2 | 1: R1−R2 overall | -1 | +9.18 ± 13.28 [+8.98, +9.37] | 5 |
| gum_upos | neighbour | 0.2 | 2: unambiguous − ambiguous/rare | -1 | +0.50 ± 1.21 [-0.02, +0.99] | 5 |
| gum_upos | neighbour | 0.2 | 1: R1−R2 overall | 0 | +13.03 ± 20.24 [+12.85, +13.23] | 5 |
| gum_upos | neighbour | 0.2 | 2: unambiguous − ambiguous/rare | 0 | +0.92 ± 2.45 [+0.22, +1.54] | 5 |
| gum_upos | neighbour | 0.2 | 1: R1−R2 overall | 1 | +9.07 ± 13.54 [+8.81, +9.29] | 5 |
| gum_upos | neighbour | 0.2 | 2: unambiguous − ambiguous/rare | 1 | -4.59 ± 8.86 [-5.08, -4.13] | 5 |
| gum_upos | neighbour | 0.2 | 1: R1−R2 overall | 2 | +4.54 ± 4.53 [+4.01, +5.05] | 5 |
| gum_upos | neighbour | 0.2 | 2: unambiguous − ambiguous/rare | 2 | +2.89 ± 5.13 [+2.16, +3.55] | 5 |
| gum_upos | neighbour | 0.5 | 1: R1−R2 overall | -1 | +9.18 ± 13.28 [+8.98, +9.37] | 5 |
| gum_upos | neighbour | 0.5 | 2: unambiguous − ambiguous/rare | -1 | +2.03 ± 3.50 [+1.55, +2.62] | 5 |
| gum_upos | neighbour | 0.5 | 1: R1−R2 overall | 0 | +13.03 ± 20.24 [+12.85, +13.23] | 5 |
| gum_upos | neighbour | 0.5 | 2: unambiguous − ambiguous/rare | 0 | +1.45 ± 2.66 [+0.63, +2.18] | 5 |
| gum_upos | neighbour | 0.5 | 1: R1−R2 overall | 1 | +9.07 ± 13.54 [+8.81, +9.29] | 5 |
| gum_upos | neighbour | 0.5 | 2: unambiguous − ambiguous/rare | 1 | -2.61 ± 5.47 [-3.27, -1.95] | 5 |
| gum_upos | neighbour | 0.5 | 1: R1−R2 overall | 2 | +4.54 ± 4.53 [+4.01, +5.05] | 5 |
| gum_upos | neighbour | 0.5 | 2: unambiguous − ambiguous/rare | 2 | +3.16 ± 4.93 [+2.46, +3.83] | 5 |

Tertile −1 means all tokens; 0/1/2 are low/medium/high training-defined label shift. Ties in training label scores can leave strata empty. Positive interactions within supported strata are consistent with H_annot; attenuation after stratification is compatible with H_shift. A zero-crossing interval does not establish equivalence. Comparison 1 alone distinguishes neither causal explanation.

**ud_upos, comparison 2:** H_annot-consistent positive interactions appear in some strata, with mixed evidence across strata. Positive / negative intervals: 2 / 0 of 6 supported threshold×tertile cells; thresholds reuse predictions.

**ud_deprel, comparison 2:** Neither explanation is established by the within-tertile intervals. Positive / negative intervals: 0 / 0 of 6 supported threshold×tertile cells; thresholds reuse predictions.

**gum_upos, comparison 2:** Neither explanation is established by the within-tertile intervals. Positive / negative intervals: 0 / 0 of 6 supported threshold×tertile cells; thresholds reuse predictions.

Deviations from the initial request are the user’s cost revision: Stage 1 only; no GUM discourse or routing; pilot-fixed λ plus one neighbour; 600-step maximum with early stopping; capped, batch-resampled MMD; entropy thresholds applied only in evaluation. The RBF bandwidth uses the capped, genre-stratified batch sample. No linear-kernel fallback was used.

See [implementation](IMPLEMENTATION_3.md), [tables](tables/invariance/), and [figures](figures/invariance/). Stage 2 remains deferred.
