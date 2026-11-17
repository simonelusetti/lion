# Context partitions: X0–X4 results

Implementation and fixed analysis choices: [IMPLEMENTATION_2.md](IMPLEMENTATION_2.md). Entries show mean ± classifier-seed SD [95% document-block bootstrap interval]. Seeds are 0/1/2; bootstrap intervals condition on the fitted models. Only the three X1 comparisons are confirmatory; the other measurements are descriptive. This is not a test of causal identification.

## X0 — amortized readout

| Dataset | Conditions within all tolerances | Conditions |
|---|---|---|
| ud_upos | 0 | 19 |
| ud_deprel | 0 | 19 |
| ud_discourse | 0 | 19 |
| gum_upos | 0 | 19 |
| ner_shared | 0 | 13 |

Operational tolerances are upper bootstrap bounds ≤1 pp for accuracy/F1 loss and ≤0.02 nats KL. Passing some conditions does not justify replacing all dedicated readouts. All subsequent analyses retain dedicated readouts.

## X1 — confirmatory comparisons

### Matched label information

| Dataset | Paired comparison | Nuisance difference (pp) | Assessment |
|---|---|---|---|
| ud_upos | linear_minus_random | undefined ± undefined [0.203, 0.317] | opposite direction |
| ud_upos | tree_minus_random | undefined ± undefined [0.221, 0.378] | opposite direction |
| ud_upos | tree_minus_linear | undefined ± undefined [0.018, 0.061] | opposite direction |
| ud_deprel | linear_minus_random | -1.118 ± 0.941 [-1.942, 0.784] | inconclusive |
| ud_deprel | tree_minus_random | -0.147 ± 0.653 [-1.480, 1.269] | inconclusive |
| ud_deprel | tree_minus_linear | 0.972 ± 0.324 [-1.099, 1.857] | inconclusive |
| ud_discourse | linear_minus_random | -0.540 ± 1.146 [-1.330, 0.154] | inconclusive |
| ud_discourse | tree_minus_random | 0.138 ± 0.404 [-0.303, 0.490] | inconclusive |
| ud_discourse | tree_minus_linear | 0.678 ± 0.833 [-0.155, 1.460] | inconclusive |
| gum_upos | linear_minus_random | 0.037 ± 0.039 [-0.098, 0.455] | inconclusive |
| gum_upos | tree_minus_random | 0.099 ± 0.061 [-0.277, 0.383] | inconclusive |
| gum_upos | tree_minus_linear | 0.062 ± 0.092 [-0.549, 0.361] | inconclusive |
| ner_shared | linear_minus_random | 0.159 ± 0.223 [-0.652, 0.542] | inconclusive |

Negative differences favour the first family. Undefined values occur when full context has no positive accuracy gain over word-only, or an interpolation is not defined. Overlapping intervals do not establish N or equivalence. Interpolation uses the first crossing in the prespecified family order; random order is mean retained count.

### Knees

| Dataset | Family | Conditions meeting the knee in all three seeds |
|---|---|---|
| ud_upos | linear | none / undefined |
| ud_upos | tree | none / undefined |
| ud_upos | random | none / undefined |
| ud_deprel | linear | none / undefined |
| ud_deprel | tree | none / undefined |
| ud_deprel | random | none / undefined |
| ud_discourse | linear | none / undefined |
| ud_discourse | tree | none / undefined |
| ud_discourse | random | none / undefined |
| gum_upos | linear | none / undefined |
| gum_upos | tree | none / undefined |
| gum_upos | random | none / undefined |
| ner_shared | linear | none / undefined |
| ner_shared | random | none / undefined |

The knee requires ≥90% of full-minus-word-only label gain and ≤50% of the positive full-minus-G predictor-nuisance excess. Seed agreement is a point estimate; stability intervals and undefined cases remain in the CSV. A window knee is consistent with locality, not sufficient to distinguish L from S.

### Domain shift

| Dataset | Family | Empirical finite peak | Peak minus full accuracy (pp) | Assessment |
|---|---|---|---|---|
| ud_upos | linear | lin1 | 1.986 [0.411, 3.560] | supports finite-window advantage |
| ud_upos | tree | tree1 | 3.908 [2.403, 5.412] | supports finite-window advantage |
| ud_deprel | linear | lin1 | -0.590 [-1.956, 0.776] | inconclusive / no finite-window advantage |
| ud_deprel | tree | tree1 | 2.866 [1.057, 4.676] | supports finite-window advantage |
| ud_discourse | linear | lin8 | -0.197 [-0.731, 0.338] | inconclusive / no finite-window advantage |
| ud_discourse | tree | tree3 | 0.247 [-0.232, 0.726] | inconclusive / no finite-window advantage |
| gum_upos | linear | lin1 | 1.836 [1.177, 2.496] | supports finite-window advantage |
| gum_upos | tree | tree1 | 1.978 [1.321, 2.635] | supports finite-window advantage |
| ner_shared | linear | lin8 | -0.133 [-1.057, 0.791] | inconclusive / no finite-window advantage |

Intervals use the maximum bootstrap deviation across finite radii within each family. They account for choosing the displayed peak within that family, but are not simultaneous across datasets or all three primary questions. Source/target curves average ordered domain pairs equally. Small GUM native test document counts still limit Y shift inference.

## X2 — token and band attribution (descriptive)

| Dataset | Quantity | Estimate |
|---|---|---|
| ud_upos | pairwise_pearson | -0.090 ± 0.035 [-0.168, 0.009] |
| ud_upos | top20_jaccard | 0.282 ± 0.014 [0.250, 0.311] |
| ud_upos | top20_excess_over_random | 0.009 ± 0.014 [-0.013, 0.033] |
| ud_deprel | pairwise_pearson | -0.184 ± 0.022 [-0.265, -0.042] |
| ud_deprel | top20_jaccard | 0.324 ± 0.024 [0.280, 0.361] |
| ud_deprel | top20_excess_over_random | 0.015 ± 0.024 [-0.017, 0.045] |
| ud_discourse | pairwise_pearson | 0.058 ± 0.045 [0.015, 0.099] |
| ud_discourse | top20_jaccard | 0.260 ± 0.007 [0.239, 0.283] |
| ud_discourse | top20_excess_over_random | 0.038 ± 0.007 [0.024, 0.049] |
| gum_upos | pairwise_pearson | 0.043 ± 0.011 [-0.015, 0.095] |
| gum_upos | top20_jaccard | 0.239 ± 0.007 [0.205, 0.268] |
| gum_upos | top20_excess_over_random | 0.021 ± 0.007 [0.004, 0.035] |
| ner_shared | pairwise_pearson | 0.035 ± 0.015 [-0.004, 0.075] |
| ner_shared | top20_jaccard | 0.234 ± 0.007 [0.211, 0.256] |
| ner_shared | top20_excess_over_random | 0.033 ± 0.007 [0.013, 0.052] |

| Dataset | Word-class minus distance partial R² for nuisance | Assessment |
|---|---|---|
| ud_upos | 0.000 ± 0.000 [-0.009, 0.002] | inconclusive |
| ud_deprel | -0.002 ± 0.001 [-0.023, 0.003] | inconclusive |
| ud_discourse | -0.004 ± 0.003 [-0.010, -0.001] | distance stronger |
| gum_upos | -0.005 ± 0.001 [-0.014, -0.001] | distance stronger |

The nuisance score uses the conditional representation Z probe, not the predictor-log-probability probe. Positive r/n means removal reduces true-label confidence; negative scores are retained. Overlap above the exact random-set Jaccard baseline and positive r–n association are consistent with shared token channels (N), but cannot establish that no other partition exists. Bands and matched-count random removals address redundancy; their complete effects are in the band CSV. All partial-effect coefficients, group partial R², support and missing-n counts are linked below. Shared NER lacks gold trees/UPOS, so its regression omits those predictors.

## X3 — gold label context (descriptive)

| Dataset | BA gain at k=2 minus gain at k=0 (pp) | Assessment |
|---|---|---|
| ud_upos | -3.146 ± 1.527 [-4.272, -2.287] | consistent with A |
| ud_deprel | -2.045 ± 1.029 [-3.332, -1.010] | consistent with A |
| ud_discourse | -1.199 ± 0.132 [-1.782, -0.375] | consistent with A |
| gum_upos | -0.998 ± 0.092 [-1.513, -0.414] | consistent with A |
| ner_shared | -6.759 ± 5.697 [-7.378, -5.974] | consistent with A |

A shrinking gain is consistent with label context absorbing information in the word-only proxy. The probes are linear on position-specific one-hot labels; failure to shrink does not rule out nonlinear label interactions. The word-only proxy also carries lexical genre information. Absolute A/B scores and cross-entropy gains are in the table CSV.

## X4 — metadata-free clustering (descriptive)

| Dataset | Visible channel | K | NMI | Matched Z BA (%) |
|---|---|---|---|---|
| ud_upos | outside | 5 | 0.032 ± 0.006 [0.031, 0.053] | 25.365 ± 0.830 [23.278, 27.499] |
| ud_upos | inside | 5 | 0.019 ± 0.001 [0.019, 0.034] | 24.438 ± 0.325 [23.245, 26.051] |
| ud_upos | outside | 10 | 0.080 ± 0.007 [0.081, 0.109] | 31.394 ± 0.754 [29.714, 33.521] |
| ud_upos | inside | 10 | 0.022 ± 0.000 [0.025, 0.039] | 24.712 ± 0.351 [23.545, 26.462] |
| ud_deprel | outside | 5 | 0.032 ± 0.006 [0.031, 0.053] | 25.365 ± 0.830 [23.278, 27.499] |
| ud_deprel | inside | 5 | 0.019 ± 0.001 [0.019, 0.034] | 24.430 ± 0.339 [23.238, 26.035] |
| ud_deprel | outside | 10 | 0.080 ± 0.007 [0.081, 0.109] | 31.399 ± 0.750 [29.716, 33.521] |
| ud_deprel | inside | 10 | 0.022 ± 0.000 [0.025, 0.039] | 24.707 ± 0.355 [23.540, 26.453] |
| ud_discourse | outside | 15 | 0.148 ± 0.004 [0.158, 0.180] | 17.537 ± 0.750 [16.296, 18.747] |
| ud_discourse | inside | 15 | 0.053 ± 0.000 [0.060, 0.074] | 10.430 ± 0.301 [9.885, 11.125] |
| ud_discourse | outside | 30 | 0.189 ± 0.002 [0.203, 0.228] | 23.031 ± 0.478 [21.903, 24.373] |
| ud_discourse | inside | 30 | 0.068 ± 0.001 [0.080, 0.095] | 11.189 ± 0.268 [10.585, 11.872] |
| gum_upos | outside | 15 | 0.148 ± 0.004 [0.158, 0.180] | 17.506 ± 0.609 [16.306, 18.624] |
| gum_upos | inside | 15 | 0.053 ± 0.000 [0.060, 0.074] | 10.406 ± 0.295 [9.852, 11.091] |
| gum_upos | outside | 30 | 0.189 ± 0.003 [0.203, 0.228] | 23.052 ± 0.546 [21.894, 24.404] |
| gum_upos | inside | 30 | 0.068 ± 0.001 [0.080, 0.095] | 11.145 ± 0.264 [10.523, 11.797] |
| ner_shared | outside | 2 | 0.075 ± 0.005 [0.065, 0.086] | 62.543 ± 1.171 [61.575, 63.533] |
| ner_shared | inside | 2 | 0.033 ± 0.000 [0.026, 0.042] | 60.576 ± 0.043 [59.376, 61.730] |
| ner_shared | outside | 4 | 0.058 ± 0.001 [0.052, 0.067] | 60.644 ± 0.548 [59.808, 61.481] |
| ner_shared | inside | 4 | 0.047 ± 0.003 [0.042, 0.055] | 59.569 ± 0.704 [58.527, 60.678] |

Clustering never uses labels; cluster-to-domain matching uses training-fold Z only. Scores use held-out documents, and NMI is computed within fold before count-weighted aggregation because cluster IDs differ between folds. Per-label scores are in the same CSV. Nonempty inside/outside pooling is required for both channels, preserving paired coverage.

## Protocol differences and limits

- GUM UPOS reuses precisely the GUM discourse focal positions; it does not enlarge that sample.
- Identical word-only/full endpoints are canonical shared conditions; random controls match each focal's window count.
- Gold UD heads/UPOS are used. No offline parser is installed, so shared NER uses only linear geometry and sentence blocks; its document-level claims are exploratory.
- Z probes use five document folds on pooled native samples, with a disjoint validation fold for checkpoint selection. X4 uses four training folds; Y readouts keep native splits.
- X2 uses three fixed independent sparse-occlusion draws per focal, sampled uniformly during training, rather than re-encoding new masks every epoch.
- The geometry probe uses WordPiece sentence length, first-WordPiece focal position and retained context-word count. Only those scalar inputs are standardized using training-fold statistics.
- X3 uses linear probes; PAD and unlabelled neighbors are separate one-hot categories. Best matching in X4 is training-only balanced Hungarian matching with capacity ceil(clusters/domains) per domain.
- Bootstrap intervals use 200 domain-stratified document-block resamples of predictions, not refits of every classifier; regressions are refitted from resampled document sufficient statistics. Undefined bootstrap cells are omitted and valid counts are recorded.
- X0 tolerances, first-crossing interpolation, nonpositive-gain exclusions, knee thresholds and simultaneous finite-radius shift bands were fixed before the run. These do not constitute a familywise causal test.

## Tables and figures

All aggregate CSVs: [tables/partition](tables/). Figures: [figures/partition](figures/). Per-model checkpoints, fold predictions, raw paired attribution tensors and cached representations remain under results/partition and data/cache/partition; no old E1–E7 result is overwritten.
