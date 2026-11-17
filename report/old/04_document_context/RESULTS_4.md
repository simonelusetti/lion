# Document-context nuisance penalty: results

Wall-clock: **1.83 hours**, 45 fits. Includes context preparation, calibration, training, probes and analysis; queue time is excluded.

The penalty and residualizer for Rz use no genre labels. Genre labels still define this study’s rotations and balanced batches, the Rg comparator, diagnostic probes, and validation policy (b). Policy (a) uses native in-domain validation. Residualization removes a fitted linear label contribution and does not prove the remaining context is exclusively nuisance.

Differences below are percentage points: mean ± seed SD [95% paired document-block bootstrap interval]. Rotations are weighted equally. Probe intervals condition on fitted probes and classifier/λ choices; probes are not refitted during bootstrap.

| Task | Validation | Primary comparison | Difference | Rotations |
|---|---|---|---:|---:|
| ud_upos | id | Rz_minus_R0 genre_dependence | +0.09 ± 0.06 [-0.08, +0.26] | 5 |
| ud_upos | id | Rz_minus_R0 ood_accuracy | +0.02 ± 0.06 [-0.00, +0.03] | 5 |
| ud_upos | id | Rz_minus_Rrand ood_accuracy | -0.03 ± 0.06 [-0.05, -0.01] | 5 |
| ud_upos | id | Rz_minus_Rg ood_accuracy | -0.05 ± 0.04 [-0.07, -0.03] | 5 |
| ud_upos | genre | Rz_minus_R0 genre_dependence | +0.05 ± 0.08 [-0.09, +0.16] | 5 |
| ud_upos | genre | Rz_minus_R0 ood_accuracy | +0.01 ± 0.01 [-0.00, +0.02] | 5 |
| ud_upos | genre | Rz_minus_Rrand ood_accuracy | -0.00 ± 0.00 [-0.01, +0.01] | 5 |
| ud_upos | genre | Rz_minus_Rg ood_accuracy | +0.00 ± 0.02 [-0.01, +0.02] | 5 |
| ud_deprel | id | Rz_minus_R0 genre_dependence | -0.01 ± 0.09 [-0.09, +0.08] | 5 |
| ud_deprel | id | Rz_minus_R0 ood_accuracy | +0.03 ± 0.06 [+0.02, +0.05] | 5 |
| ud_deprel | id | Rz_minus_Rrand ood_accuracy | +0.03 ± 0.07 [+0.02, +0.05] | 5 |
| ud_deprel | id | Rz_minus_Rg ood_accuracy | +0.01 ± 0.03 [-0.00, +0.03] | 5 |
| ud_deprel | genre | Rz_minus_R0 genre_dependence | -0.00 ± 0.03 [-0.09, +0.10] | 5 |
| ud_deprel | genre | Rz_minus_R0 ood_accuracy | +0.00 ± 0.02 [-0.01, +0.01] | 5 |
| ud_deprel | genre | Rz_minus_Rrand ood_accuracy | -0.01 ± 0.02 [-0.02, +0.01] | 5 |
| ud_deprel | genre | Rz_minus_Rg ood_accuracy | +0.01 ± 0.01 [-0.01, +0.02] | 5 |
| gum_upos | id | Rz_minus_R0 genre_dependence | -0.05 ± 0.20 [-0.24, +0.13] | 5 |
| gum_upos | id | Rz_minus_R0 ood_accuracy | +0.09 ± 0.09 [+0.06, +0.12] | 5 |
| gum_upos | id | Rz_minus_Rrand ood_accuracy | -0.00 ± 0.02 [-0.02, +0.01] | 5 |
| gum_upos | id | Rz_minus_Rg ood_accuracy | -0.01 ± 0.04 [-0.04, +0.02] | 5 |
| gum_upos | genre | Rz_minus_R0 genre_dependence | -0.01 ± 0.09 [-0.17, +0.12] | 5 |
| gum_upos | genre | Rz_minus_R0 ood_accuracy | +0.08 ± 0.05 [+0.04, +0.11] | 5 |
| gum_upos | genre | Rz_minus_Rrand ood_accuracy | +0.01 ± 0.05 [-0.01, +0.04] | 5 |
| gum_upos | genre | Rz_minus_Rg ood_accuracy | +0.08 ± 0.12 [+0.05, +0.11] | 5 |

Primary 1 favours Rz when genre-dependence differences are negative. Primary 2 favours Rz when OOD accuracy differences are positive. Zero-crossing intervals remain inconclusive.

| Task | Validation | Primary 3: Spearman ρ | Seed SD | 95% CI | Models |
|---|---|---:|---:|---|---:|
| ud_upos | id | -0.062 | 0.165 | [-0.150, +0.067] | 195 |
| ud_upos | genre | -0.074 | 0.180 | [-0.157, +0.070] | 195 |
| ud_deprel | id | +0.089 | 0.053 | [+0.015, +0.254] | 195 |
| ud_deprel | genre | +0.097 | 0.062 | [+0.006, +0.253] | 195 |
| gum_upos | id | +0.174 | 0.148 | [-0.022, +0.286] | 195 |
| gum_upos | genre | +0.215 | 0.132 | [+0.010, +0.289] | 195 |
| all_tasks | id | +0.148 | 0.042 | [+0.125, +0.216] | 585 |
| all_tasks | genre | +0.149 | 0.046 | [+0.125, +0.214] | 585 |

Rank correlations include every rule/λ candidate and one shared λ=0 baseline per rotation/seed, pooled across tasks and separately per task, for each checkpoint policy. Genre chance levels and label support differ across tasks, so the pooled correlation needs the per-task checks. Positive correlation means lower Z-free violation tracks lower genre dependence; it does not establish a causal relationship. The two checkpoint policies share optimization trajectories and are not independent replications.

| Task | Training coverage | ID-test coverage |
|---|---:|---:|
| ud_upos | 99.9% | 98.4% |
| ud_deprel | 99.9% | 98.4% |
| gum_upos | 100.0% | 100.0% |

Deviations/operational choices: GUM has only two native ID-test documents per genre, so probes use two document folds (EWT uses five), with separate native ID-validation documents for probe checkpoint selection. Balanced accuracy averages eligible genres within labels, then labels equally; support thresholds are the E3 defaults. ID bootstrap resamples documents within genres. PCA/ridge use only labelled native-training tokens from training genres; context means use all cached words in other sentences within the same native partition. Ridge strength is .001 with an unpenalized intercept. A fixed donor-document derangement with random donor-token matching defines Rrand. Pilot calibration uses a 100-step R0 trajectory and 12 batches; its shared four-value logarithmic grid brackets the rules’ median CE/penalty ratios with its middle values. See calibration.csv for actual penalty/CE ratios. No training-token or MMD subsampling is used.

Descriptive OOD macro-F1, in-domain accuracy, violation, λ selections, probe support and all candidate measurements are in [tables](tables/zfree/). [Figures](figures/zfree/); [implementation](IMPLEMENTATION_4.md).
