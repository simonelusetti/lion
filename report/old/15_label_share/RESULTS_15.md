# Label-aligned share of contextual variation, per layer

Frozen bert-base-cased word vectors (mean of WordPieces) for every word of the full native training set, layers 0–12; plus the mixed vector of the three full-data mixture heads from the coverage run (fraction 1.0, seeds 0–2), whose numbers are computed per seed and then averaged. Nothing is trained.

Labels: CoNLL BIO tags collapsed to entity type plus O; EWT base dependency relations. Eligible word types: exact case-sensitive forms with at least two labels that each occur ≥10 times in training; a type keeps only the occurrences of those labels. Per type w: T_w = mean squared distance of occurrences from the type mean; B_w = Σ_y p(y|w)·‖μ_wy − μ_w‖²; share = B_w/T_w. Baseline: share with labels shuffled within the type (label counts kept), averaged over 20 shuffles. Types are aggregated weighted by occurrences. Total spread is divided by the mean squared distance between 20000 random training-word pairs from different sentences with different forms, at the same layer. "Standardized" repeats the computation after z-scoring each dimension over all training words of that layer.

Measure: measure 1.0 min; wall-clock 2.48 h.

## conll2003

53 eligible word types, 33918 occurrences.

| View | Total spread / random pairs | Share | Permutation baseline | Share − baseline | Standardized share | Standardized share − baseline |
| --- | --- | --- | --- | --- | --- | --- |
| layer0 | 0.0456 | 0.0117 | 0.0017 | 0.0100 | 0.0116 | 0.0099 |
| layer1 | 0.0541 | 0.0141 | 0.0017 | 0.0124 | 0.0141 | 0.0123 |
| layer2 | 0.0820 | 0.0193 | 0.0017 | 0.0176 | 0.0193 | 0.0176 |
| layer3 | 0.1369 | 0.0206 | 0.0017 | 0.0189 | 0.0207 | 0.0190 |
| layer4 (peak) (standardized peak) | 0.1857 | 0.0215 | 0.0017 | 0.0198 | 0.0216 | 0.0199 |
| layer5 | 0.2116 | 0.0212 | 0.0017 | 0.0195 | 0.0212 | 0.0194 |
| layer6 | 0.2571 | 0.0212 | 0.0017 | 0.0195 | 0.0211 | 0.0194 |
| layer7 | 0.3194 | 0.0206 | 0.0017 | 0.0189 | 0.0205 | 0.0188 |
| layer8 | 0.3511 | 0.0202 | 0.0017 | 0.0185 | 0.0200 | 0.0183 |
| layer9 | 0.3697 | 0.0194 | 0.0017 | 0.0177 | 0.0192 | 0.0175 |
| layer10 | 0.3877 | 0.0187 | 0.0017 | 0.0170 | 0.0184 | 0.0167 |
| layer11 | 0.3882 | 0.0183 | 0.0017 | 0.0166 | 0.0180 | 0.0162 |
| layer12 | 0.3310 | 0.0199 | 0.0017 | 0.0181 | 0.0198 | 0.0181 |
| mixture (mean ± SD of 3 seeds) | 0.2231 ± 0.0031 | 0.0210 ± 0.0000 | 0.0017 ± 0.0000 | 0.0193 ± 0.0000 | 0.0208 ± 0.0000 | 0.0191 ± 0.0000 |

Share − baseline peaks at **layer4** (raw) and **layer4** (standardized) among layers 0–12; the mixture is 0.0193 (raw) and 0.0191 (standardized).

## en_ewt_deprel

335 eligible word types, 70151 occurrences.

| View | Total spread / random pairs | Share | Permutation baseline | Share − baseline | Standardized share | Standardized share − baseline |
| --- | --- | --- | --- | --- | --- | --- |
| layer0 | 0.0431 | 0.0294 | 0.0098 | 0.0196 | 0.0291 | 0.0192 |
| layer1 | 0.0628 | 0.0372 | 0.0098 | 0.0274 | 0.0369 | 0.0271 |
| layer2 | 0.1066 | 0.0880 | 0.0099 | 0.0781 | 0.0875 | 0.0776 |
| layer3 | 0.1605 | 0.1184 | 0.0099 | 0.1085 | 0.1185 | 0.1086 |
| layer4 (peak) (standardized peak) | 0.1958 | 0.1276 | 0.0099 | 0.1177 | 0.1276 | 0.1177 |
| layer5 | 0.2222 | 0.1248 | 0.0099 | 0.1149 | 0.1247 | 0.1147 |
| layer6 | 0.2709 | 0.1162 | 0.0099 | 0.1063 | 0.1157 | 0.1058 |
| layer7 | 0.3123 | 0.1099 | 0.0099 | 0.1000 | 0.1094 | 0.0995 |
| layer8 | 0.3389 | 0.1002 | 0.0099 | 0.0903 | 0.0996 | 0.0897 |
| layer9 | 0.3571 | 0.0890 | 0.0099 | 0.0791 | 0.0883 | 0.0784 |
| layer10 | 0.3761 | 0.0763 | 0.0099 | 0.0664 | 0.0756 | 0.0657 |
| layer11 | 0.3711 | 0.0721 | 0.0099 | 0.0622 | 0.0715 | 0.0616 |
| layer12 | 0.3232 | 0.0689 | 0.0099 | 0.0590 | 0.0683 | 0.0584 |
| mixture (mean ± SD of 3 seeds) | 0.2612 ± 0.0034 | 0.1072 ± 0.0002 | 0.0099 ± 0.0000 | 0.0973 ± 0.0002 | 0.1065 ± 0.0002 | 0.0966 ± 0.0002 |

Types with no spread at a view (identical occurrences, share undefined) are left out of that view's aggregates: layer0: 1 raw / 1 standardized.

Share − baseline peaks at **layer4** (raw) and **layer4** (standardized) among layers 0–12; the mixture is 0.0973 (raw) and 0.0966 (standardized).

Per-word-type values for every view, with the most frequent types flagged: [per_type.csv](tables/label_share/per_type.csv). Raw distances and pair references: [summary.csv](tables/label_share/summary.csv).

[Implementation](IMPLEMENTATION_15.md) · [CSV tables](tables/label_share/) · [Previous: coverage](old/14_coverage/RESULTS_14.md)
