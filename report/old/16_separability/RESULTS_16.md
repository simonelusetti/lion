# Per-word linear separability and cross-word alignment, per layer

Same eligible word types and occurrences as the label-share run (full native training sets; forms with ≥2 labels each ≥10 times; only those labels kept; CoNLL tags collapsed to entity type plus O; EWT base relations). Vectors: frozen bert-base-cased layers 0–12 and the mixed vectors of the three full-data mixture heads (coverage run, fraction 1.0); mixture numbers are per seed, then mean ± SD. Every view is z-scored per dimension over all training words before fitting, so one L2 grid applies to all layers. Nothing is trained except these probes.

- **Per-word separability**: for each type, a multinomial logistic regression on that type's occurrences only, 5-fold stratified cross-validation; every occurrence is predicted once by a model that did not see it. Balanced accuracy (mean per-label recall) per type, aggregated weighted by occurrences.
- **d′**: held-out separation-to-width along each fold's own dividing direction (w_a − w_b, unit length) between the type's two most frequent labels: |mean_a − mean_b| / √((var_a + var_b)/2) of held-out projections.
- **Cross-word transfer**: one classifier over all labels, trained on the occurrences of the other word types (5 folds of types), applied to each held-out type with the argmax restricted to that type's own labels.
- **Transfer / per-word (above chance)**: (transfer − chance) / (per-word − chance), chance = 1 / number of the type's labels, each aggregated over types.
- **Direction cosine across words**: for every label pair shared by ≥2 types, the cosine between different types' label-mean difference vectors (z-scored space), averaged over all type pairs.

L2 grid [0.001, 0.01, 0.1]; per view, the per-word and transfer columns each use the value with the highest aggregate held-out balanced accuracy (all values in the CSV). Measure 41.3 min; wall-clock 0.69 h.

## conll2003

53 eligible word types, 33918 occurrences. ▲ = peak among layers 0–12.

| View | Chance | Per-word held-out bal. acc. | Held-out d′ (top-2 labels) | Cross-word transfer bal. acc. | Transfer / per-word (above chance) | Direction cosine across words | L2 per-word / transfer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| layer0 | 0.4729 | 0.4873 | 0.344 | 0.4766 | 0.256 | 0.203 ▲ | 0.01 / 0.001 |
| layer1 | 0.4729 | 0.6426 | 2.558 | 0.5856 | 0.664 | 0.150 | 0.001 / 0.001 |
| layer2 | 0.4729 | 0.7161 | 3.434 | 0.6398 | 0.686 | 0.149 | 0.001 / 0.001 |
| layer3 | 0.4729 | 0.7484 | 3.728 | 0.6680 | 0.708 | 0.136 | 0.001 / 0.001 |
| layer4 | 0.4729 | 0.7862 | 3.888 | 0.6918 | 0.699 | 0.145 | 0.001 / 0.001 |
| layer5 | 0.4729 | 0.7764 | 4.286 | 0.7078 | 0.774 | 0.140 | 0.001 / 0.001 |
| layer6 | 0.4729 | 0.8206 | 4.634 | 0.7314 | 0.744 | 0.155 | 0.001 / 0.001 |
| layer7 | 0.4729 | 0.8198 | 4.726 | 0.7593 | 0.825 | 0.150 | 0.001 / 0.001 |
| layer8 | 0.4729 | 0.8310 ▲ | 4.621 | 0.7736 | 0.840 | 0.150 | 0.001 / 0.001 |
| layer9 | 0.4729 | 0.8269 | 4.710 | 0.7612 | 0.814 | 0.150 | 0.001 / 0.001 |
| layer10 | 0.4729 | 0.8092 | 4.762 ▲ | 0.7766 ▲ | 0.903 ▲ | 0.153 | 0.001 / 0.001 |
| layer11 | 0.4729 | 0.7995 | 4.712 | 0.7615 | 0.884 | 0.145 | 0.001 / 0.001 |
| layer12 | 0.4729 | 0.7923 | 4.138 | 0.7534 | 0.878 | 0.124 | 0.001 / 0.001 |
| mixture (3 seeds) | 0.4729 | 0.8207 ± 0.0000 | 4.959 ± 0.011 | 0.7511 ± 0.0076 | 0.800 ± 0.022 | 0.165 ± 0.000 | — |

Peaks: Per-word held-out bal. acc.: layer8; Held-out d′ (top-2 labels): layer10; Cross-word transfer bal. acc.: layer10; Transfer / per-word (above chance): layer10; Direction cosine across words: layer0.

## en_ewt_deprel

335 eligible word types, 70151 occurrences. ▲ = peak among layers 0–12.

| View | Chance | Per-word held-out bal. acc. | Held-out d′ (top-2 labels) | Cross-word transfer bal. acc. | Transfer / per-word (above chance) | Direction cosine across words | L2 per-word / transfer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| layer0 | 0.2908 | 0.3167 | 0.418 | 0.3054 | 0.566 | 0.317 ▲ | 0.001 / 0.01 |
| layer1 | 0.2908 | 0.5882 | 2.300 | 0.3991 | 0.364 | 0.292 | 0.001 / 0.001 |
| layer2 | 0.2908 | 0.7562 | 3.697 | 0.5595 | 0.577 | 0.277 | 0.001 / 0.001 |
| layer3 | 0.2908 | 0.7775 | 4.261 | 0.6142 | 0.665 | 0.239 | 0.001 / 0.001 |
| layer4 | 0.2908 | 0.7983 | 4.583 | 0.6678 | 0.743 | 0.236 | 0.001 / 0.001 |
| layer5 | 0.2908 | 0.8128 | 4.826 | 0.6920 | 0.769 | 0.246 | 0.001 / 0.001 |
| layer6 | 0.2908 | 0.8343 | 5.188 | 0.7424 | 0.831 | 0.259 | 0.001 / 0.001 |
| layer7 | 0.2908 | 0.8377 | 5.584 | 0.7546 | 0.848 | 0.240 | 0.001 / 0.001 |
| layer8 | 0.2908 | 0.8433 ▲ | 5.606 ▲ | 0.7554 ▲ | 0.841 | 0.237 | 0.001 / 0.001 |
| layer9 | 0.2908 | 0.8314 | 5.495 | 0.7494 | 0.848 ▲ | 0.218 | 0.001 / 0.001 |
| layer10 | 0.2908 | 0.8157 | 5.272 | 0.7341 | 0.844 | 0.197 | 0.001 / 0.001 |
| layer11 | 0.2908 | 0.7999 | 5.063 | 0.7114 | 0.826 | 0.182 | 0.001 / 0.001 |
| layer12 | 0.2908 | 0.7881 | 4.699 | 0.6707 | 0.764 | 0.143 | 0.001 / 0.001 |
| mixture (3 seeds) | 0.2908 | 0.8419 ± 0.0003 | 5.670 ± 0.003 | 0.7521 ± 0.0009 | 0.837 ± 0.002 | 0.260 ± 0.001 | — |

Peaks: Per-word held-out bal. acc.: layer8; Held-out d′ (top-2 labels): layer8; Cross-word transfer bal. acc.: layer8; Transfer / per-word (above chance): layer9; Direction cosine across words: layer0.

Per-type values: [per_type.csv](tables/separability/per_type.csv) (20 most frequent types flagged); per label pair cosines: [label_pairs.csv](tables/separability/label_pairs.csv); all L2 values: [summary.csv](tables/separability/summary.csv).

[Implementation](IMPLEMENTATION_16.md) · [CSV tables](tables/separability/) · [Previous: label-aligned share](old/15_label_share/RESULTS_15.md)
