# Pilot: context-changing, focal-label-changing augmentation

CoNLL-2003 pilot: 2400 fit sentences, 600 held-out training sentences for calibration. The scorer and both arms use exactly the same original fit sentences; calibration sentences are excluded from every fitted model. Native dev selects checkpoints. All CoNLL/WNUT/WikiANN evaluations use untouched raw test inputs and the existing span/label-mapping code.

Frozen bert-base-cased block 12, existing MLP and ten-epoch recipe, seeds 0/1/2 for the comparison. Identical initialization, original batches/order, original label presentations and optimizer steps in both arms. The augmented arm appends focal-only examples in their observed dataset proportion; extra labelled-token counts are recorded in budgets.csv. Mean loss is over labelled tokens; original gold labels are unchanged.

One proposal per fit focal: uniformly choose a valid neighbour at word distance 1–5, mask all pieces of that neighbour and sample once with the existing one-pass MLM fill implementation. Piece counts and focal pieces are preserved. Multi-piece neighbour fills use independent piece samples from the jointly masked pass; this is an approximate word sampler. Accept only an actual case-sensitive word change, a different native BIO prediction and confidence ≥0.9. No retries or learned selectors.

Generation: **89/34,678 accepted (0.257%)**. Of these, 65 change entity type/O and 24 change only B/I. [Label transitions](tables/label_change/label_transitions.csv).

Scorer calibration accuracy 97.22%; overall error 2.77%; high-confidence share 92.25%; conditional error among high-confidence predictions 0.41%.

**Conditional bounds:** e_U=0.006178, a_L=0.001915, ideal-sampler π_L=0.0000; δ_max=-0.006082 for desired precision .95. Joint confidence at least .95 (one-sided .975 Clopper–Pearson bounds for each quantity). 

The two calibration streams independently sample uniform focal slots with replacement from the same fixed held-out corpus. Bounds are exact for those empirical sampling risks conditional on that corpus; repeated slots are not extra evidence about a broader population. δ_Q is unknown and must cover the relevant joint distribution of text and true modified labels, including any reference-to-generator shift. A text-marginal discrepancy alone is insufficient. **No unconditional certification is claimed, and δ_Q=0 is not established.** Positive δ_max describes allowable discrepancy under the stated inequality, not a measured discrepancy.

| Test | Label space | Baseline span F1 % ± SD | Augmented span F1 % ± SD | Paired Δ percentage points ± SD |
|---|---|---|---|---|
| conll2003 | native | 80.60 ± 0.55 | 80.77 ± 0.51 | +0.16 ± 0.06 |
| conll2003 | shared_ner | 82.69 ± 0.57 | 82.85 ± 0.51 | +0.16 ± 0.07 |
| wikiann | shared_ner | 37.78 ± 0.36 | 37.83 ± 0.36 | +0.05 ± 0.01 |
| wnut | shared_ner | 40.61 ± 0.99 | 40.72 ± 1.29 | +0.11 ± 0.41 |

Runtime: 2.4 min. [Per-seed scores and budgets](tables/label_change/). [Mechanism diagnostic](tables/label_change/mechanism.csv) measures flips and agreement with the scorer on accepted held-out calibration candidates; this is descriptive pseudo-label agreement, **not gold accuracy**.

[Implementation](IMPLEMENTATION_18.md). Proposals and accepted feature arrays: `results/label_change/generate/`.
