# Label-aligned share of contextual variation, per layer

Run: `scripts/run_label_share.sh` (`python -B -m lion.label_share.run`). Config:
`conf/label_share.yaml`; stages measure and report are Forge/Hydra runs.
Raw results: `results/label_share/`; report: `report/RESULTS_15.md`. Nothing is trained.

## Data

Full native training sets, as in the coverage run: CoNLL-2003 (14,041 sentences)
and EWT (12,544 sentences, the uncapped bank `results/coverage/prepare/en_ewt_full`).
Word forms are the case-sensitive strings from each word's WordPiece IDs (saved by
the coverage prepare stage). Labels: CoNLL BIO tags collapsed to entity type plus O
(B-X and I-X → X); EWT base dependency relations.

Vectors: frozen bert-base-cased, mean of WordPiece vectors per word at layers 0–12,
computed in memory for every training word (never saved). Mixture: the three
mixture heads trained on the full native training set in the coverage run
(fraction 1.0, seeds 0–2); mixed vector = scale · Σ_l softmax(w)_l h_l. Every
quantity is computed per seed and the three results are averaged at the end.

## Eligible types

A form is eligible if at least two labels each occur ≥10 times with it in training.
Only the occurrences of those labels enter the measure (rarer labels of the same form
are left out).

## Measure

Per view (layer or mixture seed) and eligible type w, with occurrences x_i:
- T_w = mean_i ‖x_i − μ_w‖².
- B_w = Σ_y p(y|w) ‖μ_wy − μ_w‖², p(y|w) the label's share of w's kept occurrences.
- share_w = B_w / T_w.
- Baseline: labels permuted within w (counts kept), B recomputed, share averaged over
  20 shuffles (fixed generator seed).

A type whose occurrences are identical at a view (T_w ≤ 1e-9 × the random-pair reference; e.g. a form that always sits at the same position, at layer 0) has an undefined share and is left out of that view's aggregates; the count is reported. Aggregates weight the remaining types by occurrence counts. Total spread (occurrence-weighted T_w)
is divided by the mean squared distance between 20,000 random training-word pairs
from different sentences with different forms, at the same view.

Robustness: each dimension is z-scored with its mean and SD over all training words
of that view, and every number (including the pair reference) is recomputed.

## Report

One table per dataset: layers 0–12 and the mixture (mean ± SD of the three seeds),
with normalized total spread, share, baseline, share − baseline, standardized share and
standardized share − baseline. The layer where share − baseline peaks is marked
(raw and standardized). Per-type values for every view are in `per_type.csv`, with
the 20 most frequent eligible types flagged.
