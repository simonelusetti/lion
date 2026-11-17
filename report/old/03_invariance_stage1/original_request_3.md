# Follow-up: does the right invariance signature change between tokens of the same sentence?

## Context
You ran E1–E7 and X0–X4 (RESULTS.md, RESULTS_2.md and their IMPLEMENTATION files). Reuse that setup unless stated otherwise:
- frozen bert-base-cased block 12, first-WordPiece representation, full view only (no masking);
- MLP 768→256→labels with the same optimizer;
- seeds 0/1/2.

Unlike before, use **every labelled word** of each sentence, not two sampled focals. The penalties below need many tokens per (genre, label) cell, and full-view vectors cost one encoder pass per sentence.

Veitch et al. (Counterfactual Invariance to Spurious Correlations, Thm 3.2) give two regularizers whose correctness depends on causal direction:
- **anticausal mode:** conditional independence of the prediction and genre given the gold label. Penalize the MMD between genre-specific prediction distributions within each label.
- **causal mode:** marginal independence of the prediction and genre (Thm 3.2 case 2). Penalize the MMD between genre-specific prediction distributions, all labels pooled.

The question is whether the better mode differs between tokens of the same sentence, in a way predicted in advance. The hypothesis (H_annot):
- for unambiguous words, the label is fixed by the word's latent state, which is the anticausal case;
- for ambiguous, rare or unseen words, the annotator decides the label from context, so the label is effectively computed from the text, which is the causal case.

The competing explanation (H_shift): any difference between token groups only tracks how much a label's frequency changes across genres. The marginal penalty mechanically punishes genuine label shift.

Do not use a learned gate anywhere. Token groups come from a fixed lookup.

## Datasets and environments
- Main: EWT UPOS, EWT deprel, GUM UPOS. Secondary, if budget allows: GUM discourse, where genre is plausibly evidence rather than nuisance (a contrast case).
- Environment = genre: EWT has 5, GUM has 15.
- Rotations:
  - EWT: each genre in turn is the test genre. The next genre in a fixed cyclic order is the validation genre. The remaining 3 are training genres.
  - GUM: 5 prespecified rotations, chosen to spread genre types (all 15 if budget allows). Same rule for the validation genre.
- Splits:
  - training genres use their native train split;
  - validation and test genres use **all** their sentences from every native split, since the model never sees them. This gives the test genres more documents.
  - in-domain test = native test split of the training genres.

## Token groups (fixed lookup, recomputed per rotation from training genres only)
For each word form, compute from training-genre tokens:
- its frequency;
- the entropy H of its label distribution, in nats.

Groups:
- **unambiguous:** frequency ≥ 5 and H ≤ 0.2;
- **ambiguous:** frequency ≥ 5 and H > 0.2;
- **rare/unseen:** frequency < 5, including unseen.

Also run a sensitivity analysis with an H threshold of 0.5. The group of a test token is looked up from its word form, never from its gold label.

## Penalties
- Genre-stratified batching: every batch contains sentences from all training genres in equal proportion.
- MMD uses an RBF kernel on predicted probability vectors, with the bandwidth set by the median heuristic on the batch (detached).
- Marginal penalty: sum over pairs of training genres of the MMD between their prediction distributions, computed over the tokens assigned to causal mode.
- Conditional penalty: for each gold label, the same quantity over tokens of that label assigned to anticausal mode. Weight each label by its token share and skip cells with fewer than 8 tokens per genre.
- Weight each mode's penalty by its group's token share, so the total penalty strength is comparable across rules.
- Penalty strength λ ∈ {0.1, 0.3, 1, 3, 10}. λ = 0 is the shared unregularized baseline.
- Select λ per rule and rotation on validation-genre macro-F1. Report the full λ curves as well.

## Rules (main experiment: one model per rule)
| Rule | Assignment |
|---|---|
| R0 | no penalty |
| R1 | fixed anticausal: every token gets the conditional penalty |
| R2 | fixed causal: every token gets the marginal penalty |
| R3 | hypothesis split: unambiguous → conditional; ambiguous and rare/unseen → marginal |
| R4 | reversed split: unambiguous → marginal; ambiguous and rare/unseen → conditional |
| R5 | random split: word types randomly assigned to the two modes so token shares match R3; 5 independent draws |

R5 randomizes at the word-type level because R3 assigns by word type.

## Routing (secondary experiment: two models)
1. Take the selected R1 and R2 models.
2. For each token group, choose the mode with the better validation-genre accuracy.
3. Route each test token to that model by its group.

Controls:
- the average of R1's and R2's probabilities (ensembling helps transfer on its own);
- random-type routing with matched shares (5 draws).

## Measurements
- Out-of-domain accuracy and macro-F1 on the test genre, overall and per token group.
- In-domain accuracy (native test of the training genres), to show what each penalty costs.
- Label-shift magnitude, from training genres only:
  - per label: Jensen–Shannon divergence of its genre-specific frequency;
  - per token group: the token-weighted mean over its labels.

  Split test tokens into label-shift tertiles.

## Primary comparisons (fixed before looking)
1. **Fixed modes.** R1 minus R2 out-of-domain accuracy, overall.
2. **Interaction.** Compute (R1 minus R2) out-of-domain accuracy in the unambiguous group and in the ambiguous and rare/unseen groups, and take their difference.
   - Report it overall and within each label-shift tertile.
   - H_annot predicts that R1 is relatively better on unambiguous tokens *within* tertiles.
   - H_shift predicts that the difference vanishes within tertiles.
3. **Structure of the split.** R3 must beat:
   - R4;
   - the mean of the R5 draws (also report R3's rank among them);
   - the better of R1 and R2.

   Beating R5 shows the grouping carries signal. Beating R4 shows the direction of the assignment matters, not just mixing two penalties. Beating R1 and R2 shows the mixture beats either pure mode.

Everything else is descriptive.

## Statistics
- Paired comparisons across rules share rotation and seed.
- Report means with seed SD and document-block bootstrap intervals within test genres. Aggregate rotations equally.
- Do not run large families of per-cell tests.

## Deliverables
- RESULTS_3.md and IMPLEMENTATION_3.md in the same style as before, plus CSVs.
- Figures:
  - out-of-domain accuracy against λ, one curve per rule;
  - (R1 minus R2) per token group, faceted by label-shift tertile;
  - R3 against the distribution of R5 draws;
  - routing against its controls.
- In RESULTS_3.md, state for each primary comparison whether it favours H_annot, H_shift or neither, and list every deviation from this protocol.