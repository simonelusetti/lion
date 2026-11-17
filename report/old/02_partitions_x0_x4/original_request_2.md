# Follow-up diagnostics: which partition of the context separates label evidence from nuisance?

## Context
You ran E1–E7 (RESULTS.md, IMPLEMENTATION.md). Reuse that setup unless stated otherwise:
- frozen bert-base-cased block 12, first-WordPiece readout;
- MLP 768→256→labels with the same optimizer;
- seeds 0/1/2;
- the same focal sample (sampling seed 1729, same budgets);
- [MASK] replacement with attention on and length preserved;
- the same nuisance variables Z (EWT 5 genres, GUM 15 genres, shared-NER dataset identity).

These are measurements only: no selectors, no encoder updates, no invariance training.

The theory now considers that a focal word's context may split into two channels:
- words carrying evidence about the focal word's latent state (label evidence);
- words carrying the sentence-level nuisance Z.

Candidate formulations of the split, which these experiments should discriminate:
- L (linear locality): label evidence sits in a small linear window; nuisance information comes from the whole sentence, so distant words carry Z but little label evidence.
- S (syntactic locality): as L, with distance measured in the dependency tree.
- K (lexical channel): the channels are separated by the type of context word (e.g. content vs function words), not by distance.
- N (no token-level partition): the same words carry both; separation is possible only in representation space, as with E4's erasure.
- A (label context): neighbouring gold labels absorb most of the nuisance's information about the focal state.

## Datasets
- EWT UPOS, EWT deprel, the GUM task already in the cohort, shared NER.
- If GUM UPOS can be derived from the same CoNLL-U files with little effort, add it. It gives a within-corpus task contrast under the same 15 genres.
- Tree distances come from gold UD heads (word level) for EWT and GUM. Shared NER gets linear distance only, unless a parser is already installed and works offline.

## Kept-set families
For focal word i, a condition keeps a set of context words visible and replaces every other word's WordPieces with [MASK]. The focal word is always kept except in G.
- Linear window, lin(k): words with |j − i| ≤ k, for k ∈ {0, 1, 2, 3, 5, 8, full}. k = 0 is word-only.
- Tree window, tree(k): words within k edges of i in the dependency tree, for k ∈ {1, 2, 3, full}.
- Random, rand(m): m context words drawn uniformly. For each focal, m matches that focal's kept count in each lin and tree condition. This separates *which* words from *how many*.
- Geometry floor, G: everything masked, focal included. It measures what length, position and mask layout reveal on their own.

Each condition gets a fresh readout trained on that condition's masked inputs, with 3 seeds. (E2 showed full-trained readouts are miscalibrated on masked inputs.)

## X0: amortized readout check
1. Train one readout on a mixture of all X1 conditions, sampling a condition uniformly per example.
2. Compare it with the dedicated readouts, per condition: accuracy, macro-F1, and mean KL(dedicated || amortized) on test.
3. Report whether one readout can stand in for the dedicated ones. Later work relies on this if it holds.

## X1: label–nuisance frontier
Measure, per dataset and condition:
- Label information: Y accuracy, macro-F1 and cross-entropy.
- Representation nuisance: within-label Z balanced accuracy of linear probes on the masked focal representation (E3 protocol).
- Predictor nuisance: within-label Z balanced accuracy of a linear probe on the readout's log-probabilities. This is the quantity the invariance target concerns.
- Geometry leakage: within-label Z balanced accuracy of a probe on sentence length, focal position and kept count alone.

Report nuisance as excess over the G condition's balanced accuracy, not over chance.

Main figure, one per dataset:
- x-axis: label information;
- y-axis: predictor-nuisance excess;
- one curve per family (linear, tree, random), with points labelled by k or m.

Primary comparisons, fixed before looking at results:
1. **Matched label information.** Interpolate each family to 90% of the full-minus-word-only gain and ask which has the lowest nuisance excess there.
   - Under L or S, windows beat random.
   - Under S, tree beats linear.
   - Under N, all families lie on one curve.
2. **Knee.** Is there a k where a window recovers at least 90% of the label gain while carrying at most 50% of the full-context nuisance excess?
3. **Shift.** Repeat the E6 single-source protocol for lin(k) and tree(k). Plot the context gain (window accuracy minus word-only accuracy) against k, in-domain and out-of-domain. Under L or S, the out-of-domain gain peaks below full context.

## X2: per-word channel attribution
On a test subsample of up to 500 focals per dataset, compute two scores for each context word j:
- r_j: the drop in log P(true Y_i) when j alone is masked;
- n_j: the drop in log P_probe(true Z) for the within-label Z probe when j alone is masked.

Train the readout and probes with random sparse occlusion (each context word masked with probability 0.15), so that single-word masking is in distribution.

Nuisance information may be spread redundantly over many words, which single-word masking would miss. So also compute r and n for removing whole distance bands, against matched-count random removals:
- linear bands 1, 2, 3–4, 5–8, 9+;
- tree bands 1, 2, 3+.

Report:
- the correlation of r and n over (i, j) pairs;
- the overlap between the top 20% of words by r and the top 20% by n (Jaccard per focal, then averaged);
- how r and n vary with:
  - linear distance;
  - tree distance;
  - the context word's UPOS (content vs function);
  - its training frequency;
  - whether it has the same form as the focal word (a cheap coreference proxy).

Fit one regression per quantity with all predictors together, and report partial effects with document-block bootstrap intervals. Under K, word class predicts n better than distance does. Under N, r and n overlap heavily.

## X3: label context as state refinement
Use the word-only representation of the focal word as a proxy for its latent state. For window widths k = 0, 1, 2, fit two probes:
- probe A: Z from one-hot gold labels Y_{i−k..i+k};
- probe B: Z from the same labels plus the word-only representation.

Report the gain of B over A, in balanced accuracy and cross-entropy, as k grows. Under A, the gain shrinks with k. Expect a floor: the word-only representation also carries lexical genre markers (word choice depends on genre directly), and label context cannot remove those. So the trend in k is the measurement, not the level.

## X4: metadata-free nuisance proxy
1. For each focal, take the words outside its linear 3-window and mean-pool their full-view BERT vectors, with the focal and window masked.
2. Cluster these without labels: k-means with K equal to the number of domains, and with twice that.
3. Report NMI and best-matching balanced accuracy against the true Z, overall and within label.
4. Compare with the same procedure on the words inside the window.

This tests whether distant context could stand in for Z when no metadata exists.

## Statistics
- GUM test genres have only two documents each (E5). For Z probes and X4, use 5-fold cross-validation grouped by document over the pooled sample instead of the native split. Y readouts keep the native split.
- Report means with seed SD and document-block bootstrap intervals.
- The three X1 comparisons are the confirmatory analyses; everything else is descriptive. Do not run large families of per-cell significance tests.

## Deliverables
- RESULTS_2.md and IMPLEMENTATION_2.md, in the same style as before.
- CSVs for every table.
- Figures:
  - the X1 frontier per dataset;
  - the X1 shift curves;
  - X2's r and n against distance, and the r–n scatter;
  - X3's gain against k.
- In RESULTS_2.md, state for each primary comparison which formulation the measurement favours and how strongly, and list every deviation from this protocol.