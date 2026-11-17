# Task: diagnostic experiments for a causal framework of over-contextualization in sequence labelling

## Context (read first)

We are developing a theoretical framework for "over-contextualization": a token classifier on top of a pretrained encoder uses context features whose association with the label is spurious, i.e. induced by a nuisance Z (domain, genre, source). Objects and claims to test:

- Encoder h is frozen (as in this repo's baselines); classifier g is the existing small MLP. A predictor for position i reads h(X_S)_i, where X_S is the sentence with every token outside the kept set S replaced by the mask token and re-encoded.
- Three views per token i: full (nothing masked), word-only (everything except token i masked), context-only (token i masked). If a word spans several subwords, mask all of them and pool exactly as the baseline does.
- Masking is claimed to mean "conditioning on less": a classifier trained on a masked view should estimate P(Y_i | X_S), not behave arbitrarily.
- Latent-state model: the label Y_i is a coarsening of a finer latent state (which entity is meant, word sense/function). Assumption A says the nuisance carries no information about that state once the label is known. Its predicted failure mode: within a label, the predictor's output still depends on Z, and this dependence concentrates on ambiguous tokens.
- Encoder claim (after Lee et al. 2021, "Predicting What You Already Know Helps"): the context-only representation of an MLM encoder approximately encodes the posterior over (latent state, Z), so the nuisance is linearly readable from it.
- "Two places to enforce invariance": with h frozen, invariance can be imposed on the input (masking) or on the readout (removing Z-directions from h). The readout route works only if Z is linearly separable from label information in h.

This phase is diagnostic only: no selector or new training method yet.

## Step 0: inventory (do this before running anything heavy)

1. List the datasets available, their tasks (NER, POS, chunking, …), splits, label sets and sizes, and the encoder, layer and pooling used by the baselines. Report the baseline scores.
2. For each dataset, determine whether a nuisance variable Z is available, and which. Candidates: genre or document-source metadata (e.g. OntoNotes genres, UD-EWT genres from document ids), domain splits (e.g. CrossNER), or dataset identity when two datasets share a label subset. For dataset identity, map to the shared labels and state the mapping. Datasets with no usable Z are used only in experiments that don't need Z.
3. Write the inventory and the per-experiment dataset assignment to `results/plan.md`, then proceed.

Do not modify the baseline code or checkpoints. Reuse the MLP architecture and training setup for every classifier below, and run 3 seeds wherever training is involved. Re-encoding with masks costs one forward pass per masked position, so subsample (e.g. a fixed random set of up to 8 positions per sentence, same subsample for all views) and report the sample sizes.

## Experiments

Each experiment states the claim it tests. Report numbers with seed spread. Keep measurement and interpretation separate in the write-up.

**E1. Three views (does context help, and where).** Train one classifier per view (full, word-only, context-only) per dataset. Report overall and per-label accuracy/F1, and the per-token gain of full over word-only, binned by word frequency and by word-level label entropy (entropy of the label distribution of the word type in train). This tests the framework's view decomposition and whether "context helps" is concentrated on ambiguous or rare words.

**E2. Masking as conditioning on less.** For word types seen at least 20 times in train with more than one label, compare the word-only classifier's predicted distribution with the empirical label distribution of that type in train (KL, and calibration/ECE). Repeat with a classifier trained on the full view but evaluated on the word-only view. The claim predicts a good match only for the classifier trained under masking. Report a few illustrative ambiguous types.

**E3. Z is linearly readable, and from which view (encoder claim).** On datasets with Z, train linear probes for Z on full, word-only and context-only representations, in-label: within each label, so Z is not predicted through label frequency. Report balanced accuracy against chance. The claim predicts high Z-readability from the context-only view.

**E4. Separability: can Z be removed at readout without losing Y?** Remove linearly decodable Z information from the full representation with an iterative nullspace-projection or closed-form linear erasure method. Report: Z-probe accuracy after erasure (should reach chance), then retrain the Y classifier on the erased representations and report the Y drop. Small drop is evidence for additive, separable Z; a large drop is evidence for entanglement, the case where only input masking can help.

**E5. Within-label dependence on Z and its concentration on ambiguous tokens (Assumption A).** For each view's classifier, within each label, measure how the distribution of predicted probabilities differs across Z values: MMD with an RBF kernel on the probability vectors, with a permutation test. Stratify by ambiguity (word-level label entropy bins, and separately the classifier's own predictive entropy). Also measure, within each label, how much the word distribution itself differs across Z (e.g. Jensen–Shannon divergence of word-type distributions), as an observable proxy for the nuisance–state dependence. The framework predicts that within-label Z-dependence grows with ambiguity and is larger for the full and context-only views than for the word-only view.

**E6. Cross-domain shift.** Where Z defines domains, train each view's classifier on one Z value and test on the others (and in-domain). Report degradation per view. Over-contextualization predicts the context-only view degrades most and the word-only view least, with the full view in between. Relate per-label degradation to the E5 dependence measures.

**E7 (NER only, if feasible). Localised nuisance via substitution.** For a focal entity token, replace a *neighbouring* entity mention with another mention of the same type sampled from train, and separately replace the *focal* mention itself. Report how often the focal prediction changes, per view. The neighbour substitution is an exact intervention on a localised nuisance; changes under it are direct evidence of co-occurrence over-contextualization.

## Outputs

- `results/plan.md` (Step 0).
- One script per experiment, reproducible from the command line; tables as CSV; figures as PNG.
- `results/report.md`: per experiment, a few sentences on what was measured, the key table or figure, and whether the result supports, contradicts, or is inconclusive for the stated claim. Mark clearly which statements are measurements and which are interpretation.

If an experiment can't be run on a dataset (no Z, too few samples, label mapping impossible), say so in the report and move on rather than substituting a different design.