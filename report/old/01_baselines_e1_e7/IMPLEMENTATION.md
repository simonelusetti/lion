# What was implemented

This describes the code that produced the recorded results, rather than a proposed protocol. The [original theory prompt](original_request.md) requested seven diagnostics of over-contextualization. The implementation is in [lion/diagnostics](../../../lion/diagnostics/); the numerical results and their interpretation are in [RESULTS.md](RESULTS.md).

## Scope and changes to the requested protocol

The diagnostic suite trained no selectors and updated no BERT weights. It reused the baseline MLP architecture and optimizer settings. Its full-view classifiers were trained anew on the same focal samples as the masked-view classifiers; they are distinct from the full-data baselines.

| Requested component | Actual implementation |
|---|---|
| Inventory before expensive work | Saved dataset sizes, label sets, baseline scores, availability and assignments in `results/plan.md`. |
| Nuisance metadata Z | Recovered EWT and GUM genres and document IDs from downloaded raw UD sentence IDs. Retokenization checks align them with the existing token caches. |
| Potentially eight focal positions per sentence | At most two, with explicit sentence and focal caps. GUM received a larger budget after an eligibility audit. |
| All datasets | Diagnostics used the six available datasets. WikiANN and WNUT loaders failed initially; their baselines were subsequently repaired and completed, but they were not silently added to the finished diagnostic cohort. |
| Three seeds wherever diagnostic training occurs | Seeds 0, 1 and 2 for every diagnostic Y classifier and Z probe. Existing full-data baselines use seed 0 only. |
| Erasure method | A closed-form, conditional-covariance, LEACE-style affine erasure, followed by fresh Y classifiers and Z probes. |
| Permutation tests | Document-block permutations for UD; sentence-block permutations for shared NER, whose original document IDs are unavailable. |
| Entity substitutions | Same-type replacements additionally matched each word's WordPiece length, preserving alignment and sequence length. |

## Common representation and training setup

- **Encoder:** frozen `bert-base-cased`, block 12, 768-dimensional vectors, evaluation mode throughout. The `bert` family implementation is shared with the uncased checkpoint.
- **Tokenization:** matching fast tokenizer, maximum 512 WordPieces, no CLS/SEP. A word is represented by its first WordPiece. Truncation and packing follow the baseline token caches.
- **Y classifier:** 768 → 256 → labels, ReLU, dropout 0.1; independent word classification, without a sequence decoder.
- **Optimization:** AdamW, learning rate 0.001, weight decay 0.01, gradient clipping at norm 1, 10 epochs. Each optimizer step batches 64 sampled sentences. Sentence shuffling is disabled, matching the baseline configuration.
- **Selection:** best validation macro F1, then one evaluation of that checkpoint on test. Linear Z probes instead select on validation balanced accuracy.
- **Runtime:** CUDA, bf16 autocast, TF32 enabled, float16 cached representations converted to float32 when read. The diagnostic launch used `runtime.workers=0 runtime.batch_size=32`; extraction chunk sizes and GPU parallelism come from runtime configuration.

DataLoaders batch sentences with explicit collate functions. The diagnostic classifier loader concatenates the selected word vectors from those sentences. Each view is initialized from the same training seed. The encoder forward is shared once per full sentence; masked focal forwards are expanded and processed in bounded chunks. Cached vectors are reused across classifier seeds.

Relevant code: [encoder](../../../lion/encoder.py), [taggers](../../../tagger/model.py), [view extraction and sampling](../../../lion/diagnostics/data.py), [readout training](../../../lion/diagnostics/readouts.py), and [configuration](../../../conf/diagnostics/default.yaml).

## Sampling and datasets

Sampling uses seed 1729. For each native split, sentences are randomly ordered, up to two valid focal words are sampled per sentence, and explicit caps stop collection. Selected sentences are then restored to sentence-ID order. All views and all three classifier seeds use those identical focal positions. GUM words with invalid labels are excluded.

| Budget | Train sentences / focals | Validation sentences / focals | Test sentences / focals |
|---|---:|---:|---:|
| Default | 2,048 / 4,096 | 512 / 1,024 | 1,024 / 2,048 |
| GUM | 8,192 / 16,384 | 1,600 / 3,200 | 1,500 / 3,000 |

Actual counts are lower for short sentences or exhausted native splits; [sample_sizes.csv](tables/sample_sizes.csv) records them. At the smaller budget, no GUM label had enough support in two genres for conditional probing. The larger sample supports 11 labels; EWT UPOS supports 9 and dependency labels 8.

Sentence-uniform sampling changes the weighting of words relative to full-corpus evaluation. These diagnostic scores are therefore not unbiased estimates of the full-data baselines. UPOS and dependency relations share EWT sentences and focal positions; they are two tasks on one corpus, not independent corpus replications. The shared NER condition also reuses existing native focal examples.

### Nuisance variables

EWT genres are `answers`, `email`, `newsgroup`, `reviews`, and `weblog`, extracted from the prefix of raw `sent_id`. GUM genres are `academic`, `bio`, `conversation`, `court`, `essay`, `fiction`, `interview`, `letter`, `news`, `podcast`, `speech`, `textbook`, `vlog`, `voyage`, and `whow`, extracted from `GUM_<genre>_...`. Removing the final sentence-number suffix supplies document IDs. These recovered UD document IDs have no train–validation or train–test overlap in this inventory.

No usable native genre/source variable was retained for the other diagnostic datasets. A separate `ner_shared` bank pools Few-NERD and CoNLL-2003 and treats dataset identity as Z. Its targets are O/PER/ORG/LOC: native O stays O, person/PER maps to PER, organization/ORG to ORG, and location/LOC to LOC. BIO prefixes are stripped. Unsupported entity targets, including Few-NERD building and CoNLL MISC, are excluded, not converted to O; their words remain visible as context. The two datasets keep their native splits.

Dataset identity combines source, annotation and sampling differences. It is an observed domain proxy, not an identified causal nuisance.

## Three views

For every focal word, the three representations are:

1. **Full:** original sentence.
2. **Word-only:** replace every other word's WordPieces with `[MASK]`.
3. **Context-only:** replace every WordPiece of the focal word with `[MASK]`.

The original first-WordPiece position is read after re-encoding. Attention stays on for replaced words. This differs from Lion's existing learned attention gates. No words are removed and no sequence is shortened.

Word-only inputs still expose sentence length, focal position and subword counts; context-only exposes the masked word's subword count. Consequently, the experiment is not a literal isolation of the word identity or of an unobserved latent-state posterior.

## E1 — predictive value of the views

[e1.py](../../../lion/diagnostics/e1.py) trains an independent MLP for each view, dataset and seed. It records token accuracy, macro F1, cross-entropy, and per-label precision/recall/F1. Disconnected focal samples do not receive span F1.

For paired test positions it saves full-minus-word-only correctness and word-only-minus-full cross-entropy. Positive values favor full context. These gains are grouped by exact, case-sensitive word form using **all original training occurrences** to compute frequency and label entropy. Frequency bins are unseen, 1–5, 6–20 and 21+. Entropy bins are unseen, zero, `(0, ln 2]`, and `> ln 2`; entropy is undefined for unseen forms. Test labels do not determine these lexical statistics.

## E2 — masking and empirical lexical distributions

[e2.py](../../../lion/diagnostics/e2.py) selects types occurring at least 20 times in the full training split with at least two observed labels and at least one occurrence in the held-out focal sample. It compares two predictors on word-only test vectors: the word-only-trained MLP and the full-trained MLP.

For each type, the target distribution is its full-training empirical label distribution. Predictions are averaged across its held-out occurrences. The primary divergence is `KL(empirical training distribution || mean prediction)`, averaged equally across eligible types. Additional outputs include occurrence-level KL, token-weighted type KL, 10-bin confidence ECE, and multiclass Brier score. Natural logarithms are used.

This tests adaptation to masked inputs. It does not assume that the empirical distribution is the true posterior or that finite sampled training must reproduce it exactly.

## E3 — within-label linear readability of Z

[e3.py](../../../lion/diagnostics/e3.py) fits linear softmax Z probes independently within each gold Y label and view. A domain is eligible for a label only with at least 32 training, 8 validation and 8 test focal examples; at least two eligible domains are required. Other domains are omitted for that label.

Training cross-entropy uses inverse-frequency domain weights. The test metric is balanced accuracy, with chance `1 / number of eligible domains`. The eligible domain set can differ across labels, so a dataset's average chance need not equal the reciprocal of its total genre count. Labels are used to define analysis subsets, not passed to the probe as features.

## E4 — conditional linear erasure

[e4.py](../../../lion/diagnostics/e4.py) uses only training full-view vectors to fit one affine map. Let `r = one_hot(Z) − empirical P(Z | Y)` and define the concept vector as `one_hot(Y) ⊗ r`. The code estimates representation covariance and representation–concept cross-covariance, whitens the representation, and uses the cross-covariance's left singular vectors to identify a removable subspace. Eigenvalue and singular-value thresholds use relative tolerance `1e-6`.

Writing the whitener as W, its pseudoinverse as W⁺, the concept basis as U, and the training mean as μ, the map is:

```text
P = I − W⁺ U Uᵀ W
h_erased = μ + P (h − μ)
```

The same map is applied to every example without consulting Y or Z at inference. Fresh three-seed Y MLPs and conditional linear Z probes are trained on erased vectors. The reported Y drop is paired against E1's full-view classifier with the same seed. Projection rank and training covariance residuals are saved.

This is a conditional-covariance erasure, not a guarantee that every form of linearly accessible domain information vanishes. Post-erasure probes are necessary; an average near chance can also hide residual readability in individual labels.

## E5 — within-label dependence on Z

[e5.py](../../../lion/diagnostics/e5.py) compares predicted probability vectors within each gold label and domain pair. It analyzes all examples, lexical-entropy strata, and predictive-entropy strata separately. Predictive cutoffs are training-prediction tertiles for that fitted classifier.

Each group contributes at most 128 focal tokens and must retain at least 20 tokens from at least 3 documents. The statistic is unbiased squared MMD with an RBF kernel. Its bandwidth squared is the median positive pooled squared distance within that cell, fixed during permutations. Negative unbiased estimates are allowed. Bandwidths are estimated separately across views/cells, so cross-view MMD comparisons are descriptive and depend on this normalization.

There are 199 random permutations of document-level domain labels, preserving token membership within documents. The Monte Carlo p value is `(1 + number of permuted statistics ≥ observed) / 200`. Shared NER uses sentence blocks because document IDs are unavailable. Benjamini–Hochberg adjustment is applied separately within each seed across all measured datasets, views and strata.

Empirical Jensen–Shannon divergence of word-type distributions is computed on the same selected groups, without smoothing, in nats. Its finite-sample value is not an estimate of latent-state dependence without further assumptions. Cells without support are recorded as skips. In particular, every GUM test genre has only two documents, so GUM cannot satisfy this permutation protocol even with more focal words.

## E6 — cross-domain shift

[e6.py](../../../lion/diagnostics/e6.py) trains each view's MLP on a single source genre/dataset and selects its checkpoint using that source's native validation examples. Source eligibility requires at least 128 training, 16 validation and 32 test tokens. Targets require at least 32 test tokens.

It evaluates source and other-domain test samples, recording accuracy and macro F1, source-minus-target degradation, per-label recall changes, and test support for labels absent from source training. It also relates per-label recall degradation to E5's unstratified MMD for the corresponding domain pair, where at least three supported labels permit a correlation. These correlations compare source-trained classifiers with dependence measured for pooled-data classifiers and remain descriptive.

## E7 — controlled entity substitutions

[e7.py](../../../lion/diagnostics/e7.py) uses available native NER datasets only. BIO tags define CoNLL mentions; contiguous same-type runs define Few-NERD coarse spans. Adjacent same-type Few-NERD entities therefore cannot always be distinguished as separate mentions.

The training split supplies a reservoir of up to 64 distinct replacements per entity-type/per-word-WordPiece-length signature. For up to 128 accepted test focal entities per dataset, the code independently attempts to replace the nearest other entity within eight words and to replace the focal mention itself. The original surface is excluded. Retokenization must preserve packed word alignment exactly.

Modified sentences are encoded in the same three views and evaluated by the already-trained E1 classifiers. The outputs are paired argmax flip rates, probability L1 changes and before/after accuracy. Single-word and multiword focal mentions are reported separately. Neighbor word-only inputs and single-word focal context-only inputs supply controls. Same-type replacement does not establish semantic coherence or prove that the intervention changes only a nuisance.

## Results, aggregation and reproduction

All diagnostic training uses seeds 0/1/2 on one fixed focal sample. Tables show sample standard deviation across the three seeds, not confidence intervals or uncertainty from resampling data. Aggregate probe tables average eligible labels equally within a seed; domain-shift tables average ordered source–target pairs equally within a seed. MMD aggregates average eligible label/domain cells equally. The supporting CSVs retain more detailed cells and seed results.

The suite trained 702 diagnostic readouts: 63 E1 MLPs, 288 E3 probes, 12 erased Y MLPs plus 96 post-erasure Z probes, and 243 source-domain MLPs. E2, E5 and E7 reuse these models. The eight full-data baselines are separate single-seed fits.

From the project root, the original diagnostic invocation was:

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
  .venv/bin/python -u -m lion.diagnostics.run runtime.workers=0 runtime.batch_size=32
```

Each experiment can also be invoked as `python -m lion.diagnostics.e1` through `e7` after its prerequisites exist. Baseline grids are [tagger_baselines.yaml](../../../grids/tagger_baselines.yaml) and the two-dataset [retry grid](../../../grids/missing_baselines.yaml). Configurations, source paths and checksums are recorded in [provenance.json](provenance.json). Because the inventory has since changed, reproducing the original six-dataset cohort in a fresh location requires explicitly selecting those six datasets rather than silently including the repaired WikiANN/WNUT loaders.
