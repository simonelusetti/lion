# Context sensitivity across encoder layers

Run: `USE_TF=0 .venv/bin/python -B -m lion.layer_sensitivity.run` (script:
`scripts/run_layer_sensitivity.sh`). Config: `conf/layer_sensitivity.yaml`;
every stage (estimate, train, measure, analyse, report) is a Forge/Hydra run.
Raw results: `results/layer_sensitivity/`; report: `report/RESULTS_13.md`.

## Reused inputs

- **CoNLL-2003**: scorer, selectors g1/g2, cuts and training roles from
  `results/augmentation_full/source/28b4f1db440a`; test rows and four cached
  fills per row, mask and fold from `results/sensitivity_1b/generate/conll2003`;
  existing baselines from `results/augmentation_full/f3d6dc697455/train`
  (arm 0). Groups O and entity are the scorer-predicted groups stored with the fills.
- **EWT dependency relations**: everything from `results/deprel_replication`
  (source, `generate/en_ewt_deprel`, `train`). One group.

No scorer or selector is trained. Existing baselines and scorers read layer 12
(the final hidden state; `method.kind=layers, method.layer=12` in both source configs).

## Encoder and vectors

The same frozen bert-base-cased backbone (the MLM model's BERT, fp32 weights,
bf16 autocast, as in the cached features). `output_hidden_states` gives the
embedding output (layer 0) and blocks 1–12. A word's vector at a layer is the
mean of its WordPiece vectors at that layer. Training and validation vectors
for all 13 layers are computed in memory and never saved; test vectors for the
unperturbed sentences are saved once (`train/<dataset>/test_features.pt`).
The analysis reports the cosine between the cached block-12 test vectors and the
recomputed ones.

## Heads

The existing matched baseline recipe (`fewnerd_replication.train`): MLP
768→256→labels, dropout .1, AdamW lr .001, weight decay .01, per-head clip 1,
ten epochs over T1∪T2 with the same seeded sentence schedule, 64-sentence
batches, checkpoint on raw-dev token macro-F1. Seeds 0/1/2. Each seed seeds the
MLP initialisation exactly as before, so the layer-12 head reproduces the
existing baseline up to vector rounding; the report gives the per-seed
metric difference and test prediction agreement.

- One head per layer 0–12.
- One scalar mixture: `scale · Σ_l softmax(w)_l h_l`, w initialised to zero
  (uniform) and scale to one, trained jointly with the MLP by the same optimizer.
  The gradient clip applies to the MLP as before; w and scale are not clipped.

Performance: span F1 (CoNLL, existing exact BIO span code) and token accuracy
and macro-F1 (EWT). The best single layer maximises the seed-mean raw-dev
token macro-F1.

## Perturbations

For each row, fold and mask kind (selector, random with the same count):

1. **Natural**: the four cached fills (token IDs saved by the earlier runs).
2. **Masked**: every dropped word is replaced by a single [MASK] token,
   keeping word alignment; one deterministic version. The focal word and the
   kept words are unchanged. Natural fills preserve piece counts; masked versions
   shorten multi-piece dropped words, so later position embeddings shift.

A version identical to the original is not re-encoded; its KL, flip and
shift are exactly zero. All other sequences are encoded once in length-sorted
chunks (`layers.token_budget` tokens). The focal vector at every layer is passed
straight to every head; only per-row KL, flip and cosine distances are kept.
Perturbed vectors are never stored.

## Measurements

- **S** = KL(p_original ‖ p_perturbed) with γ=.01 uniform mixing; flip = argmax change.
  Natural: all draws; non-null draws (changed-word count > 0, step 1b); the
  step-1b size × distance-matched selector/random ratio (changed-count × nearest-
  changed cells, selector non-null draw weights, cells with < 50 rows or zero
  random S omitted). Masked: the raw selector/random ratio, and S and flips by
  distance from the focal word to the nearest dropped word (bins 1, 2, 3, 4–5,
  6–10, >10; rows with a drop in that bin, common support across masks).
- **Representation shift**: 1 − cos(h(x), h(x̃)) of the focal vector at the head's
  layer; for the mixture, the seed's mixed vector (cosine ignores the scale).
  Normalised by the mean cosine distance of 20,000 random test-word pairs
  from different sentences with different case-sensitive forms, at the same layer
  (for the mixture, the same seed's mixed vectors).
- **Gain**: mean S divided by mean normalised shift, per perturbation and mask kind,
  computed per seed (row SE by the delta method on seed-averaged rows).
- **Contextuality**: unperturbed test vectors. For each case-sensitive word form with
  at least five test occurrences, the cosine distance of each occurrence from the
  form's mean vector, averaged over occurrences in the group and normalised as above.

Averaging follows step 1b: draws within fold, then folds within row. Seeds give
mean ± sample SD; row SE is computed from seed-averaged row values. Single-layer
shift and contextuality do not depend on the seed.

## Runtime rule

The estimate stage measures all-layer encoding throughput on 4,000 random
perturbed sequences and scales by the tokens every later stage encodes, plus 15 s
per head. If the total exceeds six hours, single-layer heads are restricted to
layers 0, 2, 4, 6, 8, 10, 12 plus the mixture. In that case the best layer is
chosen among those trained. Stage times are saved in each stage's
`runtime.json` and in the report.
