# Conflict-insertion feasibility: coverage, supply, room

Run `scripts/run_conflict.sh`. All stages use Forge/Hydra through
`python -B -m lion.conflict.run`; config `conf/conflict.yaml`.
Raw arrays: `results/conflict/`. Report: `report/RESULTS_17.md`; CSVs under
`report/tables/conflict/`, including the 100-example manual audit.

## Reused artifacts and data

The codebase now includes layer sensitivity, full-data coverage, label-share
and separability runs. Reuse the full native banks established by coverage:
CoNLL-2003 train/dev/test (14,041/3,250/3,453 sentences), and uncapped EWT
(12,544/2,001/2,077). Source paths and saved-head paths are in
`prepare/provenance.json`. Existing layer-8 and layer-12 heads are exactly the
three-seed heads from `results/layer_sensitivity/train/`, trained on that run's
T1∪T2 populations; they are not replaced by the newer full-data coverage heads.

CoNLL labels collapse B/I into O/PER/ORG/LOC/MISC. Slots are O words plus gold
entities whose span length is one word. Retained entity-word shares are
43.57% train, 43.79% dev, 44.06% test. EWT uses every word and the existing
36 base relations. No new task labels, augmentation or selector is trained.
Training lexical counts include **all** native-training occurrences, including
CoNLL entity words excluded from slot eligibility. Forms are case-sensitive,
using bank word strings where present and the existing WordPiece reconstruction
otherwise. Slots and the vocabulary/count tensors are saved in `prepare/*.pt`.

## The only new predictor

One seed-0 context-only predictor per task. Physically remove every WordPiece
of the focal word and insert exactly one [MASK]; preserve the rest of the
sentence and the bank's special tokens. Read frozen bert-base-cased block 8 at
that position. Cache features once, then fit the existing 768→256→labels MLP:
ten epochs, 64 sentences/batch, AdamW lr=.001, decay=.01, dropout=.1, norm cap 1,
checkpoint criterion token macro-F1. A fixed random 10% **document** holdout
inside native train supplies checkpoint selection. The remaining 90% of training
documents supply fitting. Native dev is untouched by fitting/model selection and
serves calibration only. This holdout is explicit so its CE bound is held out.
Predictor dev/test accuracy and absolute CE are reported overall and by group.

## 1. Coverage

Select only by context-only max-label probability ≥ .9/.95/.99. Do **not**
require the predicted top label to match gold. Per threshold/group on native dev,
report coverage, selected gold error, selected mean/SD/SE of unmixed gold CE,
and UCB = mean + 1.645 SE (undefined for fewer than two rows). These are
pointwise one-sided 95% CLT mean-risk bounds, with the usual sampling assumptions;
CE bounds conditional entropy in that selected population. They are not simultaneous
bounds, or certificates that each particular context has a fixed label.

Test coverage includes overall, CoNLL O/entity, and each group's conflict-prone
subset. A form is conflict-prone if training contains multiple labels, or if it
is seen but never with the current gold label. Unseen words do not qualify.

## 2. Supply

Use the frozen BERT MLM at the same single [MASK] slot. Allowed token strings
are non-special, non-## Unicode alphanumeric tokens (letters/digits, no
punctuation). This is a single WordPiece in a whole-word position. Use raw
full-vocabulary softmax probabilities for masses, plausibility thresholds and
prices; normalize only for natural sampling. Conditional-on-allowed-vocabulary
masses are additionally saved in CSVs to make the convention explicit.

A lexical candidate has ≥5 native-training occurrences and a unique most
frequent label (the majority-vote label). Tied maxima are excluded and counted.
Conflict: that label differs from y_c and training share of y_c ≤10%.
Non-conflict: that label equals y_c. No test counts enter the definitions.
Report conflict/non-conflict mass means and quartiles; any-conflict probability
thresholds .01/.05; price (1−p_ctx(y_c))/p_MLM(w*) for the most plausible conflict.
No candidate gives price ∞. Distributions include those infinities, with empirical
nearest-rank quartiles; report shares at price ≤.1/.2. Price is **only a plug-in
estimate**, never a certificate. Keep per-context arrays for review.

## 3. Room and audit

Generate once per context selected at the lowest threshold; reuse the nested
subset for higher thresholds. Five independent natural draws **with replacement**
from the allowed MLM distribution, plus up to three highest-probability conflict
and three non-conflict words. Natural draws outside either definition remain
unclassified and are logged; there is no resampling to manufacture conflicts.
Top-word candidates use only the training lexical criteria and MLM probability.

Insert each word at the compressed single-token slot. Run all six existing heads
on the inserts and original sentences, using original mean-WordPiece vectors
for originals and the single inserted-token vector for inserts. CoNLL BIO
probabilities are summed over B/I before the five-type argmax. Layer 8 is in the
report; layer 12 stays in CSVs. The natural and top schemes are reported separately.

Average insertions of a class within each context, then average contexts equally.
For each class report context-label following, inserted-word-majority following
when that label differs from gold, and original accuracy on the same contexts.
The **main gap** is non-conflict minus conflict context-label following on the
intersection of contexts with both classes available under that draw scheme.
Always report the supporting context count. Marginal class supports can differ.

Classifier means/SD use seeds 0/1/2; row SE uses seed-averaged context values,
never insertions or seeds as independent rows. Flag supports below 50 rows.
Audit: uniformly sample 100 distinct contexts across both datasets with a
most-plausible conflict at τ=.9 (all if fewer than 100). Save original/inserted
sentence, slot, word, y_c, complete training label counts/distribution and all
head predictions. No gold relabelling or augmentation training is performed.

## Runtime

Before fitting, time 2,048 real masked-slot queries per task, including the MLM
head. The estimate assumes all test slots pass .9 and receive all eleven
insertions, plus context feature extraction, fitting and report allowances:
about **6.1 minutes**, allowing roughly **10–15 minutes wall-clock** for setup,
CPU batching and I/O. No rows are subsampled. Actual stage and wall-clock times
are written automatically.
