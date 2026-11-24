# Why we moved away from the over-contextualization explanation

This is a synthesis of existing experiments, not a new run. It reconstructs the
empirical reason for changing direction; it does not assume access to the theory
conversation or claim that one experiment formally disproved over-contextualization.

**The central observation was that the classifier trained on original, uncleaned
sentences already reacted surprisingly little to extensive context changes.** This
was clearest for CoNLL’s large O group: even removing the lexical identity of every
other word usually left the prediction unchanged. Training a cleaner classifier
therefore did not address an obvious, widespread excess of context sensitivity in
that baseline. The more useful question became whether models exploit context
when it is needed to distinguish the labels of the same focal word.

The defensible claim is **strong focal-word anchoring under the measured
interventions, especially on CoNLL O rows**. “The model always uses only the focal
word” is stronger than the evidence supports.

## 1. The original baseline was already largely invariant

The step-1 experiment evaluated the existing baseline trained only on original
CoNLL sentences, alongside the augmented models. These baseline results were not
an effect of training on cleaned text. Its training pool was T1∪T2: 10,531 original
sentences. BERT and the MLP were frozen during measurement.

Four MLM fills per focal/mask/fold replaced dropped context while preserving the
focal word and all its pieces. Results average draws and selector folds within
word, then average words. O/entity groups are the frozen scorer’s predictions on
the original input, **not gold-label strata**.

| Baseline test group | Words | Selector-mask prediction flips | Random-mask prediction flips |
|---|---:|---:|---:|
| CoNLL O | 38,303 | 1.68% | 1.68% |
| CoNLL entity | 8,132 | 4.98% | 5.92% |
| CoNLL all | 46,435 | 2.26% | 2.42% |

Thus about **97.74% of baseline word/draw predictions stayed unchanged** under the
selector intervention, and **98.32% in the O group**. These are complements of
averaged flip rates, not the share of words that never flipped in any draw.

For O, both calibrated cuts were **ρ=0**: *every non-focal context word was dropped*.
The O group accounts for about 82.5% of the measured words. Its low sensitivity was
therefore not simply a consequence of retaining almost the whole sentence.
Selector and random masks are identical for O; their tiny difference comes from
independent fills, not a selector advantage. For entities, cuts retained 80–85% of
context, so the entity result is **not** a focal-only intervention.

Sources: [step 1, baseline table](../old/09_sensitivity_step1/RESULTS_9.md),
[full-scale training population and cuts](../old/08_augmentation_full/RESULTS_8.md).

## 2. It was not mainly an artifact of restoring the original context

Step 1b saved the filled token IDs and examined actual case-sensitive word changes.
For CoNLL baseline O rows:

| Quantity | Selector mask | Random mask |
|---|---:|---:|
| Mean context words dropped | 22.782 | 22.782 |
| Mean context words actually changed | 22.250 | 22.249 |
| Dropped-word occurrences restored identically | 2.336% | 2.339% |
| Entire draw restored / zero-drop draw | 0.082% | 0.079% |
| Prediction flips after excluding null draws | 1.684% | 1.678% |

**Roughly twenty-two context words changed, yet the original baseline prediction
changed in only about 1.7% of eligible draws.** Excluding null draws barely altered
the result. This was the strongest evidence that the initial low sensitivity was
real for these interventions, rather than a bookkeeping artifact.

Source: [step 1b, table A](../old/10_sensitivity_step1b/RESULTS_10.md).

## 3. A direct masking control removed the other words’ lexical identities

The layer experiment added a second perturbation: replace each dropped word with
one `[MASK]`, rather than an MLM fill. On CoNLL O rows, ρ=0 again means *all context
words* were masked. The new heads were trained on original full-context data;
these were not classifiers trained to consume cleaned inputs.

| Full-input-trained head, CoNLL O | MLM-fill flips | All-context-masked flips |
|---|---:|---:|
| Layer 8 | 1.90 ± 0.26% | 0.88 ± 0.13% |
| Layer 12 | 1.69 ± 0.30% | **0.68 ± 0.09%** |

Mean ± sample SD over three seeds. The layer-12 head was newly fitted in that
experiment, not the exact checkpoint from step 1. Nevertheless, it reproduced the
low-sensitivity pattern: approximately **99.32% prediction agreement after all
other words lost their lexical identities**.

This is the closest experiment to the “already using the focal word” observation.
It still leaves the focal word, positional/length information, sentence boundaries
and a pattern of mask tokens. It is not a mathematical isolation of word identity.

The encoder was not literally ignoring context: for layer-12 O rows, mean cosine
shift was 0.528 of a random-pair reference under fills and 0.641 under masking,
despite these small flip rates. **Context changed the representation substantially
without usually changing the readout decision.**

Source: [layer sensitivity, CoNLL O](../old/13_layer_sensitivity/RESULTS_13.md).

## 4. The apparent selector advantage weakened under matched comparisons

For CoNLL entity rows, raw selector/random sensitivity was 0.1024/0.1283, a ratio
of about 0.80. But selector and random edits differed in restoration, actual edit
size and distance. Matching changed-word count and nearest-change distance gave:

| Original-data baseline | Matched selector/random S ratio | Seed SD | Word-row SE |
|---|---:|---:|---:|
| CoNLL entity | 1.1367 | 0.0111 | 0.1109 |
| Few-NERD entity replication | 1.0001 | 0.0068 | 0.0379 |
| EWT dependency relations | 0.9300 | 0.0045 | 0.0186 |

S is KL(original prediction distribution ‖ filled distribution), not accuracy.
Cell ratios use common word support and selector-frequency weights; cells below
50 paired words are omitted. The CoNLL point estimate reversed, with broad row
uncertainty; Few-NERD’s matched contrast was essentially absent. **EWT retained a
smaller contrast**, so disappearance was not universal.

This weakened the claim that selectors had isolated a uniquely harmful kind of
context. It does not itself prove that models are lexical-only.

Sources: [CoNLL matching](../old/10_sensitivity_step1b/RESULTS_10.md),
[Few-NERD replication](../old/11_fewnerd_replication/RESULTS_11.md),
[EWT replication](../old/12_deprel_replication/RESULTS_12.md).

## 5. There was little evidence that further suppressing sensitivity solved the task

In the earlier full-scale, label-preserving augmentation comparison, all arms
used the same original training population and equal step/token budgets, and all
were evaluated on the same raw test sets:

| Raw native CoNLL test | Span F1, mean ± seed SD |
|---|---:|
| Original-only baseline | **84.27 ± 0.29** |
| Selector-augmented | 83.79 ± 0.37 |
| Matched random-augmented | 83.62 ± 0.12 |

The augmented arms became less sensitive, but that was not accompanied by a
consistent generalization benefit: transfer results were mixed. Lower sensitivity
was therefore not, by itself, evidence of a better classifier.

Source: [full augmentation comparison](../old/08_augmentation_full/RESULTS_8.md).

## 6. Why the replacement question became “does it use context when necessary?”

Later natural-pair measurements used real held-out Few-NERD sentences containing
the exact same case-sensitive word with different gold coarse labels. No generated
labels or MLM edits were needed. On the identical clean comparison cohort:

| Both distinct labels predicted correctly | Small pilot scorer | Full-data scorer |
|---|---:|---:|
| All coarse labels, including O | 42.86% | **54.10%** |
| Entity-type versus entity-type | 33.27% | **47.33%** |

These are pair-weighted results, not token accuracy; frequent ambiguous forms
contribute more pairs. Pairs come from different test sentences, with exact
fitted-sentence matches and duplicate test sentences excluded. They should not be
treated as independent observations for confidence intervals.

This does not make the generator adequate, or establish a universal architectural
inability to use context. It shows that even naturally occurring, gold-labelled
contextual distinctions are difficult for the scorers—and that substantially more
training improves them. **Using those same scorers to admit only confident label
changes can hide precisely the contextual distinctions we want to learn.**

Sources: [pilot natural-pair numbers](../../results/natural_pairs/numbers.json),
[full scorer and same-cohort comparison](../../results/fewnerd_full_grid/natural_pairs/numbers.json).

## What we should—and should not—say

A concise account of the shift is:

> We initially investigated harmful excess dependence on context. The original,
> uncleaned classifier was already highly invariant to large context perturbations,
> most strikingly on CoNLL O words even after all other lexical identities were
> masked. The selector/random contrast weakened after matching the interventions,
> and further invariance did not consistently improve generalization. This shifted
> the question toward learning and recognizing the contextual distinctions that
> actually change a word’s label, rather than simply discarding context.

Two limits matter. First, the early E1 experiment trained separate full/word-only
heads and found genuine context benefits: on its small CoNLL diagnostic sample,
full-view accuracy was 89.57% versus word-only 85.17%, and macro-F1 was 57.95%
versus 41.18%. Those scores use a different sampled protocol and are not comparable
to the full-scale span-F1 table above. Second, entity and dependency interventions
were more consequential than CoNLL O interventions, and generated contexts do not
come with verified gold labels. **Stable predictions can be appropriate when the
true label stays stable.**

Consequently, the evidence supports retiring over-contextualization as our default
explanation for these results. It supports investigating insufficient or poorly
read-out task-relevant context; it does **not** prove that every classifier, token
or context is dominated by the focal word.

Counter-evidence source: [E1 full versus word-only results](../old/01_baselines_e1_e7/RESULTS.md).
