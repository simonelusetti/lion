# Experiment 8: original-plus-fill augmentation

The absent word-rank artifacts are rebuilt once, as authorized, using `certification=conll2003_pilot`: source training cap 3,000 sentences, seed 0, scorer fit/validation separated from selector folds. T1 and T2 have 1,125 sentences each; the comparison uses exactly their union, excluding both scorer partitions. Sentence IDs and actual word counts are saved. This is the matched-baseline population of the rebuilt current pilot, not a reproduction of the historical HardKuma report.

Stages reuse the existing certification preparation, scorer, word-ranking selector and calibration functions. Freeze them before augmentation. Held-out bounds reuse the calibrated test sentences and existing independent-fill CE certificate; the unrelated chained CRT diagnostic is not rerun. No gate suppresses the requested classifier comparison.

For every labelled word in T2 use g1; for every labelled word in T1 use g2. Raw scorer predictions choose O/entity group. The selector keeps the focal plus round(rho[group] × context words). The random control drops exactly the same count, uniformly across eligible context positions. All pieces of dropped words are masked together. Each arm takes exactly one independent piece fill from that single MLM pass, preserving piece counts, all focal pieces and specials. No score selects, rejects or regenerates a draw. Scorer disagreements are computed only afterward. Identity variants remain in the dataset.

Cache one focal vector per variant, using the source pipeline’s mean pooling over each word’s pieces. Originals and variants retain only their appropriate word labels; variants never contribute loss on other words. Fixed generation RNGs are shared across all classifier seeds and epochs.

Three independently parameterized 768→256→labels MLPs train with seeds 0/1/2, shared initialization within seed. Match baseline AdamW lr .001, decay .01, dropout .1, clipping 1, ten epochs and raw-dev macro-F1 checkpoint selection. The source baseline batches 64 original sentences with varying labelled-token counts. Those counts define the same token budget per optimizer step for all arms. Baseline takes originals only; augmented arms take half originals and half variants, with cumulative rounding for odd counts. Their original examples follow exactly the same sentence/word stream as baseline, at half speed. An original sentence’s cached word rows can straddle adjacent batches. All arms have equal total labelled-token presentations and optimizer steps; ten baseline-budget epochs correspond to five passes over originals and five over variants in the augmented arms. Save the common sentence permutations and token budgets per seed.

Evaluation is exclusively raw CoNLL-2003, WNUT and WikiANN. Call the existing certification `evaluate` and `span_counts` functions, including their exact native/shared mapping. Report precision, recall and span F1 per seed, mean/sample SD, and seed-paired selector−baseline / selector−random differences. Retain the source CoNLL native and shared results. No test metric selects checkpoints.

Per group, report held-out fitted-scorer CE increase and its one-sided normal UCB (z=1.645), using the calibrated cuts. These are in nats, not guaranteed inherited-label error probabilities; an information-loss interpretation needs oracle/scorer and fill-conditional assumptions. Calibration-style fills are not selection-conditioned fills. Report scorer disagreement as a biased label-flip proxy, variant counts, changed-input share and 50% variant-token share. Save generation and training wall times separately, as well as end-to-end time including rebuilt prerequisites.

Forge/Hydra entry point:

```bash
.venv/bin/python -B -m lion.augmentation.run
```

Config: `conf/augmentation.yaml`. Raw runs and source models: `results/augmentation/`. Results: `report/RESULTS_8.md` and compact CSVs; prior report archived under `report/old/07_certification/`.

Report additions: absolute gamma-smoothed fill-averaged CE is reconstructed on the same held-out rows/fills as saved excess CE plus raw smoothed CE. Report n, mean, sample SD and mean+1.645 SD/sqrt(n), alongside the original excess-CE bounds. Explicitly tabulate recalibrated fractions per selector/group. The scorer’s own A-validation predictions supply native span F1 and unsmoothed, word-mean full-context CE (also save mean sentence-summed NLL). Dev macro-F1 means the unweighted token BIO-tag F1 average including O, not span F1; both the existing matched-baseline fitter and all current arms use this criterion. These additions do not alter training, generation or checkpoint selection.

## Full-scale run

All source training sentences are used before splitting. Sentence and selector-focal caps are null; all held-out certificate sentences are used. All three arms still exclude A. Results and rebuilt prerequisites are isolated under results/augmentation_full/. The 3,000-sentence pilot is archived under report/old/08_augmentation_pilot/.
