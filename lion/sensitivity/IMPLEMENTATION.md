# Experiment 9, step 1 only: frozen-classifier sensitivity

No training, new selectors, calibration, checkpoint selection or penalty arms. Reuse the full-scale augmentation source `results/augmentation_full/source/28b4f1db440a/` and classifiers `results/augmentation_full/f3d6dc697455/train/seed{0,1,2}/model.pt`. Steps 2 and 3 remain unimplemented/on hold. The obsolete select-then-predict and average-of-logs proposals are not implemented.

Measure every focal word of raw CoNLL-2003, WNUT and WikiANN test sets. Every test row uses both g1 and g2; this run does not measure training rows. The frozen raw scorer assigns O/entity. Per selector, keep the focal and round(rho[group] × context count) highest-ranked context words. Random control drops the same number at uniform context positions. Protect every focal WordPiece and specials; expose original dropped-piece counts as in calibration. One MLM pass sees all dropped pieces masked simultaneously; draw independently once per piece for each of four fills. No criterion selects draws. Fixed generation seeds and cached focal word-mean vectors are shared by all nine classifier checkpoints and the scorer.

Compute gamma=.01 uniform-mixed probabilities over the original nine CoNLL BIO tags, also on transfer sets; do not map to shared entity types for these measures. Per draw, compute KL(original || filled) and whether the winning tag changes. Average over four draws per word, then over words in raw-scorer groups O/entity/all. Identical focal vectors reuse original probabilities to avoid numerical movement on identical input. Report each selector and the per-word mean of both selectors. Standard errors use the SD of word-level draw-averaged values divided by sqrt(number of words); neither fills nor selector folds inflate row counts. Report all nine classifiers per seed and the scorer, plus three-seed means/SD. Seed-average row SE uses per-word seed means. Ratios are S_selector/S_random, with per-seed delta-method row SE; zero denominators produce nan. Source train cross-fitting is retained in source artifacts but not used to generate any new training examples.

Use existing per-fold/group absolute CE upper bounds from the augmentation report as label references. Their label-flip interpretation requires deterministic labels, adequate oracle predictors and selection-conditioned fills. The actual fitted scorer and calibration-style independent fills do not establish those assumptions. Mark references above 1 as loose. WNUT/WikiANN are explicitly uncertified. The all-group reference is group-frequency weighted and fold means average the existing references; these are descriptive combinations, not new jointly calibrated confidence bounds.

Report calibrated cuts per fold/group, mean dropped words and exact random/selector set coincidence. Separately report the share with all available context dropped: identical sets also arise with zero drops or accidental partial matches. Each mask kind receives its own four random fills, even when masks coincide. Raw row arrays retain sentence/focal keys. Main report tables summarize seeds; CSVs give all classifier/seed/fold/group values and SEs. Generation and measurement wall times are recorded separately.

Forge/Hydra stages: generate → measure → report.

```bash
USE_TF=0 .venv/bin/python -B -m lion.sensitivity.run
```

Outputs: `results/sensitivity/`, `report/RESULTS_9.md`, `report/tables/sensitivity/`. Full augmentation results are preserved under `report/old/08_augmentation_full/`.
