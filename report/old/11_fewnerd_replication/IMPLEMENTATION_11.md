# Few-NERD replication of sensitivity steps 1 and 1b

Command: `USE_TF=0 .venv/bin/python -B -m lion.fewnerd_replication.run`.
Config: `conf/fewnerd_replication.yaml`. Each stage runs through Forge/Hydra;
outputs are in `results/fewnerd_replication/`. Reports are generated automatically.

## Data and roles

Use cached `DFKI-SLT/few-nerd`, supervised split, revision
`205f3e9c9f3577ea2561d43f2f62dc249ab92d5b`. Preparation saves source row IDs,
file SHA-256 hashes, sampling seed, split counts and label conversion in
`source/prepare/{sample.csv,counts.csv,provenance.json}`.

The coarse labels are IO. Convert to seventeen BIO tags: O; B/I for art,
building, event, location, organization, other, person, product. A non-O type
change starts B; continuation uses I. Adjacent entities with the same coarse
type merge because coarse IO contains no boundary information; fine labels are
not used to infer boundaries. This convention applies to training and scoring.

Seed 0 selects **10,531 source training sentences**, as explicitly confirmed:
2,106 scorer fit, 527 scorer validation, 3,949 T1, 3,949 T2. Classifiers train on
**7,898 original sentences**, excluding all scorer split A. CoNLL instead used
14,041 source / 10,531 classifier sentences. Sample 2,438 native validation
sentences, preserving CoNLL's 3,250/14,041 proportion. Calibration uses 70% of the
opposite selector half; the remaining 30% are held-out certificate sentences.
Within each sentence/group the existing protocol samples one focal word.

Test sentences are randomly ordered and retained until 46,435 word tokens;
whole-sentence overshoot is permitted. The actual sample is 1,870 sentences,
exactly 46,435 words. No selected sentence was truncated at 512 WordPieces.
Span membership statistics use exact case-sensitive gold strings in T1∪T2,
counting spans once, not once per constituent word.

## Part 1

Reuse `certification.selector` and `certification.calibrate` directly: frozen
bert-base-cased, block 12, mean over a word's pieces, MLP scorer 768→256→17;
selector [focal; candidate; sentence mean] → 128→64→1. Train scorer ten epochs,
selectors five epochs on all eligible focal words, four calibration-style fills
per training query, ten draws per calibration query. Same original hyperparameters,
ε=.05, δ=.1, descending cut grid 1,.95,…,0 and Learn then Test procedure.

After scorer fitting, set calibration γ to its token error rate on its own
527-sentence validation split, before selector calibration. No test labels enter
this choice. Save effective γ and the source config.

Reuse augmentation held-out bounds and absolute-CE reporting, without augmentation
or unrelated CRT stages. Save cuts per fold/group and absolute averaged-predictor CE
(mean, SD, n, mean + 1.645 SD/√n), alongside CE increase. The unchanged gate rejects
when the held-out CE-increase **lower** bound exceeds ε; ρ=1 passes by identity.
A failed group is measured anyway and explicitly marked UNCERTIFIED. Passing
this screen is not a new upper-bound certification claim.

Only baseline classifiers are fitted, seeds 0/1/2, reusing the matched baseline's
original-sentence schedule: ten epochs, 64 sentences per batch, mean word loss,
AdamW lr=.001, weight decay=.01, dropout=.1, gradient norm cap 1. Checkpoints use
raw native-dev **token macro-F1 across all seventeen BIO/O tags**, matching the
existing CoNLL baseline criterion, not span F1. Scorer uses its own A validation.
Report scorer and baseline exact native-BIO test span F1 with existing span code.

## Part 2

Reuse `sensitivity_diagnostics` through a Few-NERD data adapter. Generate **four
independent fills per row, mask and fold**, saving token IDs immediately with
word/piece offsets. Selector and random masks have identical drop counts, and
fills use one MLM pass seeing only kept context; dropped piece counts are preserved.
Raw group assignment uses the scorer argmax. All baseline/scorer analyses share
this single cache; no fills or masks are selected after measurement.

Keep step-1 sensitivity smoothing **γ=.01**, over the native seventeen tags.
Calibration γ above is separate. S is KL(original || filled); flip is argmax
change. Average draws within each selector fold, then g1/g2 within each focal word.

Run unchanged step-1b analyses: restoration/null swaps and paired non-null means;
changed-word and nearest-change bins; dropped-position distances; selector-frequency
weighted cell ratios; exact seen-span categories and original errors; focal-form
ambiguity; flip concentration and top twenty forms. Baseline means/SD use three
seeds, row SEs treat words as units. Folds/draws are never independent rows.
Bins below fifty paired rows are marked; ratios omit those cells and zero-denominator
cells. Scorer-predicted entity rows with gold O remain a separate no-gold-entity bin.

Every final diagnostic table shows CoNLL beside Few-NERD, with selector/random
side by side and common support, bins and weights. CoNLL reference values come
from the completed step-1b arrays and tables. Original CoNLL files remain unchanged.

## Runtime

Before training, use measured CoNLL stage times scaled by sentence/focal
WordPiece work (sum over words of length + length²/768). Estimated Part 1:
**3.88 h**, Part 2 **17.1 min**, total **4.17 h**, assuming comparable GPU
throughput (rough uncertainty ±50%). This is under six hours, so retain the full
test sample. The script halves only the test word target if the initial estimate
exceeds six hours. `estimate.json` retains both the initial and used projections;
stage runtimes and actual pipeline wall-clock are written to the results.

Differences from CoNLL required by this replication: dataset/type inventory,
explicit 10,531-source population, and scorer-derived calibration γ. Sensitivity
γ, masks, fill procedure, estimators and classifier recipe remain unchanged.
