# lion

Sequence labelling with frozen BERT: change the representation before classification,
measure what context and depth do, and keep the classifier separate.

The pipeline is **dataset → frozen encoder → representation method → tagger**.
Forge manages each stage and its config. Numerical operations batch words, focal
positions, and masks as PyTorch tensors; DataLoaders batch whole sentences.

## Setup

Use a Python environment with CUDA-enabled PyTorch on the training machine.
The local Forge checkout must be at `../forge`.

```bash
cd /home/simonelusetti/lion
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
source .venv/bin/activate
```

Examples below show the command to launch. When asking an agent to execute an
experiment, it must launch in tmux and leave it running without monitoring it.
For example, from the project directory:

```bash
tmux new-session -d -s lion -c "$PWD" '.venv/bin/forge run method=global_mix; exec bash'
```

## Configs and stages

The defaults are Few-NERD coarse NER, frozen `bert-base-cased`, layer 12,
a two-layer MLP, seed 0, and CUDA. Config groups select `data`, `encoder`,
`method`, `tagger`, `stage`, and `runtime`.

```bash
forge run stage=data
forge run stage=cache
forge run method=last
forge run method=layer method.layer=8 tagger=linear
forge run method=global_mix
forge run method=soft_mix
forge run method=hard_mix
forge run method=head_mix
forge run method=layer_head_mix
```

`stage=data` tokenizes and caches datasets. `stage=cache` extracts all 13
word-level hidden states, with attention contexts too for head-mixture methods.
Training a layer method also builds its cache when needed. The caches are
shared across seeds and compatible layer methods, and sharded so the complete
bank does not have to fit in RAM. `cache.enabled=false` runs the encoder live.
`cache.dtype=float16` controls storage; batches are converted to float32.

Defaults use complete datasets. For a smaller study, set explicit caps and use
the same overrides for its data, mask, training, and analysis stages:

```bash
forge run data.cap.train=20000 data.cap.validation=1000 data.cap.test=1000
```

### Datasets

`fewnerd` is the main NER dataset, with O plus eight coarse entity types.
Also available: `wikiann`, `conll2003`, `wnut`, `ud_upos`, `ud_deprel`,
`ud_discourse` (GUM), and `conll2000`. Movie rationales are excluded.

Native train/validation/test splits remain separate. CoNLL-2000 has no native
validation split, so `data.validation_fraction=0.1` holds out training sentences.
`data.seed` controls this split and capped sampling; `seed` controls model training.
Words use their first WordPiece; skipped source-token IDs are packed consistently.
GUM words with no assigned discourse relation are excluded from the objective.

### Encoders

`encoder=bert-base-cased` (default) and `encoder=bert-base-uncased` select separate
checkpoints and their matching tokenizers. Both inherit the shared BERT family
config (`conf/encoder/_bert.yaml`) and use the same frozen BERT implementation.
`encoder=bert` is an alias for the cased config.

```bash
forge run encoder=bert-base-cased
forge run encoder=bert-base-cased method=global_mix
```

`encoder.family` selects the architecture implementation; `encoder.name` selects
the pretrained checkpoint. BERT-specific layer and attention access lives in
`lion/encoder.py`, including head projections used by representation methods.
Both checkpoints have 12 blocks, 768-dimensional vectors, and 12 attention heads.
The shared config sets a 512-WordPiece limit and disables [CLS]/[SEP]. Token and
feature cache keys include the encoder config, keeping cased and uncased data
separate.

### Representation methods

| Method | Representation before the tagger |
|---|---|
| `last` | Block 12 |
| `layer` | One configured block, including embedding layer 0 |
| `global_mix` | Learned global softmax over blocks 1–12 |
| `soft_mix` | Per-word layer weights predicted from ordinary block 12 |
| `hard_mix` | One layer per word, with straight-through softmax gradients |
| `head_mix` | Global weighted sum of attention-head writes, then parameter-free LayerNorm |
| `layer_head_mix` | Separate global layer/head mixtures and a learned two-way gate |
| `mask` | Learned context gates; classify block 12 from a second BERT pass |
| `random_mask` | Random hard context subsets, with an exact retained word count |
| `supervised_mask` | Learned sigmoid gates trained against saved hard masks |
| `oracle_mask` | Apply saved gold-informed masks directly; oracle analysis |

The tagger is independently configurable as `tagger=mlp` or `tagger=linear`.
The MLP is 768 → 256 → labels with ReLU and dropout by default. There is no
sequence decoder: each word is classified from its processed representation.

The context selector reads `[focal; candidate; sentence mean]` after LayerNorm,
using ordinary block-12 vectors. The focal word always remains visible.
Gates enter every BERT layer as `log(gate)` key-attention biases. Encoder weights
stay frozen while gradients flow through the second pass to the gates.
`method.gate=anchored` uses max-anchored exponentials; `method.gate=sigmoid`
allows independent gates. `method.penalty` controls mean retained context mass.

```bash
forge run method=mask train.batch_size=8 train.bucket=true
forge run method=mask method.gate=sigmoid method.penalty=0.05 train.batch_size=8
```

### Decoupled training

Use explicit checkpoint paths to separate classifier and selector learning.
`RUN1` and `RUN2` below mean actual Forge run folders under `outputs/xps/`.

```bash
# 1. Learn a classifier under random context masks.
forge run method=random_mask train.batch_size=8 train.bucket=true

# 2. Learn a selector against that fixed classifier.
forge run method=mask train.batch_size=8 train.bucket=true \
  train.classifier_source=RUN1/model.pt 'train.freeze=[tagger]'

# 3. Fix the selector and train a classifier on its selected contexts.
forge run method=mask train.batch_size=8 train.bucket=true \
  train.selector_source=RUN2/model.pt 'train.freeze=[method]'
```

Stage 3 starts a fresh classifier. Add `train.classifier_source=RUN1/model.pt`
to initialize it from stage 1. Freezing a classifier also disables its dropout.
The architecture and gate config must match the checkpoint being loaded.

### Hard masks and supervision

```bash
forge run stage=masks masks.split=train
forge run method=supervised_mask train.batch_size=8 train.bucket=true \
  'masks.sources=[MASK_RUN/records.pt]'
```

Mask construction uses same-word full-context **training** reference clouds,
excluding the focal occurrence from its own training cloud. `masks.criterion`
selects kNN majority support or nearest-label margin; metric, k, eligible forms,
and the focal sample are configurable. Focals are sampled round-robin across
form/label groups. Only full-context-supported occurrences receive a minimum mask.
Other occurrences use full context as a control.

Up to `masks.exhaustive_context=12` other words, all binary subsets are evaluated.
Longer sentences use `masks.samples` random subsets plus empty/full controls;
their reported minima are approximate. `masks.policy` selects a representative
minimum, their union, or their intersection. Inspect `masks.jsonl` for support,
minimum cardinality, multiplicity, and the selected context.

Supervision uses balanced positive/negative context BCE weighted by
`method.supervision`, together with classification CE. Evaluation predicts gates
from ordinary representations; it uses no oracle masks or gold labels as inputs.

For an oracle classifier comparison, generate masks separately for train,
validation, and test, then supply all three files to `method=oracle_mask`.
That condition uses gold labels in preprocessing and must be reported as an oracle.

### Layerwise analysis

```bash
forge run stage=layerwise
forge grid grids/layerwise.yaml
forge run stage=analyze 'analysis.sources=[SEED0_RUN,SEED1_RUN,SEED2_RUN]'
```

`stage=layerwise` trains independent probes for layers 0–12 using identical
classifier initialization. Each probe selects its checkpoint on validation.
Artifacts include per-layer scores (`layers.csv`), per-token CE and true-label
margins (`per_token.pt`), and per-form best layers against training label entropy
(`forms_0.csv`). Entropy is undefined for forms unseen in training.

`analysis.json` reports the gold-informed per-token best-layer oracle. With
multiple seed runs, it also selects the layer using one seed's true-label CE and
evaluates another seed at that layer. These remain gold-informed ceilings.

For independently tracked single-layer experiments or broad baseline grids:

```bash
forge grid grids/layers.yaml
forge grid grids/baselines.yaml
```

The initial tagger baseline grid trains one last-layer MLP per dataset, using
frozen `bert-base-cased`, seed 0, all available sentences, and 10 epochs. Each
checkpoint is selected on validation and evaluated on test:

```bash
forge grid grids/tagger_baselines.yaml
```

### Evaluation and artifacts

```bash
forge run stage=eval evaluation.source=RUN/model.pt
forge run stage=eval data=wnut evaluation.source=FEWNERD_RUN/model.pt
forge metrics
forge artifact '*.pt'
```

Evaluation reconstructs encoder, method, and classifier settings from the saved
checkpoint. Few-NERD → WikiANN/CoNLL-2003/WNUT evaluation projects predictions
and gold tags to O/PER/ORG/LOC and reports shared token and span scores.

Each Forge run contains its config and logs. Training writes `model.pt` (best
validation macro F1), `last.pt`, `report.json`, and `predictions.pt`. Prediction
artifacts include logits, labels, sentence IDs, and word positions. BERT weights
are loaded from Hugging Face rather than duplicated in every checkpoint.

## Device and batching

`runtime=cuda` defaults to CUDA:0, bf16 autocast, TF32, and all visible GPUs.
PyTorch DataParallel distributes sentence forwards, cached-vector readouts,
and expanded focal/mask forwards. Focal masking computes each sentence's ordinary
vectors once on the primary GPU before distributing the second passes.

```bash
forge run runtime=cpu
forge run 'runtime.devices=[1,2]' runtime.device=cuda:1
forge run runtime.parallel=false runtime.device=cuda:0
```

`train.batch_size` is the number of sentences per optimizer step.
`runtime.batch_size` controls extraction/evaluation sentence batches.
`runtime.focal_chunk` and `runtime.attention_budget` limit expanded focal batches;
each chunk backpropagates immediately so masked-pass graphs do not accumulate.
`runtime.workers` controls DataLoader workers.

## Over-contextualization diagnostics

The independent `lion.diagnostics` modules implement the diagnostic protocol in
`results/plan.md`, using the same frozen encoder and MLP as the baselines. Baseline
code and checkpoints are not changed. Here masking replaces WordPieces with
`[MASK]` while leaving attention on; it does not use the attention-gating methods.

```bash
# Inventory, paired view cache, E1–E7, and final report, each tracked by Forge.
.venv/bin/python -B -m lion.diagnostics.run runtime.workers=0 runtime.batch_size=32

# Reproduce an individual experiment after inventory and cache extraction.
.venv/bin/python -B -m lion.diagnostics.e1 runtime.workers=0
.venv/bin/python -B -m lion.diagnostics.e5 runtime.workers=0
```

`conf/diagnostics/default.yaml` controls datasets, three training seeds, fixed
focal sampling, probe support, permutation counts, and substitution budgets.
The initial paired sample caps are 4,096/1,024/2,048 train/validation/test focal
tokens per dataset, with at most two per sentence. All three views and seeds
share the same positions. GUM uses larger caps of 16,384/3,200/3,000 focal tokens
to support within-label genre comparisons. Frequency and label entropy use full training data.
UD genre/document metadata is recovered from cached raw sentence IDs and checked
against token alignment; unsupported datasets/cells are skipped with reasons.

CSV tables, PNG figures, and readout checkpoints are under `results/e1` through
`results/e7`; `results/samples.csv` records actual counts and `results/report.md`
separates measurements from interpretation. Cached last-layer focal vectors are
under `data/cache/diagnostics`, distinct from the baseline feature banks. Sparse
focal samples receive token metrics, not span scores. Dataset identity, erasure,
and substitutions provide diagnostics under the assumptions documented in the
plan; they do not by themselves establish a causal nuisance effect.

The latest study is documented in [report/README.md](report/README.md), with
earlier experiment batches archived under [report/old/](report/old/README.md).

## Port from exploration

The port combines `tagger/model.py`, `tagger/masked.py`, `tagger/decoupled.py`,
`tagger/supervised_masked.py`, layerwise probes, layer/head mixtures, hard-mask
geometry, and NER transfer evaluation into the modules here. It does not import
`exploration` or require its checkpoints and caches.

The new protocol keeps native splits, makes dataset caps explicit, and selects
checkpoints on validation. Inputs default to no [CLS]/[SEP], preserving the old
encoder convention. These choices mean newly trained numbers are not exact
reproductions of the old combined-split reports.

The earlier exploratory reports and specialized post-hoc geometry studies remain
in `../exploration` as research references. Add new experiments here as small
methods or stages with configs. Working conventions are recorded in `AGENTS.md`.

## Mask-rule pilot grid

Run the shared caches, frozen scorers, subset selection and independent rule × penalty cells with:

```bash
.venv/bin/python -B -m lion.pilot.run runtime.workers=0 runtime.batch_size=64
```

The full product is in `grids/pilot_cells.yaml`; the default runner batches its 1,320 independent cells into 60 vectorized fits. Protocol and sampling choices are in `lion/pilot/IMPLEMENTATION.md`.

## Cross-fitted context selector

```bash
.venv/bin/python -B -m lion.certification.run
```

The default experiment trains HardKuma selectors on EWT UPOS and dependency labels,
then cleans the opposite training folds with accepted same-mask MLM fills.
The cleaned and original classifiers are evaluated on raw EWT and GUM sentences.
The run reports sufficiency and minimality CRTs and both held-out cross-entropy bounds.
The [protocol](lion/certification/IMPLEMENTATION.md)
describes the baselines, sampling, and result files.

Run all eight source/task configurations with a two-hour timing reference:

```bash
python3 -B -m lion.certification.run --suite
```

The suite starts with WNUT, CoNLL-2003, WikiANN, and Few-NERD. Each NER source
is evaluated in domain and on the other NER corpora. Check
`results/certification/suite.csv` for each configuration's timing and status.
