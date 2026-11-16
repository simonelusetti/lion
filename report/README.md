# Over-contextualization diagnostic report

This folder records the experiment requested in the [theory prompt](original_request.md): what was actually implemented, the measured results, and the limits of their interpretation.

- [Implementation](IMPLEMENTATION.md): inputs, sampling, recovered metadata, training, masking, E1–E7 algorithms, and departures from the requested protocol.
- [Results](RESULTS.md): all eight full-data baselines, three-seed diagnostic tables, observed support/counterexamples, and missing coverage.
- [Tables](tables/): unrounded aggregate and detailed CSV results.
- [Figures](figures/): the seven PNG figures from the experiments.
- [Provenance](provenance.json): run identifiers, source paths and checksums.

## Main findings

| Question | Observed outcome |
|---|---|
| Does context help? | Full context wins on five native tasks; word-only wins on UPOS. Ambiguity concentration is task-dependent. |
| Must the classifier be trained under masking? | Mask-trained classifiers have lower aggregate lexical KL and ECE on all six native diagnostic tasks. |
| Is nuisance Z readable from context-only vectors? | Yes, above mean chance in every eligible dataset; word-only also retains domain information. |
| Can readout erasure preserve Y? | Average loss is small for UPOS/shared NER, larger for dependency labels, and large for GUM. Some label-level Z readability remains. |
| Does within-label dependence match the proposed ordering? | Descriptive MMD ordering is reversed; no test survives BH correction at 5%. GUM and the matched ambiguity contrast lack sufficient support. |
| Does context-only always degrade most under shift? | No. Only shared NER follows that ordering in aggregate; individual domain-pair results are mixed. |
| Do neighboring entity substitutions affect predictions? | Full-view predictions flip in 11.78% of Few-NERD and 17.46% of CoNLL-2003 cases; word-only controls have zero flips. |

All eight baselines completed. Diagnostics remain the original six native datasets plus shared NER; later WikiANN/WNUT baseline repairs did not add diagnostic cases. Diagnostic training uses three seeds on a single fixed focal sample. Read the reported seed SDs as optimization variation, not confidence intervals or evidence of causal identification.
