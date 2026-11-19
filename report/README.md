# Cross-fitted context selector

Experiment 7 trained two CoNLL-2003 NER selectors and compared classifiers trained on joint-fill and raw inputs. Both selectors kept zero context tokens on their certification samples. The sufficiency tests rejected, and the cleaned classifier lost span F1 on CoNLL-2003, WNUT, and WikiANN.

- [Results and diagnosis](RESULTS_7.md)
- [Current selector protocol](../lion/certification/IMPLEMENTATION.md)
- [Experiment tables](tables/certification/)
- [Earlier experiment batches](old/README.md)
