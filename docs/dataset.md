# Dataset

This snapshot uses the frozen paired G4 length-comparison campaign, not the older 370,738-row short-only benchmark.

- `short`: 13–49-nt G4-associated cores.
- `nt512` / `nt1024`: real hg38 windows centred on the same loci, not padded cores.
- Shared IDs and splits: 74,872 train / 12,343 validation / 2,824 test.
- Splits protect core groups, genomic-overlap components, exact/reverse-complement matches and screened long-sequence homology groups. Exhaustive pairwise Levenshtein checking was not performed.
- Test is not used for training, normalization or checkpoint selection.
- Unconditional LSTM used effective batch 512 with a dropped final incomplete effective batch: 74,752 training examples per epoch. Validation used all 12,343 rows.

## Tm cohort — weights not yet released

Finite G4STAB reference-core predictions with ensemble SD ≤5 °C yield 31,669 train / 5,000 validation / 1,123 test rows. The same qualified IDs apply to all three lengths.
Train-only normalization: mean 57.84716468707569 °C, sample SD 8.054890567895972 °C.
These are core-level computational predictions, not experimental melting temperatures or whole-window Tm.
The null branch of a jointly Tm-trained model uses this qualified subset and is not a full-data unconditional baseline.

Real training sequences and IDs are not included in this initial weights-and-generation release.

