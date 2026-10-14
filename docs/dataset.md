# Dataset

This snapshot uses the frozen paired G4 length-comparison campaign, not the older 370,738-row short-only benchmark.

- `short`: 13–49-nt G4-associated cores.
- `nt512` / `nt1024`: real hg38 windows centred on the same loci, not padded cores.
- Shared IDs and splits: 74,872 train / 12,343 validation / 2,824 test.
- Splits protect core groups, genomic-overlap components, exact/reverse-complement matches and screened long-sequence homology groups. Exhaustive pairwise Levenshtein checking was not performed.
- Test is not used for training, normalization or checkpoint selection.
- Unconditional LSTM used effective batch 512 with a dropped final incomplete effective batch: 74,752 training examples per epoch. Validation used all 12,343 rows.

## Tm cohort

Finite G4STAB reference-core predictions with ensemble SD ≤5 °C yield 31,669 train / 5,000 validation / 1,123 test rows. The same qualified IDs apply to all three lengths.
Train-only normalization: mean 57.84716468707569 °C, sample SD 8.054890567895972 °C.
These are core-level computational predictions, not experimental melting temperatures or whole-window Tm.
The null branch of a jointly Tm-trained model uses this qualified subset and is not a full-data unconditional baseline.

## Uploaded data

The private [Hugging Face repository](https://huggingface.co/vsemmir13/return-of-the-lstm-nonb-dna/tree/main/datasets/g4_tm) now includes the qualified Tm train/validation subset:

```text
datasets/g4_tm/
  short/{train.csv,validation.csv}
  nt512/{train.csv,validation.csv}
  nt1024/{train.csv,validation.csv}
  README.md
  schema.json
```

Each regime contains 31,669 train and 5,000 validation records. ID order, splits, labels and all exported source fields are preserved. No test rows or test IDs are uploaded. Test remains held out locally; it is still needed for final evaluation.
`sequence` is the DNA target. `core_predicted_tm` and `core_g4stab_std` are the core prediction and ensemble uncertainty in °C. `sequence[core_start_offset:core_end_offset]` is the labelled reference core, using zero-based, end-exclusive offsets.
Use `split` for the current partition (`train`/`val`); `source_split` is earlier provenance. CSVs retain the frozen parent schema, described in `schema.json`.
The complete 74,872-row unconditional training cohort, original source interval data, hg38 FASTA and predictor weights are not uploaded. Do not treat this smaller qualified Tm subset as the full-data unconditional cohort.

