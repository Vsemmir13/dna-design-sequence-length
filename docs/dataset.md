# Dataset

Current model comparisons use a qualified subset of a paired G4 length-comparison dataset, not the older 370,738-row short-only benchmark.

- `short`: 13–49-nt G4-associated cores.
- `nt512` / `nt1024`: real hg38 windows centred on the same loci, not padded cores.
- Parent IDs and splits: 74,872 train / 12,343 validation / 2,824 test.
- Splits protect core groups, genomic-overlap components, exact/reverse-complement matches and screened long-sequence homology groups. Exhaustive pairwise Levenshtein checking was not performed.
- Test is not used for training, normalization or checkpoint selection.
- The older full-data unconditional LSTM is a historical reference, not the current matched-data baseline.

## Tm cohort

Finite G4STAB reference-core predictions with ensemble SD ≤5 °C yield 31,669 train / 5,000 validation / 1,123 held-out test rows. The same qualified train/validation IDs apply to all three lengths and all current G4 models, including the retrained unconditional LSTM. No test rows enter training, normalization, checkpoint selection or the current validation-based generation comparison.
Train-only normalization: mean 57.84716468707569 °C, sample SD 8.054890567895972 °C.
These are core-level computational predictions, not experimental melting temperatures or whole-window Tm.
The null branch of a jointly Tm-trained model uses this qualified subset. The current unconditional LSTM is a separate fit on this same subset, with no temperature encoder. Its effective batch size is 512: 31,232 training rows are used per epoch after dropping the incomplete effective batch; validation uses all 5,000 rows.

## Uploaded data

The [Hugging Face repository](https://huggingface.co/vsemmir13/return-of-the-lstm-nonb-dna/tree/main/datasets/g4_tm) contains the qualified train/validation subset:

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
The complete 74,872-row parent training cohort, original source interval data, hg38 FASTA and predictor weights are not uploaded. Do not mix older full-data LSTM results with the current matched-data comparison.

## Z-DNA-associated data

`datasets/z_dna/nt256/` on HF contains 4,237 training and 585 validation mouse sequences, each 256 nt, from the untreated wild-type Z22 ChIP-seq subset of GEO GSE224553. They are positive Z-DNA-associated windows, not experimentally confirmed isolated Z-DNA structures. Source processing and the chromosome-based split are described in the [dataset card](https://huggingface.co/vsemmir13/return-of-the-lstm-nonb-dna/blob/main/datasets/z_dna/README.md). Test rows and Tm labels are not uploaded; no new Z-DNA model results are claimed here.

