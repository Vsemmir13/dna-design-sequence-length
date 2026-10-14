# Return of the LSTM: Non-B DNA Generation

Code and architecture snapshot for G-quadruplex-associated and Z-DNA-associated sequence generation.
The repository slug is `return-of-the-lstm-nonb-dna`; regulatory enhancer/promoter experiments are outside its scope.

This private research snapshot initially contains the **new paired G4 campaign**, not the historical manuscript runs.
Completed training and verified transfer do not establish biological quality. New-campaign generation metrics are **PENDING**.

## Code and artifacts

- GitHub: [Vsemmir13/return-of-the-lstm-nonb-dna](https://github.com/Vsemmir13/return-of-the-lstm-nonb-dna).
- Weights, frozen generated cohorts and processed Tm datasets: [vsemmir13/return-of-the-lstm-nonb-dna on Hugging Face](https://huggingface.co/vsemmir13/return-of-the-lstm-nonb-dna).
- GitHub contains architecture classes, a safe weight loader, configurations, documentation and tests. It contains no checkpoints, generated cohorts, dataset CSVs, credentials or cloud-operation scripts.
- Hugging Face contains tensor-only `model.safetensors`, per-run metadata, original 5,000-row generated cohorts, training CSVs, processed Tm train/validation datasets and SHA-256 inventories. Optimizer states, original pickle checkpoints and held-out test rows are not uploaded.

## Processed Tm datasets

On Hugging Face, `datasets/g4_tm/{short,nt512,nt1024}/{train.csv,validation.csv}` contains 31,669 train and 5,000 validation rows per regime. The three lengths share frozen sample IDs, splits and reference-core Tm labels. No test rows or test IDs are uploaded.
Short sequences are 13–49 nt; long sequences are real 512/1,024-nt hg38 windows. Tm is a G4STAB prediction for the embedded core, not the entire window. See [dataset notes](docs/dataset.md).

## Initial verified weights

| Family / task | Sequence regime | Parameters | Training epochs | Selected epoch | Generated sequences |
|---|---|---:|---:|---:|---:|
| LSTM / unconditional | G4-associated cores, 13–49 nt | 3,662,383 | 50 | 15 | 5,000 |
| LSTM / unconditional | G4-containing hg38 context, 512 nt | 3,662,383 | 50 | 31 | 5,000 |
| LSTM / unconditional | G4-containing hg38 context, 1,024 nt | 3,662,383 | 50 | 38 | 5,000 |

The models share the same architecture and were independently initialized with seed 42.
Best checkpoints minimize validation cross-entropy after all 50 epochs. No test rows were used for training or selection.
The new Tm-LSTM and Discrete Flow Matching weights are **not in this release**; they require completed runs and verified retrieval.
Z-DNA artifacts will be added as a separately identified release, not relabelled as these G4 runs.

## Layout

```text
models/
  lstm/unconditional/architecture.py
  lstm/conditional/architecture.py
  discrete_flow_matching/architecture.py
configs/lstm/unconditional/{short,nt512,nt1024}.json
docs/{dataset,lstm,discrete_flow_matching}.md
scripts/load_lstm.py
tests/test_architectures.py
release_manifest.json
```

LSTM architecture files contain only `torch.nn.Module` classes. The Discrete Flow Matching file is our continuous-Tm adapter and a loader for pinned upstream source; it does not vendor the author's CNN.
Training and full evaluation entrypoints are not part of this initial code snapshot.

## Load a released LSTM

```bash
python -m pip install -r requirements.txt
hf auth login                 # access to the private weights repository
python scripts/load_lstm.py --regime short
```

The loader downloads only the chosen configuration and tensor-only weights, checks their hash, loads the exact architecture strictly, and performs a CPU forward check. It does not train or generate a new scientific cohort.

## Interpretation

Short sequences are G4-associated cores; 512/1,024-nt sequences are real genomic contexts, not padded cores or experimentally established full-length quadruplex structures.
Unconditional LSTM has no Tm encoder, class encoder or neural length encoder. The released generations use train-only length priors and constrained EOS decoding.
Predictor-derived Tm labels describe the reference G4 core, not the melting temperature of an entire long window. See [dataset](docs/dataset.md) and [architecture](docs/lstm.md) notes.

## Provenance and licensing

See [THIRD_PARTY.md](THIRD_PARTY.md). No new blanket software/model/data license is asserted in this private draft. Upstream dependencies retain their own terms.
The repository name is a paper identifier, not a claim that these newly released runs have already beaten the other models.

