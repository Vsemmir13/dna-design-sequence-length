# Return of the LSTM: Non-B DNA Generation

Models for generating G-quadruplex-associated DNA: short sequences of 13–49 nucleotides and genomic windows of 512 or 1,024 nucleotides around G4 loci.

- **LSTM** generates DNA one nucleotide at a time.
- **Discrete Flow Matching** uses a convolutional network to update a whole sequence over sampling steps.

The models support unconditional generation and conditioning on the melting temperature (Tm) of a G4 core. Tm labels are predictions, not experimental measurements; for long windows they describe the embedded core, not the entire sequence.

[Hugging Face: weights, generated sequences, datasets and training curves](https://huggingface.co/vsemmir13/return-of-the-lstm-nonb-dna)

## Repository layout

```text
models/
  lstm/
    configs/
      unconditional.json
      conditional.json
    architectures/
      unconditional.py
      conditional.py
  discrete_flow_matching/
    configs/
      unconditional.json
      conditional.json
    architectures/
      unconditional.py
      conditional.py
docs/
README.md
requirements.txt
```

`architectures` contains the PyTorch model definitions; `configs` records their dimensions and experiment settings. The unconditioned Discrete Flow Matching file is an architecture reference. The joint Tm-trained model uses `conditional.py` for both temperature-conditioned and null-condition generation.

## Getting started

```bash
python -m pip install -r requirements.txt
```

See [LSTM](docs/lstm.md), [Discrete Flow Matching](docs/discrete_flow_matching.md) and [datasets](docs/dataset.md). The DFM architecture requires the pinned author checkout described in [sources](docs/third_party.md).

This repository contains only architectures, configs and documentation. Weights and data live on Hugging Face; training curves are available in its Metrics tab. The manuscript will be added here when it is ready.

