# Return of the LSTM: Non-B DNA Generation

Models for generating G-quadruplex-associated DNA: short sequences of 13–49 nucleotides and genomic windows of 512 or 1,024 nucleotides around G4 loci.

- **LSTM** generates DNA one nucleotide at a time.
- **Discrete Flow Matching** uses a convolutional network to update a whole sequence over sampling steps.
- **Dirichlet Flow Matching** moves continuous nucleotide probabilities along a learned flow.
- **Dirichlet Diffusion** reverses a Jacobi diffusion process on nucleotide probabilities; its adaptation is prepared but not yet trained.

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
  dirichlet_flow_matching/
    configs/{unconditional,conditional}.json
    architectures/{unconditional,conditional}.py
  dirichlet_diffusion/
    configs/{unconditional,conditional}.json
    architectures/{unconditional,conditional}.py
docs/
README.md
requirements.txt
```

`architectures` contains the PyTorch model definitions; `configs` records their dimensions and experiment settings. The unconditioned Discrete Flow Matching file is an untrained architecture reference. For the two Dirichlet models, `unconditional.py` selects the learned null branch of the same joint Tm network, with identical checkpoint keys; it is not a separate unconditional fit.

## Getting started

```bash
python -m pip install -r requirements.txt
```

See [LSTM](docs/lstm.md), [Discrete Flow Matching](docs/discrete_flow_matching.md), [Dirichlet Flow Matching](docs/dirichlet_flow_matching.md), [Dirichlet Diffusion](docs/dirichlet_diffusion.md) and [datasets](docs/dataset.md). Only the Discrete Flow Matching architecture requires an external author checkout; the Dirichlet CNN definitions are included with their original license notices. See [sources](docs/third_party.md).

This repository contains only architectures, configs and documentation, not training or generation scripts. Available weights, data and training curves live on Hugging Face; adding an architecture here does not mean its weights or evaluation results are released. The manuscript will be added here when it is ready.

