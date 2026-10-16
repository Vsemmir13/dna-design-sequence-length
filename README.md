# Return of the LSTM: Non-B DNA Generation

PyTorch architectures and experiment settings for generating G-quadruplex-associated DNA: short cores and 512- or 1,024-nucleotide genomic windows containing those cores.

- **LSTM** generates DNA one nucleotide at a time.
- **Discrete Flow Matching** uses a convolutional network to update a whole sequence over sampling steps.
- **Dirichlet Flow Matching** moves continuous nucleotide probabilities along a learned flow.
- **Dirichlet Diffusion** reverses Jacobi diffusion on nucleotide probabilities.
- **Discrete Diffusion (D3)** models nucleotide substitutions; GPU training has started, but completed weights and samples are not released. Its sampler still requires a numerical check described in the model notes.

The current comparison uses the same 31,669 training and 5,000 validation examples for all models. LSTM has separate unconditional and Tm-conditioned fits; the convolutional models use joint Tm/null fits. Tm is a G4STAB prediction for the G4 core, not an experimental measurement or the temperature of the whole genomic window.

LSTM generates until its own EOS token, without forcing a requested length or cropping. Thus a model trained on 512-nt windows need not output exactly 512 nt. Convolutional models instead generate at the requested tensor width: short lengths follow the training distribution, while long windows are fixed at 512 or 1,024 nt. They do not use an EOS token or a learned length embedding.

[Hugging Face: weights, generated sequences, datasets and training curves](https://huggingface.co/vsemmir13/return-of-the-lstm-nonb-dna)

## Repository layout

```text
models/
  <model>/
    configs/{unconditional,conditional}.json
    architectures/{unconditional,conditional}.py
docs/
README.md
requirements.txt
```

`<model>` is `lstm`, `discrete_flow_matching`, `dirichlet_flow_matching`, `dirichlet_diffusion` or `discrete_diffusion`. Architecture files define the networks; configs record dimensions, training and sampling settings. The unconditional Discrete Flow Matching file is an untrained source reference, not the released joint checkpoint. Its released null mode is loaded through `conditional.py` with a null payload. The Dirichlet and D3 unconditional files expose the null branch of the joint network.

## Available artifacts

Release snapshot: 16 October 2026. LSTM, both flow-matching models and Dirichlet Diffusion have validation-selected weights and 5,000-sequence generation cohorts on Hugging Face. D3 has no completed release yet. Only final selected weights and completed training curves are published; generation settings are recorded beside each cohort. Temperature and CFG sweeps are comparisons, not a preselected best setting.

The uploaded datasets contain G4 train/validation tables for all three lengths and a separate mouse Z-DNA-associated 256-nt dataset. New Z-DNA model results are not part of this release. Evaluation is in progress; these artifacts do not establish a winning model or experimentally confirmed DNA structures.

## Getting started

```bash
python -m pip install -r requirements.txt
```

See [LSTM](docs/lstm.md), [Discrete Flow Matching](docs/discrete_flow_matching.md), [Dirichlet Flow Matching](docs/dirichlet_flow_matching.md), [Dirichlet Diffusion](docs/dirichlet_diffusion.md), [Discrete Diffusion](docs/discrete_diffusion.md) and [datasets](docs/dataset.md). Discrete Flow Matching and D3 require external author checkouts; the Dirichlet CNN definitions are included with their original license notices. See [sources](docs/third_party.md).

This repository contains only architectures, configs and documentation, not training or generation scripts. Available weights, data and training curves live on Hugging Face; adding an architecture here does not mean its weights or evaluation results are released. The manuscript will be added here when it is ready.

