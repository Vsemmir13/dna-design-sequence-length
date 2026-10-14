# Non-B DNA Generation

PyTorch architectures and loading code for G-quadruplex-associated DNA generation.
This is the code repository for **Return of the LSTM: Non-B DNA Generation**.

Weights, generated sequences, processed datasets and training curves are on [Hugging Face](https://huggingface.co/vsemmir13/return-of-the-lstm-nonb-dna).

## Models

- **LSTM:** autoregressive DNA generation, with unconditional and Tm-conditioned architectures.
- **Discrete Flow Matching:** convolutional DNA generation with Tm conditioning and an unconditional mode.

Hugging Face currently contains unconditional LSTM weights for short, 512-nt and 1,024-nt sequences. Other weights are not yet released.

## Load a model

```bash
python -m pip install -r requirements.txt
hf auth login
python scripts/load_lstm.py --regime short
```

Use `--regime nt512` or `--regime nt1024` for long sequences.
The loader loads the architecture strictly and runs a CPU forward check.
It does not launch training or create a new generation cohort.

## Files

```text
models/lstm/{unconditional,conditional}/architecture.py
models/discrete_flow_matching/architecture.py
configs/lstm/unconditional/{short,nt512,nt1024}.json
scripts/load_lstm.py
docs/{dataset,lstm,discrete_flow_matching}.md
tests/test_architectures.py
```

See [LSTM](docs/lstm.md), [Discrete Flow Matching](docs/discrete_flow_matching.md) and [dataset](docs/dataset.md) notes for architecture and data details.
The Discrete Flow Matching adapter requires the pinned author's CNN checkout; it does not redistribute that source. Instructions and dependencies are listed in [THIRD_PARTY.md](THIRD_PARTY.md).

Training curves are in the Hugging Face **Metrics** tab. DNA datasets and weights are stored there, not in this GitHub repository.
Evaluation of generation quality for the newly released models is not finished. Predictor-derived Tm labels and generated structures are not experimental measurements.

