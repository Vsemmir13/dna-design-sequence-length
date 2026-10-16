# LSTM

## Unconditional

3,662,383 trainable parameters for every length. Vocabulary: A/C/G/T/BOS/EOS/PAD, IDs 0/1/2/3/4/5/6.
Token embedding 128; input LayerNorm, projection to 372, GELU and dropout 0.2; two LSTM layers with hidden width 372; output LayerNorm; one residual MLP block; a 372→744→7 output head.
No class, Tm, null-condition or sequence-length encoder. BOS starts left-to-right decoding.

Current unconditional fits use the shared 31,669 training / 5,000 validation subset, not the older 74,872-row training set. Fresh seed 42, 50 epochs, Adam 5e-4, effective batch 512 and mixed FP16 training; checkpoints minimize full-validation cross-entropy, never test metrics. The incomplete effective training batch is dropped; all 5,000 validation examples are evaluated each epoch.

## Tm-conditioned

3,665,463 trainable parameters. Same backbone and output head.
Train-standardized Tm passes through Linear 1→8, SiLU and Linear 8→8. The 8-D vector is concatenated with token embeddings at every position; the input projection receives 136 dimensions.
There is no length embedding or learned null condition in this LSTM. Tm is standardized using the training mean and SD stored in the checkpoint.

The conditional fits use the same subset and 50-epoch protocol. Selected epochs (one-based, short / 512 / 1,024) are 18 / 45 / 47 for unconditional LSTM and 19 / 46 / 50 for Tm-conditioned LSTM.

## Generation

Sampling is categorical from `softmax(logits / temperature)`, not greedy. A/C/G/T and EOS are allowed; BOS and PAD cannot be generated. The model stops at its own EOS: no length is sampled or imposed, EOS is not forced, and outputs are not cropped. Short/512/1,024 describe the training regimes, not guaranteed generated lengths.

The safety limits are 200 / 2,052 / 4,100 decoding steps. A prefix that reaches a limit without EOS is flagged unfinished, not counted as a completed sequence. Temperature 1 is the reference; the released sweeps use 0.8, 0.9, 1, 1.1, 1.2 and 1.3 for short unconditional/conditional and 512-nt conditional models. Temperature can exceed 1; it flattens token probabilities. LSTM does not use CFG.

Each cohort has 5,000 outputs. Conditional cohorts request 625 at each G4-core Tm from 45 to 80 °C in 5 °C steps. Original problematic cohorts and separately named replacement variants must not be combined. `invalid_output_resampled_20261016_v1` replaces only unfinished rows; `context_resampled_20261016_v1` separately replaces outputs exceeding the HyenaDNA context. These variants change the sampling protocol and are explicitly labelled, not silently substituted. Old externally constrained-EOS cohorts were removed from HF.

## Files and inputs

`models/lstm/architectures/unconditional.py` defines `G4LSTM`; `conditional.py` defines `G4TmLSTM`. Their settings are in `models/lstm/configs` and apply to all three sequence lengths.

```python
import json
from pathlib import Path
from models.lstm.architectures.unconditional import G4LSTM

config = json.loads(Path("models/lstm/configs/unconditional.json").read_text())
model = G4LSTM(**config["architecture"])
```

Call `G4LSTM(tokens)` or `G4TmLSTM(tokens, target_tm_celsius)`. Tokens have shape `[batch, length]`; temperature targets have shape `[batch]`. Both return nucleotide logits `[batch, length, 7]`.

These files define the network only. Training batch padding/masks and decoding are outside the model forward method. Available weights, samples and per-cohort settings are on [Hugging Face](https://huggingface.co/vsemmir13/return-of-the-lstm-nonb-dna).

