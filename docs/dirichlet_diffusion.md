# Dirichlet Diffusion

Adapted from [Avdeyev et al., ICML 2023](https://proceedings.mlr.press/v202/avdeyev23a.html),
using `jzhoulab/ddsm@ff20be49138589ae0f477b4039feafa7343de2a2`.
All three G4 fits are complete; selected weights, training curves and generated
sequences are available on [Hugging Face](https://huggingface.co/vsemmir13/return-of-the-lstm-nonb-dna).

The original promoter CNN has 256 channels and 13,272,069 trainable parameters.
For comparison with our other models, this version uses 128 channels and a
244-D time embedding: **3,666,795 trainable parameters** including Tm/null
conditioning. Capacity is identical for short, 512-nt and 1,024-nt sequences.

The author CNN forward, 20 kernel-9 blocks, GroupNorm(1, channels), swish and
GELU output-head activation are retained. Dilations are `[1,1,4,16,64]` repeated
four times. These are the DDSM author's choices, not the normalization and
activation used by our flow-matching models.

Tm uses the same `Linear(1,8) → SiLU → Linear(8,8)` encoder as LSTM, followed
by an additional `Linear(8,1)` projection. This scalar feature replaces the
original CAGE input channel and is broadcast along the sequence. A separate
learned null scalar supports 30% condition dropout. Thus the encoder is shared
in design, but its connection to the CNN differs from LSTM's concatenation.

The protocol uses the same 31,669 training and 5,000 validation G4 examples as
the other Tm models, with no test-based fitting or selection. Short sequences
are grouped by exact length; there is no sequence PAD, crop or length encoder.
Tm labels describe the reference G4 core, including for long genomic windows.

## Architecture interface

```python
import json
import torch
from pathlib import Path
from models.dirichlet_diffusion.architectures.conditional import TmDirichletDiffusionCNN

config = json.loads(Path("models/dirichlet_diffusion/configs/conditional.json").read_text())
model = TmDirichletDiffusionCNN(**config["architecture"]).eval()
x = torch.full((2, 20, 4), 0.25)  # noisy A/C/G/T simplex
time = torch.tensor([1.0, 1.0])
with torch.no_grad():
    conditional_score = model(x, time, torch.tensor([60.0, 70.0]))
    null_score = model(x, time)  # both outputs [2,20,4]
```

`architectures/unconditional.py` exposes `NullDirichletDiffusionCNN`, with the
same state-dict keys as the joint model. It only selects the null branch.
The constructor initializes time-dependent score weights to ones for shape
compatibility; real fitting requires train-calibrated weights, and loading a
trained checkpoint restores them. Random initialization is not a trained model.

## Training and generation

The objective is the author's stick-breaking denoising score loss,
not cross-entropy. It requires the original float64 Jacobi noise table
(100,000 samples, 400 time points); approximate noise is not a fallback.
Noise is generated in independent-sample chunks, and time-weight calibration
uses `scatter_add_` to accumulate repeated time bins correctly. These are
declared implementation adaptations, not a bitwise reproduction claim.

Training: Adam 5e-4, FP32, global batch 256, maximum 200 / minimum 50
epochs, patience 20. Selection uses fixed-noise validation score loss, weighted
0.7 conditional + 0.3 null, rather than the paper's promoter SEI MSE.
The short / 512 / 1,024 fits completed 77 / 119 / 102 epochs, selecting epochs
69 / 113 / 100. No auxiliary cross-entropy, LR scheduler or gradient clipping
was used.

Generation uses the original Euler–Maruyama sampler with 100 updates from time
4 to 0.01. A categorical draw from the final simplex exports A/C/G/T, using a
separate decoder RNG; it is not an argmax decoder. Each length has 5,000 outputs
for the null condition and for each CFG scale 0.5, 1, 2 and 3. Conditional cohorts
allocate 625 outputs to each requested core Tm from 45 to 80 °C. Short tensor
widths follow the training length distribution; 512/1,024 widths are fixed.
No EOS token, sequence padding, post-hoc cropping or output filtering is used.
Guidance-scale selection and generation-quality evaluation are separate from
the completed training runs.

The original code is restricted to academic research. Its full license notice
is retained in the architecture file; see [sources and licensing](third_party.md).
