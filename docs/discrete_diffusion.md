# Discrete Diffusion — D3

Adaptation of [DNA Discrete Diffusion](https://github.com/anirbansarkar-cs/D3-DNA-Discrete-Diffusion),
commit `76d1976b03bb844531482f2104e589562a56880b`, from
[Sarkar et al.](https://www.biorxiv.org/content/10.1101/2024.05.23.595630v3).
Training is complete for all three lengths. Selected EMA weights and full
TensorBoard histories are on Hugging Face. Null and CFG generation is in progress;
only completed, verified cohorts are uploaded.

The author promoter CNN retains 20 kernel-9 blocks, GroupNorm, SiLU/GELU and
`[1,1,4,16,64]×4` dilations. Channels 256→128 and time embedding 256→210 give
**3,662,565 trainable parameters**, identical for short, 512-nt and 1,024-nt fits.
The original CNN has 13,403,653 parameters. The count retains 44,311 unused
trainable slots from the author's Fourier/linear branch and scale parameter.

Tm uses the same `Linear(1,8) → SiLU → Linear(8,8)` encoder as LSTM, then an
8→1 projection broadcast into the original signal channel. A learned null
scalar and 30% condition dropout support both modes in one joint network.
Tm labels describe the reference G4 core, not the complete genomic window.

The training protocol uses the same 31,669 train / 5,000 validation IDs as the
other Tm models, with no test-based fitting or selection. Short examples are
grouped by exact length, without a length encoder, sequence PAD or cropping.

## Load the architecture

Keep the author checkout outside this repository:

```bash
git clone https://github.com/anirbansarkar-cs/D3-DNA-Discrete-Diffusion.git /path/to/d3-author
git -C /path/to/d3-author checkout 76d1976b03bb844531482f2104e589562a56880b
```

The adapter checks source-file hashes and loads only the original CNN classes.
It calls the unmodified **promoter-specific forward**; the generic CNN's
signal-axis mismatch is not patched or used. No FlashAttention, Lightning,
OmegaConf, author installation scripts or network access are needed at import.

```python
import json
import torch
from pathlib import Path
from models.discrete_diffusion.architectures.conditional import make_d3_model

config = json.loads(Path("models/discrete_diffusion/configs/conditional.json").read_text())
settings = {**config["architecture"], "sequence_length": 49}  # or 512 / 1024
model = make_d3_model(source_dir="/path/to/d3-author", **settings).eval()
tokens = torch.zeros(2, 20, dtype=torch.long)  # A/C/G/T IDs: 0/1/2/3
condition = torch.tensor([[60.0, 1.0], [0.0, 0.0]])  # Tm/presence; second row is null
with torch.no_grad():
    log_scores = model(tokens, condition, False, sigma=torch.ones(2))  # [2,20,4]
    null_scores = model(tokens, None, False, sigma=torch.ones(2))
```

`make_unconditional_model` returns this same joint architecture; pass
`labels=None` to select the learned null branch. Checkpoint keys are identical.
The output is a log-score estimate, not normalized probabilities or a completed
generation. For fixed-length models, the input must have exactly 512 / 1024 nt.

## Training and sampling boundary

The method uses the author's uniform graph, geometric noise and
score-entropy objective, not cross-entropy or flow matching. Loss sums positions
before averaging sequences; raw losses are not comparable across lengths.
Training settings: AdamW 3e-4, warmup 2,500, clip 1, batch 128 and EMA 0.9999.
Maximum 300 / minimum 50 epochs and fixed-noise 0.7 conditional + 0.3 null EMA
validation selection are our stopping/selection adaptations. FP32 explicitly
replaces the author's forced CUDA-FP16 score scope.

Completed epochs are 132 / 156 / 206 for short / 512 / 1,024 nt. Selected EMA
epochs are 113 / 136 / 187, using the strict validation criterion above.

The original Euler-128 sampler can produce **negative categorical weights**;
the final analytic denoiser also failed a numerical check. Our frozen generation
configuration uses the author's unchanged Euler predictor with **512 updates,
FP32, epsilon 1e-5 and no final denoiser**. This is a sampling adaptation, not a
claim of reproducing the upstream default. It passed 20 checks covering lengths
13, 49, 512 and 1,024 at null and CFG scales 0.5, 1, 2 and 3. Sampling fails on
negative/nonfinite weights or zero mass, without clipping, renormalization or
output repair. CFG is applied in log-score space as
`null + scale * (conditional - null)` before the author's score conversion.

Each requested cohort contains 5,000 sequences; conditional cohorts allocate
625 to each target from 45 to 80 degrees Celsius in five-degree increments.
Numerical validity is not a biological-quality result. No sampler, trainer,
weights or technical smoke artifacts are bundled in this GitHub repository.

Source provenance and licensing are in [third-party notes](third_party.md).
