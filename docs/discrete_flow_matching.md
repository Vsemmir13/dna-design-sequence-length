# Discrete Flow Matching

Pinned author DNA source: `hashaam13/flow@cf186ba7f8551d2c305b79a863d569eed86665b7`.
CNN: width 128, 20 dilated 1-D convolution layers, kernel 9, LayerNorm, ReLU and dropout 0.3. Actual source dilation order: eight layers at 1, then four each at 4,16,64; not the prose pattern repeated four times.

Clean unconditional source reference: 3,326,597 trainable parameters plus 64 fixed Fourier weights. No class/Tm/length encoder. This is not the jointly trained model below.

Joint Tm/CFG model: 3,658,205 trainable parameters plus 64 fixed Fourier weights, identical across short/512/1024.
Our adapter replaces categorical conditioning with train-standardized scalar Tm → Linear 1→8 → SiLU → Linear 8→8 → Linear 8→128, plus a learned 128-D null vector. Original per-block class projections and CNN forward remain unchanged.
Payload: `[raw Celsius, condition-present flag]`; null is not a physical target of 0 °C.

Short sequences are grouped by exact length; no sequence padding/cropping or length MLP. Requested tensor widths follow the train empirical prior at generation.
Training: upstream Meta generalized KL, MASK source, quadratic path, AdamW 5e-4, global batch 1024, 500-step warmup, gradient clip 1 and condition dropout 0.3.
Maximum 600 epochs; minimum 50, validation patience 20 and minimum significant improvement 0.001. Strict-minimum checkpoint selection is separate from plateau stopping; budget stopping is a separate status.
Sampling uses the original author wrapper and 100 solver steps. The released grid has a null-condition cohort and conditional CFG scales 0.5, 1, 2 and 3, with 5,000 outputs per setting. The wrapper evaluates conditional and null branches; no CFG scale is designated the validation-selected winner. Best epochs are 55 / 45 / 162 for short / 512 / 1,024 nt.

Original MASK outcomes are exported as N and retained in their original cohorts. The explicitly named `invalid_output_resampled_20261016_v1` variants replace only those invalid rows using the same weights, CFG, requested length and Tm. They are separate rejection-sampling cohorts, not untouched outputs. This does not introduce EOS decoding or cropping; the generated tensor width stays fixed.

Available checkpoints and generated samples are listed on [Hugging Face](https://huggingface.co/vsemmir13/return-of-the-lstm-nonb-dna).

## Load the upstream architecture without vendoring it

```bash
git clone https://github.com/hashaam13/flow /path/to/author-flow
git -C /path/to/author-flow checkout cf186ba7f8551d2c305b79a863d569eed86665b7
```

Use a checkout outside this repository. The architecture files import the pinned CNN without copying its source here.

```python
from models.discrete_flow_matching.architectures.conditional import make_discrete_flow_model

model = make_discrete_flow_model(
    source_dir="/path/to/author-flow",
    tm_mean=57.84716468707569,
    tm_std=8.054890567895972,
)
```

Call `model(tokens, time, condition)`: tokens are `[batch, length]`, time is `[batch]`, and condition is `[batch, 2]`. A row `[60, 1]` requests 60 °C; `[0, 0]` selects the learned null condition. The output is `[batch, length, 5]` nucleotide/MASK logits. This forward call is not a complete sampler.

`architectures/unconditional.py` builds the clean unconditioned source reference. It is not the joint checkpoint with its temperature switched off. The two JSON configs make this distinction explicit. See [sources](third_party.md) for the upstream repositories and terms.

