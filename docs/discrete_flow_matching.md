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
Sampling: original author wrapper, 100 solver steps / 200 CNN forwards, CFG scale 3. MASK outcomes are not suppressed, replaced or topped up.

The active campaign's final weights and generations are not included until completed and SHA-verified.

## Load the upstream architecture without vendoring it

```bash
git clone https://github.com/hashaam13/flow external/author-flow
git -C external/author-flow checkout cf186ba7f8551d2c305b79a863d569eed86665b7
```

Pass that checkout path explicitly to `make_discrete_flow_model(source_dir=..., conditional=True, tm_mean=..., tm_std=...)` from `models.discrete_flow_matching.architecture`. The loader verifies `dna_model.py` before import. See THIRD_PARTY.md for the licensing boundary.

