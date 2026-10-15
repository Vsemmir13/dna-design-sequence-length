# Dirichlet Flow Matching

Adapted from [Stark et al., ICML 2024](https://proceedings.mlr.press/v235/stark24b.html),
using `HannesStark/dirichlet-flow-matching@6f360612da8f69ecf4860c075f74ed52b37ce64b`.

The joint Tm/null network has **3,661,532 trainable parameters**, identical for
short, 512-nt and 1,024-nt sequences. It retains the author's CNN forward:
128 channels, 20 kernel-9 blocks, LayerNorm, ReLU and zero dropout.
The actual source dilation order is eight layers at 1, then four each at 4, 16
and 64. The four-category simplex is expanded to eight input channels by the
original method; output is four A/C/G/T logits per position.

We replace categorical conditioning with train-standardized G4-core Tm:
`Linear(1,8) → SiLU → Linear(8,8) → Linear(8,128)` and a learned 128-D null
vector. The first two linear layers are the same Tm encoder as in LSTM.
Original per-block condition projections remain unchanged. Training drops
the Tm condition with probability 0.3.

Short sequences are grouped by exact length, without a neural length encoder,
sequence padding or cropping. Three separate fits use the same qualified
31,669 training and 5,000 validation IDs. No test rows enter fit or selection.
For long windows, Tm describes the embedded G4 core, not the entire window.

Training uses the original Dirichlet path and per-sequence cross-entropy,
Adam 5e-4, FP32 and global batch 256. Our stopping policy is maximum 800 /
minimum 50 epochs, patience 20 and minimum improvement 0.001. Best-checkpoint
selection uses fixed-noise validation loss, weighted 0.7 conditional + 0.3 null;
this is an adaptation, not the paper's FBD-based selection.

## Architecture interface

```python
import json
import torch
from pathlib import Path
from models.dirichlet_flow_matching.architectures.conditional import TmDirichletFlowCNN

config = json.loads(Path("models/dirichlet_flow_matching/configs/conditional.json").read_text())
model = TmDirichletFlowCNN(**config["architecture"]).eval()
seq = torch.full((2, 20, 8), 0.125)  # expanded simplex; normally built by the sampler
time = torch.tensor([2.0, 2.0])
condition = torch.tensor([[60.0, 1.0], [0.0, 0.0]])  # Tm/presence; second row is null
with torch.no_grad():
    logits = model(seq, time, cls=condition)  # [2,20,4]
```

`architectures/unconditional.py` exposes `NullDirichletFlowCNN`, a null-only view
of this same network. Its state-dict keys and parameter count are identical;
it does not represent a separately trained unconditional checkpoint.

These files define the network, not a full sampler. The prepared original
inference uses 100 grid points (99 Euler updates); guidance is not yet selected
on validation. This addition contains no weights, generation cohorts or quality
measurements. See [sources and licensing](third_party.md).
