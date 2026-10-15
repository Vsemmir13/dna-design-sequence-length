# Sources and licensing

## Discrete Flow Matching

Author-designated DNA implementation: [hashaam13/flow](https://github.com/hashaam13/flow), commit `cf186ba7f8551d2c305b79a863d569eed86665b7`.
Its `dna_model.py` is adapted from [HannesStark/dirichlet-flow-matching](https://github.com/HannesStark/dirichlet-flow-matching).
No LICENSE file was present in the audited author checkout. This snapshot therefore does not redistribute that CNN implementation; it loads an explicitly supplied, hash-checked local checkout.

Official method dependency: [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching), commit `11568d37f8d5a080e12aa7b5305d9c35ae07d136`, upstream CC-BY-NC terms. No Meta source is vendored here.

Our declared modification is a continuous scalar Tm encoder and learned null condition, replacing the author's categorical class embedding. The CNN forward, normalization, activations, dropout and dilation order are not rewritten by the adapter.

## Dirichlet Flow Matching

[HannesStark/dirichlet-flow-matching](https://github.com/HannesStark/dirichlet-flow-matching), commit `6f360612da8f69ecf4860c075f74ed52b37ce64b`, accompanying [Stark et al., ICML 2024](https://proceedings.mlr.press/v235/stark24b.html).
The CNN, Fourier projection and dense layers are included under the upstream MIT license; its full notice is retained in `models/dirichlet_flow_matching/architectures/conditional.py`.
We replace the categorical embedding with a continuous Tm encoder and learned null vector. Exact-length batching and validation-loss checkpoint selection are task adaptations. No upstream trainer or sampler is bundled in this architecture-only repository.

## Dirichlet Diffusion

[jzhoulab/ddsm](https://github.com/jzhoulab/ddsm), commit `ff20be49138589ae0f477b4039feafa7343de2a2`, accompanying [Avdeyev et al., ICML 2023](https://proceedings.mlr.press/v202/avdeyev23a.html).
The actual upstream LICENSE permits **academic research use only** and prohibits commercial/for-profit use. It must not be relabeled as unrestricted BSD or MIT. Its complete notice is retained in `models/dirichlet_diffusion/architectures/conditional.py`.
Our capacity-matched adaptation reduces CNN channels 256→128 and time embedding 256→244, and replaces the CAGE input channel with a broadcast Tm/null feature. GroupNorm, activations and the CNN forward remain from the author implementation. This is not an unchanged original-capacity reproduction.
The license restriction applies to this DDSM-derived code; no repository-wide license overrides it.

## Data and predictors

G4-associated loci originate from EndoQuad and hg38. Tm labels are computational G4STAB core predictions. The private Hugging Face repository includes the frozen qualified train/validation tables for short, 512-nt and 1,024-nt sequences. Reference genomes, original source interval datasets, predictor weights, full unconditional cohorts and held-out test rows are not uploaded.
Generated samples are frozen model outputs, not experimentally validated structures. Underlying resources retain their own terms.

## Release status

Software, weight and sample licensing for a public paper release remains to be approved by the authors. Private storage does not grant additional redistribution rights.

