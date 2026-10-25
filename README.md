# A Compact LSTM Matches Diffusion and Flow Matching Across DNA Design Tasks

## Abstract

Diffusion and flow matching dominate generative design of regulatory and non-canonical DNA, evaluated mainly against one another. We compare a two-layer autoregressive LSTM of 3.7M parameters with Dirichlet and discrete diffusion and with Dirichlet, Fisher and discrete flow matching at matched parameter count, on enhancers, promoters, G-quadruplexes and Z-DNA. The LSTM matched the real data most closely on both enhancer datasets, on Z-DNA and on 1,024-nt GQ windows, altering one of 775 transcription factor motifs on melanoma enhancers against 80 for Dirichlet flow matching. Control did not follow fidelity. The LSTM gave the lowest FBD on three of five enhancer cell types per dataset and tracked the requested GQ melting temperature best on short cores, but doubled the best promoter error, and on long windows it combined the best distributional fidelity with no temperature control. Guided models bought control by sacrificing fidelity. A parameter-matched autoregressive baseline belongs in future benchmarks.

## Repository contents

PyTorch architectures, experiment configurations and documentation for G-quadruplex and Z-DNA generation. Checkpoints, generated sequences, processed datasets and training curves are on [Hugging Face](https://huggingface.co/vsemmir13/dna-design-sequence-length). Promoter and enhancer code: [hashaam13/flow](https://github.com/hashaam13/flow).
