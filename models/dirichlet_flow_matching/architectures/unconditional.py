"""Null-condition view of the joint Tm model; identical state-dict keys."""
import torch
from .conditional import TmDirichletFlowCNN


class NullDirichletFlowCNN(TmDirichletFlowCNN):
    """Select the learned null branch, not a separate unconditioned network."""
    def forward(self, seq, t):
        null = torch.ones(len(seq), device=seq.device, dtype=torch.long)
        return super().forward(seq, t, cls=null)
