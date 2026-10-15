"""Null-condition view of the joint Tm model; identical state-dict keys."""
from .conditional import TmDirichletDiffusionCNN


class NullDirichletDiffusionCNN(TmDirichletDiffusionCNN):
    """Select the learned null branch, not a separate unconditioned network."""
    def forward(self, x, t):
        return super().forward(x, t, target_tm_celsius=None)
