"""Author CNN with continuous Tm conditioning and a learned null condition."""

import torch
from torch import nn

from .unconditional import load_author_cnn


class TemperatureEncoder(nn.Module):
    def __init__(self, out_dim=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, standardized_tm):
        return self.mlp(standardized_tm)


class TmConditionEmbedding(nn.Module):
    """Columns are raw Celsius and a 0/1 condition-present flag."""
    def __init__(self, hidden_dim=128, cond_dim=8, tm_mean=0., tm_std=1.):
        super().__init__()
        if tm_std <= 0 or cond_dim <= 0:
            raise ValueError("Positive train Tm standard deviation and condition width required")
        self.register_buffer("tm_mean", torch.tensor(float(tm_mean)))
        self.register_buffer("tm_std", torch.tensor(float(tm_std)))
        self.condition_encoder = TemperatureEncoder(out_dim=cond_dim)
        self.projection = nn.Linear(cond_dim, hidden_dim)
        self.null_embedding = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, payload):
        if payload is None or payload.ndim != 2 or payload.shape[1] != 2:
            raise ValueError("Expected [batch,2]: raw Tm and presence flag")
        payload = payload.to(device=self.tm_mean.device, dtype=self.tm_mean.dtype)
        tm, flag = payload[:, :1], payload[:, 1:2]
        if not bool(((flag == 0) | (flag == 1)).all()):
            raise ValueError("Condition-presence flag must be 0 or 1")
        present = flag == 1
        if not bool(torch.isfinite(tm[present]).all()):
            raise ValueError("Present Tm targets must be finite")
        standardized = torch.where(present, (tm-self.tm_mean)/self.tm_std, torch.zeros_like(tm))
        embedding = self.projection(self.condition_encoder(standardized))
        return torch.where(present, embedding, self.null_embedding.unsqueeze(0))


def make_discrete_flow_model(*, source_dir, tm_mean=0., tm_std=1.):
    """Load the pinned author architecture and replace its categorical condition."""
    CNNModel = load_author_cnn(source_dir)

    class TmDiscreteFlowCNN(CNNModel):
        def __init__(self):
            super().__init__(vocab_size=5, hidden_dim=128, num_cnn_stacks=4,
                             p_dropout=.3, num_classes=1, cls_free_guidance=True)
            self.cls_embedder = TmConditionEmbedding(hidden_dim=128, cond_dim=8,
                                                     tm_mean=tm_mean, tm_std=tm_std)

    return TmDiscreteFlowCNN()

