"""Tm-conditioned architecture only; verified weights not in the initial release."""
import torch
from torch import nn


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class TemperatureEncoder(nn.Module):
    """Lightweight continuous Tm embedding; receives a train-standardized scalar."""

    def __init__(self, out_dim=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, standardized_tm):
        return self.mlp(standardized_tm)


class G4TmLSTM(nn.Module):
    def __init__(self, vocab_size=7, emb_dim=128, hidden_dim=372, num_layers=2,
                 mlp_layers=1, dropout=0.2, cond_dim=8, tm_mean=0.0, tm_std=1.0):
        super().__init__()
        if tm_std <= 0 or cond_dim <= 0:
            raise ValueError("Tm standard deviation and condition dimension must be positive")
        self.register_buffer("tm_mean", torch.tensor(float(tm_mean)))
        self.register_buffer("tm_std", torch.tensor(float(tm_std)))
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.condition_encoder = TemperatureEncoder(cond_dim)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(emb_dim + cond_dim), nn.Linear(emb_dim + cond_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.mlp_blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim, dropout) for _ in range(mlp_layers)]
        )
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, vocab_size),
        )

    def forward(self, tokens, target_tm_celsius):
        batch, steps = tokens.shape
        tm = target_tm_celsius.to(device=tokens.device, dtype=self.emb.weight.dtype)
        tm = tm.reshape(batch, 1)
        cond = self.condition_encoder((tm - self.tm_mean) / self.tm_std)
        features = torch.cat([self.emb(tokens), cond.unsqueeze(1).expand(batch, steps, -1)], dim=-1)
        out, _ = self.lstm(self.input_proj(features))
        return self.fc_out(self.mlp_blocks(self.out_norm(out)))
