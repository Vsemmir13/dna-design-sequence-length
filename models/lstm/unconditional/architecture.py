"""Architecture only. State-dict compatible with the new unconditional runs."""
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


class G4LSTM(nn.Module):
    def __init__(self, vocab_size=7, emb_dim=128, hidden_dim=372,
                 num_layers=2, mlp_layers=1, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(emb_dim), nn.Linear(emb_dim, hidden_dim),
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

    def forward(self, tokens):
        out, _ = self.lstm(self.input_proj(self.emb(tokens)))
        return self.fc_out(self.mlp_blocks(self.out_norm(out)))
