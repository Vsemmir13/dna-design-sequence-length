"""Unmodified author neural class bodies.
Source: HannesStark/dirichlet-flow-matching@6f360612da8f69ecf4860c075f74ed52b37ce64b

MIT License

Copyright (c) 2024 Hannes Stärk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import copy
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    """

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Dense(nn.Module):
    """
    A fully connected layer that reshapes outputs to feature maps.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[...]

class CNNModel(nn.Module):
    def __init__(self, args, alphabet_size, num_cls, classifier=False):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.args = args
        self.classifier = classifier
        self.num_cls = num_cls

        if self.args.clean_data:
            self.linear = nn.Embedding(self.alphabet_size, embedding_dim=args.hidden_dim)
        else:
            expanded_simplex_input = args.cls_expanded_simplex or not classifier and (args.mode == 'dirichlet' or args.mode == 'riemannian')
            inp_size = self.alphabet_size * (2 if expanded_simplex_input else 1)
            if (args.mode == 'ardm' or args.mode == 'lrar') and not classifier:
                inp_size += 1 # plus one for the mask token of these models
            self.linear = nn.Conv1d(inp_size, args.hidden_dim, kernel_size=9, padding=4)
            self.time_embedder = nn.Sequential(GaussianFourierProjection(embed_dim= args.hidden_dim),nn.Linear(args.hidden_dim, args.hidden_dim))

        self.num_layers = 5 * args.num_cnn_stacks
        self.convs = [nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=9, padding=4),
                                     nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=9, padding=4),
                                     nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=9, dilation=64, padding=256)]
        self.convs = nn.ModuleList([copy.deepcopy(layer) for layer in self.convs for i in range(args.num_cnn_stacks)])
        self.time_layers = nn.ModuleList([Dense(args.hidden_dim, args.hidden_dim) for _ in range(self.num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(args.hidden_dim) for _ in range(self.num_layers)])
        self.final_conv = nn.Sequential(nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=1),
                                   nn.ReLU(),
                                   nn.Conv1d(args.hidden_dim, args.hidden_dim if classifier else self.alphabet_size, kernel_size=1))
        self.dropout = nn.Dropout(args.dropout)
        if classifier:
            self.cls_head = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(args.hidden_dim, self.num_cls))

        if self.args.cls_free_guidance and not self.classifier:
            self.cls_embedder = nn.Embedding(num_embeddings=self.num_cls + 1, embedding_dim=args.hidden_dim)
            self.cls_layers = nn.ModuleList([Dense(args.hidden_dim, args.hidden_dim) for _ in range(self.num_layers)])
    def forward(self, seq, t, cls = None, return_embedding=False):
        if self.args.clean_data:
            feat = self.linear(seq)
            feat = feat.permute(0, 2, 1)
        else:
            time_emb = F.relu(self.time_embedder(t))
            feat = seq.permute(0, 2, 1)
            feat = F.relu(self.linear(feat))

        if self.args.cls_free_guidance and not self.classifier:
            cls_emb = self.cls_embedder(cls)

        for i in range(self.num_layers):
            h = self.dropout(feat.clone())
            if not self.args.clean_data:
                h = h + self.time_layers[i](time_emb)[:, :, None]
            if self.args.cls_free_guidance and not self.classifier:
                h = h + self.cls_layers[i](cls_emb)[:, :, None]
            h = self.norms[i]((h).permute(0, 2, 1))
            h = F.relu(self.convs[i](h.permute(0, 2, 1)))
            if h.shape == feat.shape:
                feat = h + feat
            else:
                feat = h
        feat = self.final_conv(feat)
        feat = feat.permute(0, 2, 1)
        if self.classifier:
            feat = feat.mean(dim=1)
            if return_embedding:
                embedding = self.cls_head[:1](feat)
                return self.cls_head[1:](embedding), embedding
            else:
                return self.cls_head(feat)
        return feat


class TemperatureEncoder(nn.Module):
    """Lightweight continuous Tm embedding; receives a train-standardized scalar."""

    def __init__(self, out_dim=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, standardized_tm):
        return self.mlp(standardized_tm)


class TmConditionEmbedding(nn.Module):
    """Raw Celsius/presence payload; integer class1 is the author's null sentinel."""
    def __init__(self, hidden_dim=128, cond_dim=8, tm_mean=0., tm_std=1.):
        super().__init__()
        if tm_std <= 0 or cond_dim != 8:
            raise ValueError("Use positive training standard deviation and the shared 8D encoder")
        self.register_buffer("tm_mean", torch.tensor(float(tm_mean)))
        self.register_buffer("tm_std", torch.tensor(float(tm_std)))
        self.condition_encoder = TemperatureEncoder(cond_dim)
        self.projection = nn.Linear(cond_dim, hidden_dim)
        self.null_embedding = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, payload):
        if payload.ndim == 1:
            if payload.dtype != torch.long or not bool((payload == 1).all()):
                raise ValueError("Only the author's null-class sentinel1 is supported")
            return self.null_embedding.unsqueeze(0).expand(len(payload), -1)
        if payload.ndim != 2 or payload.shape[1] != 2:
            raise ValueError("Expected [batch,2]: raw Celsius and presence flag")
        payload = payload.to(device=self.tm_mean.device, dtype=self.tm_mean.dtype)
        tm, present = payload[:, :1], payload[:, 1:2]
        if not bool(((present == 0) | (present == 1)).all()):
            raise ValueError("Condition presence must be zero or one")
        if not bool(torch.isfinite(tm[present == 1]).all()):
            raise ValueError("Nonfinite temperature target")
        standardized = torch.where(present == 1, (tm-self.tm_mean)/self.tm_std, torch.zeros_like(tm))
        condition = self.projection(self.condition_encoder(standardized))
        return torch.where(present == 1, condition, self.null_embedding.unsqueeze(0))


class TmDirichletFlowCNN(CNNModel):
    """Original CNN forward, with an explicitly adapted continuous Tm embedding."""
    def __init__(self, hidden_dim=128, num_cnn_stacks=4, dropout=0.0,
                 tm_mean=0., tm_std=1.):
        args = SimpleNamespace(hidden_dim=hidden_dim, num_cnn_stacks=num_cnn_stacks,
            dropout=dropout, clean_data=False, cls_expanded_simplex=False,
            mode="dirichlet", cls_free_guidance=True)
        super().__init__(args, alphabet_size=4, num_cls=1)
        self.cls_embedder = TmConditionEmbedding(hidden_dim=hidden_dim,
                                                tm_mean=tm_mean, tm_std=tm_std)

