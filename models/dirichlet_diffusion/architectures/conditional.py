"""DDSM promoter ScoreNet: original forward, capacity-matched width, Tm input.

Copyright (c) 2023 The University of Texas Southwestern Medical Center.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted for academic research use only (subject to the limitations in the disclaimer below) provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
* Neither the name of the copyright holders nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

ANY USE OR REDISTRIBUTION OF THIS SOFTWARE FOR COMMERCIAL PURPOSES, WHETHER IN SOURCE OR BINARY FORM, WITH OR WITHOUT MODIFICATION, IS EXPRESSLY PROHIBITED; ANY USE OR REDISTRIBUTION BY A FOR-PROFIT ENTITY SHALL COMPRISE USE OR REDISTRIBUTION FOR COMMERCIAL PURPOSES.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE, AND ANY ACCOMPANYING DOCUMENTATION, IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE OR ANY OF ITS ACCOMPANYING DOCUMENTATION, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np
import torch
from torch import nn


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


class ScoreNet(nn.Module):
    """Author CNN, not a U-Net despite the original class docstring."""
    def __init__(self, embed_dim=244, time_dependent_weights=None, time_step=0.01, channels=128):
        super().__init__()
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))
        n = channels
        self.linear = nn.Conv1d(5, n, kernel_size=9, padding=4)
        self.blocks = nn.ModuleList([
            nn.Conv1d(n, n, kernel_size=9, dilation=d, padding=4*d)
            for d in [1, 1, 4, 16, 64]*4
        ])
        self.denses = nn.ModuleList([Dense(embed_dim, n) for _ in range(20)])
        self.norms = nn.ModuleList([nn.GroupNorm(1, n) for _ in range(20)])
        self.act = lambda x: x * torch.sigmoid(x)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.scale = nn.Parameter(torch.ones(1))
        self.final = nn.Sequential(nn.Conv1d(n, n, kernel_size=1),
                                   nn.GELU(),
                                   nn.Conv1d(n, 4, kernel_size=1))
        self.register_buffer("time_dependent_weights", time_dependent_weights)
        self.time_step = time_step

    def forward(self, x, t, t_ind=None, return_a=False):
        # Obtain the Gaussian random feature embedding for t
        # embed: [N, embed_dim]
        embed = self.act(self.embed(t / 2))

        # Encoding path
        # x: NLC -> NCL
        out = x.permute(0, 2, 1)
        out = self.act(self.linear(out))

        # pos encoding
        for block, dense, norm in zip(self.blocks, self.denses, self.norms):
            h = self.act(block(norm(out + dense(embed)[:, :, None])))
            if h.shape == out.shape:
                out = h + out
            else:
                out = h

        out = self.final(out)

        out = out.permute(0, 2, 1)

        if self.time_dependent_weights is not None:
            t_step = (t / self.time_step) - 1
            w0 = self.time_dependent_weights[t_step.long()]
            w1 = self.time_dependent_weights[torch.clip(t_step + 1, max=len(self.time_dependent_weights) - 1).long()]
            out = out * (w0 + (t_step - t_step.floor()) * (w1 - w0))[:, None, None]

        out = out - out.mean(axis=-1, keepdims=True)
        return out


class TemperatureEncoder(nn.Module):
    """Lightweight continuous Tm embedding; receives a train-standardized scalar."""

    def __init__(self, out_dim=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, standardized_tm):
        return self.mlp(standardized_tm)


class TmDirichletDiffusionCNN(nn.Module):
    """Replace the author's CAGE channel by a broadcast Tm/null scalar feature.

    No length embedding, sequence PAD or sequence cropping. Conv boundary padding
    remains part of the original CNN and is not sequence padding.
    """
    def __init__(self, tm_mean=0., tm_std=1., channels=128, embed_dim=244):
        super().__init__()
        if tm_std <= 0 or channels != 128 or embed_dim != 244:
            raise ValueError('Frozen capacity: channels128, time244, positive Tm std')
        self.score_net = ScoreNet(embed_dim, torch.ones(400), channels=channels)
        self.register_buffer('tm_mean', torch.tensor(float(tm_mean)))
        self.register_buffer('tm_std', torch.tensor(float(tm_std)))
        self.condition_encoder = TemperatureEncoder(8)
        self.condition_projection = nn.Linear(8, 1)
        self.null_condition = nn.Parameter(torch.zeros(1))

    def forward(self, x, t, target_tm_celsius=None, present=None):
        count = len(x)
        if x.ndim != 3 or x.shape[-1] != 4:
            raise ValueError('Expected exact-length simplex [batch,length,4]')
        if target_tm_celsius is None:
            if present is not None and bool(present.any()):
                raise ValueError('Present condition requires Tm')
            feature = self.null_condition.reshape(1, 1).expand(count, 1)
        else:
            tm = torch.as_tensor(target_tm_celsius, dtype=x.dtype, device=x.device).reshape(count, 1)
            flag = torch.ones_like(tm, dtype=torch.bool) if present is None else present.reshape(count, 1)
            flag = flag.to(device=x.device, dtype=torch.bool)
            if not bool(torch.isfinite(tm[flag]).all()):
                raise ValueError('Nonfinite present Tm')
            z = torch.where(flag, (tm-self.tm_mean)/self.tm_std, torch.zeros_like(tm))
            encoded = self.condition_projection(self.condition_encoder(z))
            feature = torch.where(flag, encoded, self.null_condition.reshape(1, 1))
        return self.score_net(torch.cat([x, feature[:, None, :].expand(count, x.shape[1], 1)], -1), t)
