"""Transformer building blocks with axial (space/time) self-attention.

The axial attention is implemented in :class:`Transformer` via the ``swap``
flag: even layers attend across time for each joint, odd layers attend across
joints for each frame. This keeps self-attention cheap while still modeling
both spatial and temporal dependencies.
"""
import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class RemixerBlock(nn.Module):
    """Optional token-mixing block (kept for experimentation, unused by default)."""

    def __init__(self, dim, seq_len, causal=False, bias=False):
        super().__init__()
        self.causal = causal
        self.proj_in = nn.Linear(dim, 2 * dim, bias=bias)
        self.mixer = nn.Parameter(torch.randn(seq_len, seq_len))
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.proj_out = nn.Linear(dim, dim, bias=bias)

    def forward(self, x):
        mixer, causal, device = self.mixer, self.causal, x.device
        x, gate = self.proj_in(x).chunk(2, dim=-1)
        x = F.gelu(gate) * x

        if causal:
            seq = x.shape[1]
            mask_value = -torch.finfo(x.dtype).max
            mask = torch.ones((seq, seq), device=device, dtype=torch.bool).triu(1)
            mixer = mixer[:seq, :seq].masked_fill(mask, mask_value)

        mixer = mixer.softmax(dim=-1)
        mixed = einsum("b n d, m n -> b m d", x, mixer)
        alpha = self.alpha.sigmoid()
        out = (x * mixed) * alpha + (x - mixed) * (1 - alpha)
        return self.proj_out(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.pos_embedding = PositionalEncoding(dim, 0.1, 128)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        x = x + self.pos_embedding(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv
        )
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    """Stacked attention + feed-forward layers.

    When ``swap=True`` the module runs *axial* attention: it alternates between
    attending over the temporal axis (all frames per joint) and the spatial
    axis (all joints per frame), rearranging the tensor back after each layer.
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x, swap=False, collect_attn=None):
        if swap:
            b, t, n, c = x.size()

        for idx, (attn, ff) in enumerate(self.layers):
            if swap:
                if idx % 2 == 0:
                    # attention across all frames for each joint
                    x = rearrange(x, "b t n c -> (b n) t c")
                else:
                    # attention across all joints in each frame
                    x = rearrange(x, "b t n c -> (b t) n c")

            s = attn(x)
            x = s + x  # residual

            if swap and idx % 2 != 0 and collect_attn is not None:
                collect_attn.append(s.detach().cpu())

            x = ff(x) + x  # residual

            if swap:
                if idx % 2 == 0:
                    x = rearrange(x, "(b n) t c -> b t n c", b=b)
                else:
                    x = rearrange(x, "(b t) n c -> b t n c", b=b)

        return x
