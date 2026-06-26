"""The full model: CurveNet point encoder + axial-attention transformer head."""
import torch
from einops import rearrange
from torch import nn

from core.models.curvenet_cls import CurveNet

from .transformer import Transformer


class TemporalModel(nn.Module):
    """Point-cloud transformer for rehabilitation exercise scoring.

    Pipeline:
        1. CurveNet aggregates curve-based, geometry-aware point features per frame.
        2. A 1x1 conv downsamples the joint dimension.
        3. An axial-attention transformer mixes spatial + temporal information.
        4. A temporal transformer pools over frames.
        5. An MLP head regresses a single quality score.

    Args:
        num_joints: number of skeleton joints (input channels to the downsample conv).
        dim: transformer embedding dimension.
        spatial_depth: layers in the axial spatial-temporal transformer.
        temporal_depth: layers in the temporal-pooling transformer.
        heads: attention heads.
        dropout: dropout probability.
    """

    def __init__(
        self,
        num_joints: int = 25,
        dim: int = 256,
        spatial_depth: int = 6,
        temporal_depth: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.encoder = CurveNet()  # curve-based point aggregation
        self.downsample = nn.Sequential(
            nn.Conv1d(num_joints, 32, kernel_size=1, bias=False),
            nn.BatchNorm1d(32),
        )
        self.transformer = Transformer(dim, spatial_depth, heads, dim // heads, dim * 2, dropout)
        self.time = Transformer(dim, temporal_depth, heads, dim // heads, dim * 2, dropout)
        self.dropout = nn.Dropout(dropout)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, x, return_features=False, collect_attn=None):
        """
        Args:
            x: tensor of shape (B, T, N, C) = (batch, frames, joints, coords).
            return_features: if False, return the score only; if True, return
                ``(score, x4)``; if ``"all"``, return ``(score, x1, x2, x3, x4)``
                for t-SNE / interpretability.
            collect_attn: optional list to receive per-layer attention maps.
        """
        b, t, n, c = x.size()

        x = rearrange(x, "b t n c -> (b t) c n")
        x = x1 = rearrange(self.dropout(self.encoder(x)), "b c n -> b n c")
        x = x2 = self.downsample(x).view(b, t, 32, -1)
        x = x3 = self.transformer(x, swap=True, collect_attn=collect_attn).view(b, t, -1, self.dim).mean(2)
        x = x4 = self.time(x).mean(1)
        score = self.mlp_head(x)

        if return_features == "all":
            return score, x1, x2, x3, x4
        if return_features:
            return score, x4
        return score
