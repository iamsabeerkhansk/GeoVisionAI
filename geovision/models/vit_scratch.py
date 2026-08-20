"""
vit_scratch.py - Minimal Vision Transformer encoder implemented directly
in PyTorch, used when no pretrained backbone is available. Implements:
patch embedding (strided conv) -> learnable [CLS] token + positional
embedding -> N x transformer encoder blocks (pre-LN MHSA + MLP) ->
optional classification head.

This mirrors buildViTFromScratch.m / vitEmbeddingLayer.m /
selectCLSTokenLayer.m from the original MATLAB implementation.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class TransformerEncoderBlock(nn.Module):
    """Pre-LN transformer block: x + MHSA(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        y = self.ln1(x)
        attn_out, _ = self.attn(y, y, y, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class ViTFromScratch(nn.Module):
    """Compact ViT encoder + optional classification head.

    Forward returns:
        - if with_head=True : class logits [B, num_classes]
        - if with_head=False: patch tokens [B, num_patches+1, embed_dim]
          (index 0 is the [CLS] token, matching selectCLSTokenLayer.m)
    """

    def __init__(self, cfg, num_classes: int | None = None, with_head: bool = True):
        super().__init__()
        h, w, c = cfg.vit.input_size
        patch = cfg.vit.patch_size
        embed_dim = cfg.vit.embed_dim

        assert h % patch == 0 and w % patch == 0, "Input size must be divisible by patch size"
        self.grid_size = h // patch
        num_patches = self.grid_size * self.grid_size

        self.patch_embed = nn.Conv2d(c, embed_dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(0.02 * torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(0.02 * torch.randn(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, cfg.vit.num_heads)
            for _ in range(cfg.vit.num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.with_head = with_head
        if with_head:
            assert num_classes is not None, "num_classes required when with_head=True"
            self.head = nn.Linear(embed_dim, num_classes)

    def forward_tokens(self, x):
        # x: [B, C, H, W]
        x = self.patch_embed(x)                       # [B, embed_dim, grid, grid]
        b, d, gh, gw = x.shape
        x = x.flatten(2).transpose(1, 2)               # [B, num_patches, embed_dim]
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)                 # [B, num_patches+1, embed_dim]
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x   # [B, T, embed_dim], token 0 = [CLS]

    def forward(self, x):
        tokens = self.forward_tokens(x)
        if self.with_head:
            cls = tokens[:, 0]
            return self.head(cls)
        return tokens
