"""
siamese_vit_changedet.py - Siamese Vision Transformer for bi-temporal
change detection. Two passes through a SINGLE shared ViT encoder (true
weight sharing, since it's literally the same nn.Module) extract patch
token features from image t1 and t2; a difference-fusion head + a
lightweight decoder produce a pixel-wise change probability map.

Architecture:
    t1 -> [shared ViT encoder] -> tokens1  \
                                             }-> |tok1-tok2|, concat -> fusion MLP
    t2 -> [shared ViT encoder] -> tokens2  /        -> reshape to grid
                                                     -> upsample -> sigmoid
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenFusionDecoder(nn.Module):
    """Per-token MLP that maps [|tok1-tok2| ; tok1+tok2] -> a single
    change logit per patch."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, fused_tokens):
        # fused_tokens: [B, T, 2*embed_dim] -> [B, T, 1]
        return self.net(fused_tokens)


class SiameseViTChangeDetector(nn.Module):
    def __init__(self, cfg, shared_encoder: nn.Module, encoder_kind: str = "scratch"):
        """
        shared_encoder: a module whose forward(x) returns patch tokens
            [B, T, embed_dim] with token 0 = [CLS] (ViTFromScratch style),
            OR a torchvision vit_b_16 with its head removed (see
            build_change_detection_vit for adaptation).
        encoder_kind: "scratch" or "torchvision" - controls how tokens
            are extracted from the encoder's forward pass.
        """
        super().__init__()
        self.encoder = shared_encoder
        self.encoder_kind = encoder_kind
        self.embed_dim = cfg.vit.embed_dim
        self.grid_size = cfg.vit.input_size[0] // cfg.vit.patch_size
        self.out_size = cfg.vit.input_size[:2]
        self.decoder = TokenFusionDecoder(self.embed_dim)

    def _encode(self, x):
        """Returns patch tokens only (CLS stripped): [B, num_patches, embed_dim]"""
        if self.encoder_kind == "scratch":
            tokens = self.encoder.forward_tokens(x)     # [B, T, D], T = num_patches+1
            return tokens[:, 1:, :]
        elif self.encoder_kind == "torchvision":
            tokens = self.encoder(x)                    # via forward hook, see builder
            return tokens[:, 1:, :]
        else:
            raise ValueError(f"Unknown encoder_kind: {self.encoder_kind}")

    def forward(self, t1, t2):
        tok1 = self._encode(t1)   # [B, N, D]
        tok2 = self._encode(t2)   # [B, N, D]

        diff = torch.abs(tok1 - tok2)
        fused = torch.cat([diff, tok1 + tok2], dim=-1)   # [B, N, 2D]

        logits = self.decoder(fused).squeeze(-1)          # [B, N]

        b, n = logits.shape
        grid = self.grid_size
        logits_grid = logits.view(b, 1, grid, grid)
        logits_full = F.interpolate(logits_grid, size=self.out_size,
                                     mode="bilinear", align_corners=False)
        prob_map = torch.sigmoid(logits_full)              # [B, 1, H, W]
        return prob_map


class _TorchvisionTokenWrapper(nn.Module):
    """Wraps a torchvision vit_b_16 so forward(x) returns raw patch
    tokens (including CLS) instead of class logits, by re-implementing
    the encoder forward without the final head."""

    def __init__(self, vit_model):
        super().__init__()
        self.conv_proj = vit_model.conv_proj
        self.class_token = vit_model.class_token
        self.encoder = vit_model.encoder
        self.hidden_dim = vit_model.hidden_dim
        self.patch_size = vit_model.patch_size

    def forward(self, x):
        n, c, h, w = x.shape
        x = self.conv_proj(x)                       # [n, hidden, h/p, w/p]
        x = x.reshape(n, self.hidden_dim, -1).permute(0, 2, 1)  # [n, num_patches, hidden]
        cls = self.class_token.expand(n, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)                          # adds its own positional embedding internally
        return x                                      # [n, num_patches+1, hidden]


def build_change_detection_vit(cfg):
    """Builds the Siamese change-detection model, preferring a pretrained
    torchvision ViT-B/16 encoder (weights shared by construction, since
    both temporal images are passed through the SAME module instance),
    with an automatic fallback to the from-scratch ViT encoder."""
    try:
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        base = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        encoder = _TorchvisionTokenWrapper(base)
        model = SiameseViTChangeDetector(cfg, encoder, encoder_kind="torchvision")
        print("Built Siamese ViT change detector with pretrained ViT-B/16 shared encoder.")
        return model
    except Exception as e:  # noqa: BLE001
        print(f"[warn] Pretrained ViT unavailable ({e}); using from-scratch "
              f"encoder for the Siamese branches.")
        from geovision.models.vit_scratch import ViTFromScratch
        encoder = ViTFromScratch(cfg, with_head=False)
        model = SiameseViTChangeDetector(cfg, encoder, encoder_kind="scratch")
        print("Built Siamese ViT change detector with from-scratch shared encoder.")
        return model
