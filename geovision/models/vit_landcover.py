"""
vit_landcover.py - Loads a pretrained Vision Transformer (torchvision
ViT-B/16, ImageNet weights) and replaces its classification head for
EuroSAT's 10 land-cover classes.

Falls back to a from-scratch ViT (see vit_scratch.py) if torchvision /
the pretrained weights are unavailable (e.g. no internet access).
"""
from __future__ import annotations
import torch
import torch.nn as nn


def build_landcover_vit(cfg):
    try:
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        net = vit_b_16(weights=weights)
        in_features = net.heads.head.in_features
        net.heads.head = nn.Linear(in_features, cfg.landcover.num_classes)
        print(f"Loaded pretrained ViT-B/16 and replaced classification head "
              f"for {cfg.landcover.num_classes} classes.")
        return net
    except Exception as e:  # noqa: BLE001
        print(f"[warn] Pretrained ViT unavailable ({e}). "
              f"Building ViT from scratch instead.")
        from geovision.models.vit_scratch import ViTFromScratch
        return ViTFromScratch(cfg, num_classes=cfg.landcover.num_classes, with_head=True)
