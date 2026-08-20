"""
config.py - Central configuration for the GeoVision AI pipeline.

Python/PyTorch port of the original MATLAB config.m. Edit the paths
below to point at your local copies of EuroSAT and OSCD.

    - EuroSAT : single-date multi-class LAND COVER CLASSIFICATION
    - OSCD    : bi-temporal CHANGE DETECTION (Sentinel-2 image pairs)
"""
from __future__ import annotations
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class ViTConfig:
    name: str = "vit_b_16"           # torchvision / timm backbone identifier
    input_size: tuple = (224, 224, 3)
    patch_size: int = 16
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12


@dataclass
class LandCoverConfig:
    classes: List[str] = field(default_factory=lambda: [
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
        "Industrial", "Pasture", "PermanentCrop", "Residential",
        "River", "SeaLake",
    ])
    batch_size: int = 32
    max_epochs: int = 15
    initial_lr: float = 1e-4
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    early_stop_patience: int = 4

    @property
    def num_classes(self) -> int:
        return len(self.classes)


@dataclass
class ChangeDetConfig:
    patch_size: int = 96      # sliding-window patch size on full scenes
    stride: int = 48
    batch_size: int = 16
    max_epochs: int = 30
    initial_lr: float = 5e-5
    pos_weight: float = 4.0   # class-imbalance weight (changed pixels are rare)


@dataclass
class Paths:
    root: Path = Path(__file__).resolve().parent.parent
    eurosat_dir: Path = None
    oscd_dir: Path = None
    output_dir: Path = None
    checkpoint_dir: Path = None

    def __post_init__(self):
        self.eurosat_dir = self.eurosat_dir or (self.root / "data" / "EuroSAT")
        self.oscd_dir = self.oscd_dir or (self.root / "data" / "OSCD")
        self.output_dir = self.output_dir or (self.root / "outputs")
        self.checkpoint_dir = self.checkpoint_dir or (self.output_dir / "checkpoints")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    vit: ViTConfig = field(default_factory=ViTConfig)
    landcover: LandCoverConfig = field(default_factory=LandCoverConfig)
    changedet: ChangeDetConfig = field(default_factory=ChangeDetConfig)
    seed: int = 42
    device: str = "cuda"  # falls back to cpu automatically in get_config()


def get_config() -> Config:
    cfg = Config()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    try:
        import torch
        torch.manual_seed(cfg.seed)
        if not torch.cuda.is_available():
            cfg.device = "cpu"
    except ImportError:
        cfg.device = "cpu"
    return cfg
