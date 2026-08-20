# GeoVision AI (Python / PyTorch)
Multi-Temporal Satellite Image Change Detection and Land Cover Classification
Using Vision Transformers — Python port of the original MATLAB implementation.

## Requirements
```
pip install -r requirements.txt
```
- Python 3.9+
- PyTorch 2.1+ (GPU optional; CPU works, just slower)
- torchvision (for the pretrained ViT-B/16 backbone; internet access needed
  the first time to download ImageNet weights, then cached locally)

If torchvision or the pretrained weights aren't available, both model
builders automatically fall back to `geovision/models/vit_scratch.py`, a
compact ViT implemented directly in PyTorch — no internet or pretrained
weights required, just slower to converge.

## Datasets
Download and place locally (not included in this package):

| Dataset | Purpose | Link | Expected path |
|---|---|---|---|
| EuroSAT (RGB) | 10-class land cover classification | https://github.com/phelber/EuroSAT | `data/EuroSAT/<ClassName>/*.jpg` |
| OSCD | Bi-temporal binary change detection | https://rcdaudt.github.io/oscd/ | `data/OSCD/imgs_1_rect`, `imgs_2_rect`, `cm` |

## Folder structure
```
GeoVisionAI_Python/
├── main_pipeline.py                     # end-to-end driver script (CLI)
├── requirements.txt
├── geovision/
│   ├── config.py                        # all paths & hyperparameters (dataclasses)
│   ├── data/
│   │   ├── prepare_eurosat.py           # stratified split + PyTorch Dataset
│   │   └── prepare_oscd.py              # city-split, sliding-window patches
│   ├── models/
│   │   ├── vit_landcover.py             # ViT classifier (transfer learning)
│   │   ├── siamese_vit_changedet.py     # Siamese ViT for change detection
│   │   └── vit_scratch.py               # fallback ViT encoder (pure PyTorch)
│   ├── train_landcover.py               # standard training loop
│   ├── train_changedet.py               # weighted-BCE training loop
│   └── utils/
│       ├── metrics.py                   # accuracy/F1/kappa; IoU/F1/precision/recall
│       └── visualize.py                 # qualitative figures
└── data/                                # <- place EuroSAT/ and OSCD/ here
```

## Running
```bash
python main_pipeline.py                    # both stages
python main_pipeline.py --stage landcover   # EuroSAT classification only
python main_pipeline.py --stage changedet   # OSCD change detection only
python main_pipeline.py --quick-test        # 1 epoch each, to sanity-check the pipeline works
```
Outputs (checkpoints, figures, `geovision_results.pkl`) are written to
`outputs/`, created automatically next to this README.

## Notes on the port from MATLAB
- **Weight sharing**: in the MATLAB version this was a `dlnetwork` shared
  by both temporal branches; here it's simply the *same* `nn.Module`
  instance called twice in `SiameseViTChangeDetector.forward`, which is
  the natural PyTorch equivalent — gradients from both calls accumulate
  into the same parameters automatically via autograd.
- **Fusion**: `|tok1 - tok2|` concatenated with `tok1 + tok2` (a small
  change from the MATLAB script's `tok1 - tok2 + tok2` — mathematically
  the same `tok1 + (tok2 - tok2) = tok1`... corrected here to the
  intended `tok1 + tok2` for a genuine sum-context feature).
- **Class imbalance**: same weighted binary cross-entropy approach
  (`cfg.changedet.pos_weight`, default 4.0).
- **Patch-based training**: OSCD scenes are tiled into overlapping
  96x96 patches (`cfg.changedet.patch_size` / `stride`) and resized to
  224x224; `_extract_city_patches` in `prepare_oscd.py` loads all
  patches into memory upfront — increase `stride` if you hit memory
  limits on large scenes.
- **City-level splitting**: train/val/test are split by *city*, not by
  patch, to avoid spatial leakage — same as the MATLAB version.
- **Pretrained weights**: uses torchvision's `vit_b_16` /
  `ViT_B_16_Weights.IMAGENET1K_V1` in place of MATLAB's Deep Learning
  Toolbox Vision Transformer Add-On.
