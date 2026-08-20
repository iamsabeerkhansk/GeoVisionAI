"""
visualize.py - Qualitative figures: (1) t1/t2/prediction/ground-truth
change-map panels, and (2) a confusion matrix heatmap for land cover.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def _denorm(img_tensor: torch.Tensor) -> np.ndarray:
    """[C,H,W] normalized tensor -> [H,W,C] float image in [0,1]."""
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


@torch.no_grad()
def visualize_change_detection(net, dataset, cfg, out_dir: Path | None = None, index: int = 0):
    out_dir = out_dir or (cfg.paths.output_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)
    net = net.to(device).eval()

    t1, t2, gt = dataset[index]
    prob = net(t1.unsqueeze(0).to(device), t2.unsqueeze(0).to(device))
    pred = (prob[0, 0].cpu().numpy() > 0.5).astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    axes[0].imshow(_denorm(t1)); axes[0].set_title("t\u2081 (before)"); axes[0].axis("off")
    axes[1].imshow(_denorm(t2)); axes[1].set_title("t\u2082 (after)"); axes[1].axis("off")
    axes[2].imshow(gt[0].numpy(), cmap="gray"); axes[2].set_title("Ground-truth change"); axes[2].axis("off")
    axes[3].imshow(pred, cmap="gray"); axes[3].set_title("Predicted change"); axes[3].axis("off")
    fig.suptitle("GeoVision AI \u2014 Bi-Temporal Change Detection")

    out_path = out_dir / "change_detection_panel.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved qualitative change-detection figure to {out_path}")


def plot_confusion_matrix(confusion: np.ndarray, class_names, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    normalized = confusion / confusion.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("EuroSAT Land Cover \u2014 Confusion Matrix (ViT)")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, confusion[i, j], ha="center", va="center",
                    color="white" if normalized[i, j] > 0.5 else "black", fontsize=8)

    fig.colorbar(im, ax=ax, label="Row-normalized fraction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix figure to {out_path}")
