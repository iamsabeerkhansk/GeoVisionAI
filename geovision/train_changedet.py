"""
train_changedet.py - Training loop for the Siamese ViT change detector.
Uses weighted binary cross-entropy to counter the strong class imbalance
typical of OSCD (changed pixels are a small minority), with Adam +
gradient clipping.
"""
from __future__ import annotations
import time

import torch
from torch.utils.data import DataLoader

from geovision.utils.metrics import evaluate_change_detection


def weighted_bce(pred, target, pos_weight: float, eps: float = 1e-7):
    pred = pred.clamp(eps, 1 - eps)
    loss = -(pos_weight * target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    return loss.mean()


def train_change_detection_model(net, train_ds, val_ds, cfg):
    device = torch.device(cfg.device)
    net = net.to(device)

    # Avoid creating DataLoader with an empty dataset (RandomSampler requires >0 samples)
    if len(train_ds) == 0:
        raise ValueError(
            f"Change-detection training dataset is empty. Check OSCD data at {cfg.paths.oscd_dir} "
            "and ensure `imgs_1_rect`, `imgs_2_rect`, and `cm` folders contain the expected files."
        )

    use_pin_memory = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=cfg.changedet.batch_size,
                               shuffle=True, num_workers=4, pin_memory=use_pin_memory)

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.changedet.initial_lr)
    history = []

    print(f"Starting change-detection training: {cfg.changedet.max_epochs} epochs, "
          f"batch={cfg.changedet.batch_size}, lr={cfg.changedet.initial_lr:.1e}, device={device}")

    for epoch in range(1, cfg.changedet.max_epochs + 1):
        t0 = time.time()
        net.train()
        running_loss, n_batches = 0.0, 0

        for t1, t2, mask in train_loader:
            t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)

            optimizer.zero_grad()
            prob_map = net(t1, t2)
            loss = weighted_bce(prob_map, mask, cfg.changedet.pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        val_stats = evaluate_change_detection(net, val_ds, cfg)

        print(f"[ChangeDet] Epoch {epoch:2d}/{cfg.changedet.max_epochs} | "
              f"train loss {avg_loss:.4f} | val IoU {val_stats['iou']:.3f} | "
              f"val F1 {val_stats['f1']:.3f} | {time.time()-t0:.1f}s")

        history.append({"epoch": epoch, "train_loss": avg_loss, **val_stats})

        ckpt_path = cfg.paths.checkpoint_dir / f"changedet_epoch{epoch:02d}.pt"
        torch.save({"model_state": net.state_dict()}, ckpt_path)

    return net, history
