"""
train_landcover.py - Fine-tunes the ViT land-cover classifier on
EuroSAT using cross-entropy loss with a step-decayed learning rate and
early stopping on validation loss.
"""
from __future__ import annotations
import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_landcover_model(net, train_ds, val_ds, cfg):
    device = torch.device(cfg.device)
    net = net.to(device)

    use_pin_memory = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=cfg.landcover.batch_size,
                               shuffle=True, num_workers=4, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_ds, batch_size=cfg.landcover.batch_size,
                             shuffle=False, num_workers=4, pin_memory=use_pin_memory)

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.landcover.initial_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    patience_ctr = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(f"Starting land-cover fine-tuning: {cfg.landcover.max_epochs} epochs, "
          f"batch={cfg.landcover.batch_size}, lr={cfg.landcover.initial_lr:.1e}, device={device}")

    for epoch in range(1, cfg.landcover.max_epochs + 1):
        t0 = time.time()
        net.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_ds)

        net.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = net(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
        val_loss /= len(val_ds)
        val_acc = correct / len(val_ds)

        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:2d}/{cfg.landcover.max_epochs} | "
              f"train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
              f"val acc {val_acc*100:.2f}% | {time.time()-t0:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(net.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.landcover.early_stop_patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no val-loss improvement for {cfg.landcover.early_stop_patience} epochs)")
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    ckpt_path = cfg.paths.checkpoint_dir / "landcover_vit_final.pt"
    torch.save({"model_state": net.state_dict(), "history": history}, ckpt_path)
    print(f"Saved best land-cover checkpoint to {ckpt_path}")

    return net, history
