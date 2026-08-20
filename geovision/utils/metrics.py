"""
metrics.py - Evaluation metrics.
    - evaluate_change_detection: IoU, F1, precision, recall for the
      "changed" pixel class.
    - evaluate_landcover: overall accuracy, per-class precision/recall/F1,
      macro-F1, Cohen's kappa, confusion matrix.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_change_detection(net, dataset, cfg, threshold: float = 0.5):
    device = torch.device(cfg.device)
    net = net.to(device).eval()
    loader = DataLoader(dataset, batch_size=cfg.changedet.batch_size,
                         shuffle=False, num_workers=2)

    tp = fp = fn = tn = 0
    for t1, t2, mask in loader:
        t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)
        prob = net(t1, t2)
        pred = (prob > threshold)
        gt = (mask > 0.5)

        tp += (pred & gt).sum().item()
        fp += (pred & ~gt).sum().item()
        fn += (~pred & gt).sum().item()
        tn += (~pred & ~gt).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = tp / max(tp + fp + fn, 1)
    overall_acc = (tp + tn) / max(tp + fp + fn + tn, 1)

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "iou": iou, "overall_accuracy": overall_acc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _cohens_kappa(C: np.ndarray) -> float:
    n = C.sum()
    po = np.trace(C) / n
    row_marg = C.sum(axis=1)
    col_marg = C.sum(axis=0)
    pe = np.sum(row_marg * col_marg) / (n ** 2)
    return (po - pe) / (1 - pe + 1e-12)


@torch.no_grad()
def evaluate_landcover(net, dataset, cfg):
    device = torch.device(cfg.device)
    net = net.to(device).eval()
    loader = DataLoader(dataset, batch_size=cfg.landcover.batch_size,
                         shuffle=False, num_workers=2)

    num_classes = cfg.landcover.num_classes
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = net(imgs)
        preds = outputs.argmax(1).cpu().numpy()
        labels = labels.numpy()
        for t, p in zip(labels, preds):
            confusion[t, p] += 1

    overall_acc = np.trace(confusion) / confusion.sum()

    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    for k in range(num_classes):
        tp = confusion[k, k]
        fp = confusion[:, k].sum() - tp
        fn = confusion[k, :].sum() - tp
        precision[k] = tp / max(tp + fp, 1)
        recall[k] = tp / max(tp + fn, 1)
        f1[k] = 2 * precision[k] * recall[k] / max(precision[k] + recall[k], 1e-8)

    macro_f1 = f1.mean()
    kappa = _cohens_kappa(confusion)

    results = {
        "confusion_matrix": confusion,
        "overall_accuracy": overall_acc,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "macro_f1": macro_f1,
        "kappa": kappa,
    }

    print("\n=== Land Cover Classification Results ===")
    print(f"Overall Accuracy : {overall_acc*100:.2f}%")
    print(f"Macro F1-score   : {macro_f1:.4f}")
    print(f"Cohen's Kappa    : {kappa:.4f}")
    for i, cls in enumerate(cfg.landcover.classes):
        print(f"  {cls:22s} P={precision[i]:.3f}  R={recall[i]:.3f}  F1={f1[i]:.3f}")

    return results
