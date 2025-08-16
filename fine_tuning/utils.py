"""Helper utilities for fine-tuning experiments."""

import os
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def parse_epoch(name: str) -> int:
    """Extract epoch number from checkpoint file name."""

    try:
        return int(os.path.splitext(os.path.basename(name))[0].split("_")[-1])
    except Exception:
        return -1


def evaluate(model, loader, device) -> Tuple[float, float, float, float, np.ndarray]:
    """Evaluate ``model`` on ``loader`` and compute metrics."""

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits, _, _ = model(imgs)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return acc, prec, rec, f1, cm
