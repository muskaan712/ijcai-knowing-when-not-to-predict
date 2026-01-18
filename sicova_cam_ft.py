import os
import glob
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import ImageFolder
from tqdm import tqdm

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

def exclude_bias_and_norm(n):
    return n.endswith(".bias") or "norm" in n.lower()

# =============================================================
# 1 ──────────────────────── Data transforms
# =============================================================
class RemoveBackgroundTransform:
    """Zero‑out low‑intensity background pixels (very coarse)."""

    def __init__(self, threshold: int = 10):
        self.threshold = threshold

    def __call__(self, img: Image.Image) -> Image.Image:
        import cv2  # local import keeps global deps minimal

        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        arr[mask == 0] = 0
        return Image.fromarray(arr)


class CLAHETransform:
    """Apply CLAHE per‑image (improves local contrast)."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size=(8, 8)):
        import cv2

        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img: Image.Image) -> Image.Image:
        import cv2

        arr = np.array(img)
        if arr.ndim == 3 and arr.shape[2] == 3:
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge((l, a, b))
            arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            arr = self.clahe.apply(arr)
        return Image.fromarray(arr)


transform = T.Compose(
    [
        T.Resize((512, 512)),
        RemoveBackgroundTransform(10),
        CLAHETransform(clip_limit=2.0),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ]
)

# =============================================================
# 2 ─────────────────────────────── Dataset
# =============================================================
TRAIN_PATH = "/home/s13mchop/HybridML/data/aptos/train"
VAL_PATH = "/home/s13mchop/HybridML/data/aptos/test"
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count() or 4

train_ds = ImageFolder(TRAIN_PATH, transform)
val_ds = ImageFolder(VAL_PATH, transform)

train_loader = DataLoader(train_ds, BATCH_SIZE, True, num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)
val_loader = DataLoader(val_ds, BATCH_SIZE, False, num_workers=NUM_WORKERS, drop_last=False, pin_memory=True)

print(f"Dataset sizes  ▶  train: {len(train_ds)}   val: {len(val_ds)}")

# =============================================================
# 3 ──────────────────────────── Model pieces
# =============================================================
class CAMExtractor(nn.Module):
    def __init__(self, in_channels: int = 2048):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1, bias=False)

    def forward(self, feat_map):  # [B,C,H,W]
        cam = F.relu(self.conv(feat_map)).squeeze(1)  # [B,H,W]
        B, H, W = cam.shape
        flat = cam.view(B, -1)
        cam = (flat - flat.min(1, keepdim=True)[0]) / (flat.max(1, keepdim=True)[0] + 1e-5)
        return cam.view(B, H, W)


class RefinementCAM(nn.Module):
    def __init__(self, thresholds=(0.3, 0.4, 0.5)):
        super().__init__()
        self.thresholds = thresholds

    def forward(self, cam, feat):  # cam [B,H,W], feat [B,C,H,W]
        masks = [(cam >= t).float() for t in self.thresholds]
        mask = torch.stack(masks, 1).mean(1).unsqueeze(1)  # [B,1,H,W]
        if mask.shape[-2:] != feat.shape[-2:]:
            raise RuntimeError("CAM/feature size mismatch, check extractor.")
        masked = feat * mask
        refined = self.self_attention(cam, masked)
        loss = F.l1_loss(refined, cam.detach())
        return refined, loss

    @staticmethod
    def self_attention(cam, feat):
        B, C, H, W = feat.shape
        feat_flat = F.normalize(feat.view(B, C, -1), dim=1)
        sim = torch.bmm(feat_flat.transpose(1, 2), feat_flat)  # [B,HW,HW]
        cam_flat = cam.view(B, -1, 1)
        refined = torch.bmm(sim, cam_flat).squeeze(-1)
        refined = (refined - refined.min(1, keepdim=True)[0]) / (refined.max(1, keepdim=True)[0] + 1e-5)
        return refined.view(B, H, W)


class VICRegNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # projection head kept for weight‑loading compatibility
        self.expander = nn.Sequential(
            nn.Linear(2048, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, 8192),
        )

    def forward(self, x):
        fmap = self.backbone(x)
        pooled = self.avgpool(fmap).view(x.size(0), -1)
        embeds = self.expander(pooled)
        return fmap, pooled, embeds


class CAMClassification(nn.Module):
    def __init__(self, backbone, num_classes=5, alpha=0.1):
        super().__init__()
        self.backbone = backbone
        self.cls_head = nn.Linear(2048, num_classes)
        self.cam_extractor = CAMExtractor(2048)
        self.refiner = RefinementCAM()
        self.alpha = alpha

    def forward(self, x):
        fmap, pooled, _ = self.backbone(x)
        logits = self.cls_head(pooled)
        cam0 = self.cam_extractor(fmap)
        cam, loss_ref = self.refiner(cam0, fmap)
        return logits, cam, loss_ref


# =============================================================
# 4 ──────────────────────────── Utils
# =============================================================

def parse_epoch(name: str) -> int:
    try:
        return int(os.path.splitext(os.path.basename(name))[0].split("_")[-1])
    except Exception:
        return -1


def evaluate(model, loader):
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


# =============================================================
# 5 ─────────────────────────── Experiment config
# =============================================================
PRETRAIN_DIR = "/home/s13mchop/HybridML/experiments/pretrain/VR1_CLAHE_Jigsaw"
RESULTS_ROOT = "/home/s13mchop/HybridML/experiments/aptos/1best_run/augs_2550/downstreamresults"
TOTAL_DS_EPOCHS = 200
EVAL_EPOCHS = {25, 50}
NUM_CLASSES = 5
PARTIAL_SAVE_EVERY = 10  # to allow resume within a ckpt

# chunk support ─────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 5))
CHUNK_IDX = int(os.getenv("CHUNK_IDX", os.getenv("SLURM_ARRAY_TASK_ID", 0)))

os.makedirs(RESULTS_ROOT, exist_ok=True)

ckpt_paths = sorted(glob.glob(os.path.join(PRETRAIN_DIR, "*.pt")))
if not ckpt_paths:
    raise FileNotFoundError(f"No checkpoints found in {PRETRAIN_DIR}")

# slice for this chunk
start, end = CHUNK_IDX * CHUNK_SIZE, (CHUNK_IDX + 1) * CHUNK_SIZE
ckpt_paths = ckpt_paths[start:end]
print(f"Chunk {CHUNK_IDX}: {len(ckpt_paths)} checkpoints → {ckpt_paths}")

if not ckpt_paths:
    print("Nothing to do for this chunk — exiting.")
    exit(0)

CSV_PATH = os.path.join(RESULTS_ROOT, "summary_metrics.csv")
append_header = not os.path.exists(CSV_PATH)

records = []

# =============================================================
# 6 ──────────────────────────── Training loop
# =============================================================
for ckpt_path in ckpt_paths:
    pre_ep = parse_epoch(ckpt_path)
    run_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    run_dir = os.path.join(RESULTS_ROOT, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # build backbone and load encoder weights → map to backbone.* keys
    backbone = VICRegNet().to(device)
    raw = torch.load(ckpt_path, map_location=device)
    state = raw.get("model_state_dict", raw)
    mapped = {k.replace("encoder.", "backbone."): v for k, v in state.items()}
    _ = backbone.load_state_dict(mapped, strict=False)

    model = CAMClassification(backbone, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(model.parameters(), lr=0.003, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, TOTAL_DS_EPOCHS)

    # --- downstream training ---
    for epoch in range(1, TOTAL_DS_EPOCHS + 1):
        model.train()
        for imgs, labels in tqdm(train_loader, leave=False, desc=f"{run_name} e{epoch:03d}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimiser.zero_grad()
            logits, _, loss_ref = model(imgs)
            loss = criterion(logits, labels) + model.alpha * loss_ref
            loss.backward()
            optimiser.step()
        scheduler.step()

        # partial save for resume safety
        if epoch % PARTIAL_SAVE_EVERY == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optimiser.state_dict(),
            }, os.path.join(run_dir, "latest.pt"))

        if epoch in EVAL_EPOCHS:
            acc, prec, rec, f1, cm = evaluate(model, val_loader)
            print(
                f"Eval ▶ pre‑ep {pre_ep:3d}  ds‑ep {epoch:3d}  "
                f"acc {acc:.4f}  prec {prec:.4f}  rec {rec:.4f}  f1 {f1:.4f}"
            )

            # save downstream ckpt & confusion matrix
            torch.save({
                "pretrain_ckpt": ckpt_path,
                "downstream_epoch": epoch,
                "model_state_dict": model.state_dict(),
            }, os.path.join(run_dir, f"downstream_epoch{epoch}.pt"))
            np.save(os.path.join(run_dir, f"confusion_matrix_epoch{epoch}.npy"), cm)

            records.append({
                "pretrain_epoch": pre_ep,
                "downstream_epoch": epoch,
                "accuracy": round(acc * 100, 2),
                "precision": round(prec * 100, 2),
                "recall": round(rec * 100, 2),
                "f1_score": round(f1 * 100, 2),
            })

    # cleanup
    del model, backbone, optimiser, scheduler
    torch.cuda.empty_cache()

# =============================================================
# 7 ──────────────────────────── Save aggregated CSV
# =============================================================
pd.DataFrame(records).to_csv(CSV_PATH, mode="a", header=append_header, index=False)
print(f"All done for chunk {CHUNK_IDX}. Results appended to {CSV_PATH}.")
