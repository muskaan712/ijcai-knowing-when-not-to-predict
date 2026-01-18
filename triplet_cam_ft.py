"""
Triplet-SSL APTOS downstream DR classification with CAM refinement.

High-level overview:
1) Load triplet-pretrained ResNet50 encoder checkpoint.
2) Fine-tune downstream classifier on APTOS train split with CAM refinement.
3) Evaluate at selected downstream epochs.
4) CAM refinement adds self-attention regularization to improve feature localization.
"""
import os
import glob
import warnings
from datetime import datetime
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
from torchvision.datasets import ImageFolder
from torchvision import models
from tqdm import tqdm

# ─── Config & Device ─────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Anonymous paths - replace with your data directories
DATA_ROOT = os.getenv("DATA_ROOT", "/path/to/data")
EXPERIMENTS_ROOT = os.getenv("EXPERIMENTS_ROOT", "/path/to/experiments")

# ─── 1) Data Transforms ───────────────────────────────────────────────────────
class RemoveBackgroundTransform:
    """
    Remove low-intensity background pixels from medical images.
    
    Converts RGB to grayscale, applies binary threshold to identify background,
    and sets background pixels to zero.
    """
    def __init__(self, threshold=10):
        """
        Args:
            threshold (int): Gray value threshold for background detection. Default: 10.
        """
        self.threshold = threshold

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): Input image.
        
        Returns:
            PIL.Image: Image with background removed.
        """
        import cv2
        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        arr[mask == 0] = 0
        return Image.fromarray(arr)

class CLAHETransform:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Enhances local contrast in the L channel (LAB color space) to improve
    visibility of retinal features in medical images.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Args:
            clip_limit (float): Clip limit for CLAHE. Default: 2.0.
            tile_grid_size (tuple): Grid size for adaptive histogram. Default: (8, 8).
        """
        import cv2
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): Input image.
        
        Returns:
            PIL.Image: CLAHE-enhanced image.
        """
        import cv2
        arr = np.array(img)
        if arr.ndim == 3 and arr.shape[2] == 3:
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge((l,a,b))
            arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            arr = self.clahe.apply(arr)
        return Image.fromarray(arr)

transform = T.Compose([
    T.Resize((224, 224)),
    RemoveBackgroundTransform(),
    CLAHETransform(),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
])

# ─── 2) Datasets & Loaders ───────────────────────────────────────────────────
TRAIN_PATH = os.path.join(DATA_ROOT, "aptos", "train")
VAL_PATH   = os.path.join(DATA_ROOT, "aptos", "test")
BATCH_SIZE  = 32
NUM_WORKERS = os.cpu_count() or 4

train_ds = ImageFolder(TRAIN_PATH, transform)
val_ds   = ImageFolder(VAL_PATH,   transform)

train_loader = DataLoader(train_ds, BATCH_SIZE, True,
                          num_workers=NUM_WORKERS, drop_last=True,  pin_memory=True)
val_loader   = DataLoader(val_ds,   BATCH_SIZE, False,
                          num_workers=NUM_WORKERS, drop_last=False, pin_memory=True)

print(f"Dataset sizes ▶ train: {len(train_ds)}   val: {len(val_ds)}")

# ─── 3) CAM modules ──────────────────────────────────────────────────────────
class CAMExtractor(nn.Module):
    """
    Extract Class Activation Map (CAM) from feature maps.
    
    Applies a 1x1 convolution to generate channel-wise attention,
    then normalizes per sample for interpretability.
    """
    def __init__(self, in_ch):
        """
        Args:
            in_ch (int): Number of input channels (feature map channels).
        """
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, bias=False)

    def forward(self, feat_map):
        """
        Args:
            feat_map (torch.Tensor): Feature maps (B, C, H, W).
        
        Returns:
            torch.Tensor: Normalized CAM (B, H, W).
        """
        cam = nn.functional.relu(self.conv(feat_map)).squeeze(1)  # [B,H,W]
        B,H,W = cam.shape
        flat = cam.view(B, -1)
        mn, mx = flat.min(1, True)[0], flat.max(1, True)[0] + 1e-5
        return ((flat - mn) / (mx - mn)).view(B, H, W)

class RefinementCAM(nn.Module):
    """
    Refine CAM using multi-threshold masking and self-attention.
    
    Creates soft masks at multiple confidence thresholds, applies spatial
    attention to feature maps, and regularizes refined CAM with L1 loss.
    """
    def __init__(self, thresholds=(0.3,0.4,0.5)):
        """
        Args:
            thresholds (tuple): CAM confidence thresholds for multi-scale masking. 
                Default: (0.3, 0.4, 0.5).
        """
        super().__init__()
        self.thresholds = thresholds

    def forward(self, cam, feat):
        """
        Args:
            cam (torch.Tensor): Input CAM (B, H, W).
            feat (torch.Tensor): Feature maps (B, C, H, W).
        
        Returns:
            tuple: (refined_cam, refinement_loss).
        """
        masks = [(cam >= t).float() for t in self.thresholds]
        m = torch.stack(masks,1).mean(1).unsqueeze(1)  # [B,1,H,W]
        if m.shape[-2:] != feat.shape[-2:]:
            raise RuntimeError("CAM/feat map size mismatch")
        masked = feat * m
        ref = self.self_att(cam, masked)
        loss = nn.functional.l1_loss(ref, cam.detach())
        return ref, loss

    @staticmethod
    def self_att(cam, feat):
        """
        Apply self-attention over feature maps weighted by CAM.
        
        Computes pairwise feature similarity and aggregates using CAM as weights.
        
        Args:
            cam (torch.Tensor): CAM (B, H, W).
            feat (torch.Tensor): Masked features (B, C, H, W).
        
        Returns:
            torch.Tensor: Refined CAM (B, H, W).
        """
        B,C,H,W = feat.shape
        f = feat.view(B, C, -1)
        fn = nn.functional.normalize(f, dim=1)
        sim = torch.bmm(fn.transpose(1,2), fn)      # [B,HW,HW]
        cf = cam.view(B, -1, 1)                    # [B,HW,1]
        out = torch.bmm(sim, cf).squeeze(-1)       # [B,HW]
        mn,mx = out.min(1,True)[0], out.max(1,True)[0] + 1e-5
        return ((out - mn)/(mx - mn)).view(B, H, W)

# ─── 4) Triplet‐pretrained ResNet50 backbone ─────────────────────────────────
class ResNet50TripletSelfSup(nn.Module):
    """
    ResNet50 encoder trained with triplet loss for self-supervised learning.
    
    Extracts feature maps and normalized embeddings for downstream tasks.
    """
    def __init__(self, embedding_dim=128):
        """
        Args:
            embedding_dim (int): Dimension of normalized embeddings. Default: 128.
        """
        super().__init__()
        resnet = models.resnet50(pretrained=False)
        # children() includes: conv1...layer4, avgpool, fc
        # encoder includes up to avgpool
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, embedding_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images (B, 3, 224, 224).
        
        Returns:
            tuple: (feature_maps, normalized_embeddings).
        """
        # x -> [B,3,224,224]
        feat = self.encoder[:-1](x)    # exclude avgpool -> [B,2048,7,7]
        pooled = self.encoder[-1](feat)  # avgpool -> [B,2048,1,1]
        pooled = self.flatten(pooled)    # [B,2048]
        emb = self.fc(pooled)            # [B,128]
        emb = nn.functional.normalize(emb, dim=1)
        return feat, emb

# ─── 5) Downstream CAM‐enabled model ─────────────────────────────────────────
class CAMFinetuneResNet50(nn.Module):
    """
    Downstream classifier on top of triplet-pretrained embedding with CAM.
    
    Loads triplet SSL encoder, adds classification head, and includes
    CAM refinement loss for interpretability and feature localization.
    """
    def __init__(self, encoder_ckpt_path, embedding_dim=128, num_classes=5, alpha=0.1):
        """
        Args:
            encoder_ckpt_path (str): Path to triplet SSL encoder checkpoint.
            embedding_dim (int): Embedding dimension. Default: 128.
            num_classes (int): Number of output classes. Default: 5.
            alpha (float): Weight for CAM refinement loss. Default: 0.1.
        """
        super().__init__()
        # load triplet‐trained backbone
        self.backbone = ResNet50TripletSelfSup(embedding_dim=embedding_dim)
        state = torch.load(encoder_ckpt_path, map_location="cpu")
        self.backbone.encoder.load_state_dict(state, strict=True)
        self.alpha = alpha
        # classification head on embedding
        self.cls_head = nn.Linear(embedding_dim, num_classes)
        # CAM & refinement
        self.cam_ext = CAMExtractor(in_ch=2048)
        self.refiner = RefinementCAM()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images (B, 3, 224, 224).
        
        Returns:
            tuple: (logits, cam, refinement_loss).
        """
        feat_map, emb = self.backbone(x)
        logits = self.cls_head(emb)
        cam0 = self.cam_ext(feat_map)               # [B,7,7]
        cam, lr_loss = self.refiner(cam0, feat_map) # [B,7,7], scalar
        return logits, cam, lr_loss

# ─── 6) Evaluation function ──────────────────────────────────────────────────
def evaluate(model, loader):
    """
    Evaluate classification model on a dataset.
    
    Args:
        model (nn.Module): Model to evaluate.
        loader (DataLoader): Data loader.
    
    Returns:
        tuple: (accuracy, precision, recall, f1, confusion_matrix).
    """
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits, _, _ = model(imgs)
            ys.extend(labels.cpu().tolist())
            ps.extend(logits.argmax(1).cpu().tolist())
    return (
        accuracy_score(ys, ps),
        precision_score(ys, ps, average="macro", zero_division=0),
        recall_score(ys, ps, average="macro", zero_division=0),
        f1_score(ys, ps, average="macro", zero_division=0),
        confusion_matrix(ys, ps),
    )

# ─── 7) Chunking & checkpoint loop ──────────────────────────────────────────
PRETRAIN_DIR       = os.path.join(EXPERIMENTS_ROOT, "aptos", "triplet_ssl", "triplet_ssl_checkpoints")
RESULTS_ROOT       = os.path.join(EXPERIMENTS_ROOT, "aptos", "triplet_ssl", "downstream", "jigsaw", "2550", "chunk0")
TOTAL_DS_EPOCHS    = 50
EVAL_EPOCHS        = {25, 50}
NUM_CLASSES        = 5
PARTIAL_SAVE_EVERY = 20

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 5))
CHUNK_IDX  = int(os.getenv("CHUNK_IDX", os.getenv("SLURM_ARRAY_TASK_ID", 0)))

os.makedirs(RESULTS_ROOT, exist_ok=True)

all_ckpts = sorted(glob.glob(os.path.join(PRETRAIN_DIR, "encoder_epoch_*.pth")))
ckpts = all_ckpts[CHUNK_IDX*CHUNK_SIZE:(CHUNK_IDX+1)*CHUNK_SIZE]
print(f"Chunk {CHUNK_IDX}: {len(ckpts)} checkpoints → {ckpts}")
if not ckpts:
    print("No checkpoints to process in this chunk"); exit(0)

CSV_PATH    = os.path.join(RESULTS_ROOT, "summary_metrics.csv")
append_head = not os.path.exists(CSV_PATH)
records     = []

# =============================================================
# 8) Main Training Loop
# =============================================================
for ckpt in ckpts:
    run_name = os.path.splitext(os.path.basename(ckpt))[0]
    run_dir  = os.path.join(RESULTS_ROOT, run_name)
    os.makedirs(run_dir, exist_ok=True)

    model = CAMFinetuneResNet50(ckpt, embedding_dim=128, num_classes=NUM_CLASSES, alpha=0.1).to(device)
    crit  = nn.CrossEntropyLoss()
    optim_ = optim.SGD(model.parameters(), lr=3e-3, momentum=0.9, weight_decay=1e-4)
    sched  = optim.lr_scheduler.CosineAnnealingLR(optim_, TOTAL_DS_EPOCHS)

    for epoch in range(1, TOTAL_DS_EPOCHS+1):
        model.train()
        epoch_loss, nb = 0.0, 0
        for imgs, labels in tqdm(train_loader, desc=f"{run_name} Ep{epoch:02d}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optim_.zero_grad()
            logits, cam, lr_loss = model(imgs)
            loss = crit(logits, labels) + model.alpha * lr_loss
            loss.backward()
            optim_.step()
            epoch_loss += loss.item()
            nb += 1
        sched.step()
        avg_loss = epoch_loss/nb if nb else 0
        print(f"{run_name} Epoch {epoch}/{TOTAL_DS_EPOCHS} — avg_loss {avg_loss:.4f}")

        if epoch % PARTIAL_SAVE_EVERY == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optim_.state_dict(),
            }, os.path.join(run_dir, "latest.pt"))

        if epoch in EVAL_EPOCHS:
            acc, prec, rec, f1, cm = evaluate(model, val_loader)
            print(f"Eval ▶ {run_name} | ds_epoch {epoch} | acc {acc:.4f} prec {prec:.4f} rec {rec:.4f} f1 {f1:.4f}")
            torch.save({
                "pretrain_ckpt": ckpt,
                "downstream_epoch": epoch,
                "model_state_dict": model.state_dict(),
            }, os.path.join(run_dir, f"downstream_ep{epoch}.pt"))
            np.save(os.path.join(run_dir, f"confmat_ep{epoch}.npy"), cm)

            records.append({
                "pretrain_ckpt": run_name,
                "downstream_epoch": epoch,
                "accuracy": round(acc*100,2),
                "precision": round(prec*100,2),
                "recall": round(rec*100,2),
                "f1_score": round(f1*100,2),
            })

    del model, optim_, sched
    torch.cuda.empty_cache()

# =============================================================
# 9) Save Summary (append-safe)
# =============================================================
if records:
    pd.DataFrame(records).to_csv(CSV_PATH, mode="a", header=append_head, index=False)
    print(f"Appended metrics to {CSV_PATH}")
else:
    print("No results recorded for this chunk.")
