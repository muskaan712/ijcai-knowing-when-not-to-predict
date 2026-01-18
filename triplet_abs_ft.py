"""
Triplet-SSL APTOS downstream DR classification with temperature scaling and 
confidence-based abstention (selective prediction).

High-level overview:
1) Load triplet-pretrained ResNet50 encoder checkpoint.
2) Fine-tune downstream classifier on APTOS train split (supervised).
3) Fit temperature scaling on held-out calibration split (from train).
4) Evaluate on APTOS test split with abstention sweep.
5) Save per-epoch metrics, temperature, confusion matrix, and acceptance/rejection manifests.

Notes:
- Accepted/rejected manifests are saved from the TEST set (VAL_PATH) only.
- Train/calib split is dynamic via random_split (deterministic given CALIB_SEED).
- Temperature scaling is supervised calibration using held-out split.
- Abstention is post-hoc and does NOT affect training.
"""

import os
import glob
import json
import warnings

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
    cohen_kappa_score,
)
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import models
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =============================================================
# 0) CONFIG (edit here, no argparse)
# =============================================================
DATASET_NAME = "aptos"
ARCH_NAME = "resnet50"
SSL_TAG = "triplet_ssl"

# Anonymous paths - replace with your data directories
DATA_ROOT = os.getenv("DATA_ROOT", "/path/to/data")
EXPERIMENTS_ROOT = os.getenv("EXPERIMENTS_ROOT", "/path/to/experiments")

TRAIN_PATH = os.path.join(DATA_ROOT, "aptos", "train")
VAL_PATH = os.path.join(DATA_ROOT, "aptos", "test")

PRETRAIN_DIR = os.path.join(EXPERIMENTS_ROOT, "aptos", "triplet_ssl", "triplet_ssl_checkpoints")
RESULTS_ROOT = os.path.join(EXPERIMENTS_ROOT, "aptos", "abstention")

CKPT_GLOB = "encoder_epoch_*.pth"  # example: encoder_epoch_10.pth

NUM_CLASSES = 5
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count() or 4

TOTAL_DS_EPOCHS = 50
EVAL_EPOCHS = {25, 50}

# Downstream finetune hyperparams
LR = 3e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

# Calibration split from train (dynamic random_split like before)
CALIB_FRAC = 0.10
CALIB_SEED = 42

# Abstention thresholds sweep
THRESH_START = 0.50
THRESH_END = 0.95
THRESH_STEP = 0.05

# Optional CAM refinement loss during downstream finetuning
# For clean abstention/calibration studies, keep this False.
USE_CAM_LOSS = False
CAM_ALPHA = 0.1  # only used if USE_CAM_LOSS=True

# Chunking (SLURM array support)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 5))
CHUNK_IDX = int(os.getenv("CHUNK_IDX", os.getenv("SLURM_ARRAY_TASK_ID", 0)))

# =============================================================
# Device
# =============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

os.makedirs(RESULTS_ROOT, exist_ok=True)

# =============================================================
# 1) Data transforms
# =============================================================
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
    visibility of retinal features in medical images. Works on RGB and grayscale.
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
            lab = cv2.merge((l, a, b))
            arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            arr = self.clahe.apply(arr)
        return Image.fromarray(arr)

transform = T.Compose(
    [
        T.Resize((224, 224)),
        RemoveBackgroundTransform(),
        CLAHETransform(),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
    ]
)

# =============================================================
# 2) Datasets & loaders (dynamic split: train/calib, test fixed)
# =============================================================
train_full = ImageFolder(TRAIN_PATH, transform=transform)
val_ds = ImageFolder(VAL_PATH, transform=transform)

n_calib = int(len(train_full) * CALIB_FRAC)
n_train = len(train_full) - n_calib

g = torch.Generator().manual_seed(CALIB_SEED)
train_ds, calib_ds = random_split(train_full, [n_train, n_calib], generator=g)

train_loader = DataLoader(
    train_ds,
    BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    drop_last=True,
    pin_memory=True,
)
calib_loader = DataLoader(
    calib_ds,
    BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    drop_last=False,
    pin_memory=True,
)
val_loader = DataLoader(
    val_ds,
    BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    drop_last=False,
    pin_memory=True,
)

print(f"Dataset sizes ▶ train {len(train_ds)} | calib {len(calib_ds)} | val {len(val_ds)}")
print(f"Calibration split ▶ CALIB_FRAC={CALIB_FRAC}, CALIB_SEED={CALIB_SEED}\n")

# =============================================================
# 3) Model definitions (Triplet SSL backbone + downstream head)
# =============================================================
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
        cam = F.relu(self.conv(feat_map)).squeeze(1)  # [B,H,W]
        B, H, W = cam.shape
        flat = cam.view(B, -1)
        mn, mx = flat.min(1, True)[0], flat.max(1, True)[0] + 1e-5
        return ((flat - mn) / (mx - mn)).view(B, H, W)

class RefinementCAM(nn.Module):
    """
    Refine CAM using multi-threshold masking and self-attention.
    
    Creates soft masks at multiple confidence thresholds, applies spatial
    attention to feature maps, and regularizes refined CAM with L1 loss.
    """
    def __init__(self, thresholds=(0.3, 0.4, 0.5)):
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
        m = torch.stack(masks, 1).mean(1).unsqueeze(1)  # [B,1,H,W]
        if m.shape[-2:] != feat.shape[-2:]:
            raise RuntimeError("CAM/feat map size mismatch")
        masked = feat * m
        ref = self.self_att(cam, masked)
        loss = F.l1_loss(ref, cam.detach())
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
        B, C, H, W = feat.shape
        f = feat.view(B, C, -1)
        fn = F.normalize(f, dim=1)
        sim = torch.bmm(fn.transpose(1, 2), fn)  # [B,HW,HW]
        cf = cam.view(B, -1, 1)                  # [B,HW,1]
        out = torch.bmm(sim, cf).squeeze(-1)     # [B,HW]
        mn, mx = out.min(1, True)[0], out.max(1, True)[0] + 1e-5
        return ((out - mn) / (mx - mn)).view(B, H, W)

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
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # up to avgpool
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, embedding_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images (B, 3, H, W).
        
        Returns:
            tuple: (feature_maps, normalized_embeddings).
        """
        feat = self.encoder[:-1](x)              # [B,2048,7,7]
        pooled = self.encoder[-1](feat)          # [B,2048,1,1]
        pooled = self.flatten(pooled)            # [B,2048]
        emb = self.fc(pooled)                    # [B,128]
        emb = F.normalize(emb, dim=1)
        return feat, emb

class TripletDownstreamModel(nn.Module):
    """
    Downstream classifier on top of triplet-pretrained embedding.
    
    Loads triplet SSL encoder, adds classification head, and optionally
    includes CAM refinement loss for interpretability.
    """
    def __init__(self, encoder_ckpt_path, embedding_dim=128, num_classes=5, use_cam_loss=False, alpha=0.1):
        """
        Args:
            encoder_ckpt_path (str): Path to triplet SSL encoder checkpoint.
            embedding_dim (int): Embedding dimension. Default: 128.
            num_classes (int): Number of output classes. Default: 5.
            use_cam_loss (bool): Whether to use CAM refinement loss. Default: False.
            alpha (float): Weight for CAM loss. Default: 0.1.
        """
        super().__init__()
        self.use_cam_loss = use_cam_loss
        self.alpha = alpha

        self.backbone = ResNet50TripletSelfSup(embedding_dim=embedding_dim)

        # Triplet SSL ckpt expected to match backbone.encoder state dict
        state = torch.load(encoder_ckpt_path, map_location="cpu")
        self.backbone.encoder.load_state_dict(state, strict=True)

        self.cls_head = nn.Linear(embedding_dim, num_classes)

        # CAM modules (only used if use_cam_loss=True)
        self.cam_ext = CAMExtractor(in_ch=2048)
        self.refiner = RefinementCAM()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images (B, 3, H, W).
        
        Returns:
            tuple: (logits, refinement_loss or None).
        """
        feat_map, emb = self.backbone(x)
        logits = self.cls_head(emb)

        if self.use_cam_loss:
            cam0 = self.cam_ext(feat_map)
            _, lr_loss = self.refiner(cam0, feat_map)
            return logits, lr_loss

        return logits, None

# =============================================================
# 4) Metrics, calibration, abstention saving
# =============================================================
def compute_metrics(y_true, y_pred):
    """
    Compute classification metrics.
    
    Args:
        y_true (array): Ground truth labels.
        y_pred (array): Predicted labels.
    
    Returns:
        tuple: (accuracy, precision, recall, f1, qwk) - all floats in [0, 1].
    """
    acc = accuracy_score(y_true, y_pred)
    pr = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rc = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return acc, pr, rc, f1, qwk

def _get_logits(out):
    """
    Extract logits from model output (which may include optional loss).
    
    Args:
        out: Model output, either logits or (logits, loss) tuple.
    
    Returns:
        torch.Tensor: Logits tensor.
    """
    if isinstance(out, (tuple, list)):
        return out[0]
    return out

def evaluate_standard(model, loader):
    """
    Evaluate classification model on a dataset.
    
    Args:
        model (nn.Module): Model to evaluate.
        loader (DataLoader): Data loader.
    
    Returns:
        tuple: (accuracy, precision, recall, f1, qwk, confusion_matrix).
    """
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            logits = _get_logits(out)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())

    acc, pr, rc, f1, qwk = compute_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return acc, pr, rc, f1, qwk, cm

class TemperatureScaler(nn.Module):
    """
    Learnable temperature scaling for confidence calibration.
    
    Divides logits by temperature T to calibrate predicted probabilities
    to match empirical accuracy. Learned on held-out calibration set.
    """
    def __init__(self):
        """Initialize temperature parameter (log scale for numerical stability)."""
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        """
        Args:
            logits (torch.Tensor): Raw model outputs (B, C).
        
        Returns:
            torch.Tensor: Scaled logits (B, C).
        """
        return logits / torch.exp(self.log_T)

    @property
    def T(self):
        """Get temperature value (exp of log_T)."""
        return float(torch.exp(self.log_T).detach().cpu().item())

def fit_temperature(model, loader):
    """
    Fit temperature scaling on calibration set.
    
    Uses L-BFGS optimization to minimize cross-entropy loss on calibration data,
    learning a single temperature parameter that calibrates confidence estimates.
    
    Args:
        model (nn.Module): Trained model to calibrate.
        loader (DataLoader): Calibration data loader.
    
    Returns:
        TemperatureScaler: Fitted temperature scaler.
    """
    model.eval()
    scaler = TemperatureScaler().to(device)
    nll = nn.CrossEntropyLoss()

    logits_all, labels_all = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            logits = _get_logits(out)
            logits_all.append(logits)
            labels_all.append(labels)

    logits_all = torch.cat(logits_all)
    labels_all = torch.cat(labels_all)

    optimizer = optim.LBFGS([scaler.log_T], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = nll(scaler(logits_all), labels_all)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler

def _safe_threshold_folder(t: float) -> str:
    """
    Convert float threshold to safe folder name.
    
    Args:
        t (float): Threshold value (e.g., 0.70).
    
    Returns:
        str: Safe folder name (e.g., 'threshold_0p70').
    """
    return f"threshold_{t:.2f}".replace(".", "p")

def abstention_eval_and_save_manifests(model, loader, scaler, thresholds, out_epoch_dir: str):
    """
    Evaluate selective prediction and save per-threshold acceptance/rejection manifests.
    
    For each confidence threshold, computes selective accuracy and saves
    CSV files listing accepted/rejected samples with their predictions and confidence.
    
    Args:
        model (nn.Module): Trained classifier.
        loader (DataLoader): Validation data loader.
        scaler (TemperatureScaler): Fitted temperature scaler.
        thresholds (list): Confidence thresholds to evaluate.
        out_epoch_dir (str): Output directory for results.
    
    Returns:
        pd.DataFrame: Metrics per threshold (coverage, selective accuracy, F1, etc.).
    """
    model.eval()
    y_true_list, y_pred_list, conf_list, path_list = [], [], [], []

    global_i = 0  # relies on loader shuffle=False (we set it false)
    with torch.no_grad():
        for imgs, labels in loader:
            bs = imgs.size(0)
            imgs, labels = imgs.to(device), labels.to(device)

            out = model(imgs)
            logits = _get_logits(out)
            logits = scaler(logits)

            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

            y_true_list.append(labels.detach().cpu())
            y_pred_list.append(pred.detach().cpu())
            conf_list.append(conf.detach().cpu())

            for j in range(bs):
                path_list.append(val_ds.samples[global_i + j][0])
            global_i += bs

    y_true = torch.cat(y_true_list).numpy()
    y_pred = torch.cat(y_pred_list).numpy()
    conf = torch.cat(conf_list).numpy()
    n_total = len(y_true)

    df_all = pd.DataFrame(
        {
            "image_path": path_list,
            "true_label": y_true.astype(int),
            "pred_label": y_pred.astype(int),
            "confidence": conf.astype(float),
        }
    )
    df_all["is_correct"] = (df_all["true_label"] == df_all["pred_label"]).astype(int)

    rows = []
    for t in thresholds:
        keep = conf >= t
        kept_n = int(keep.sum())
        rejected_n = int(n_total - kept_n)
        coverage_percent = (kept_n / n_total) * 100.0 if n_total > 0 else 0.0

        thr_dir = os.path.join(out_epoch_dir, _safe_threshold_folder(float(t)))
        os.makedirs(thr_dir, exist_ok=True)

        df_kept = df_all[keep].copy()
        df_rej = df_all[~keep].copy()

        df_kept.to_csv(os.path.join(thr_dir, "accepted.csv"), index=False)
        df_rej.to_csv(os.path.join(thr_dir, "rejected.csv"), index=False)

        if kept_n == 0:
            rows.append(
                {
                    "threshold": float(t),
                    "coverage_percent": round(coverage_percent, 2),
                    "kept_n": kept_n,
                    "rejected_n": rejected_n,
                    "selective_acc": 0.0,
                    "selective_precision_macro": 0.0,
                    "selective_recall_macro": 0.0,
                    "selective_f1_macro": 0.0,
                    "selective_qwk": 0.0,
                }
            )
            continue

        yt = y_true[keep]
        yp = y_pred[keep]
        sel_acc, sel_pr, sel_rc, sel_f1, sel_qwk = compute_metrics(yt, yp)

        rows.append(
            {
                "threshold": float(t),
                "coverage_percent": round(coverage_percent, 2),
                "kept_n": kept_n,
                "rejected_n": rejected_n,
                "selective_acc": round(sel_acc * 100, 2),
                "selective_precision_macro": round(sel_pr * 100, 2),
                "selective_recall_macro": round(sel_rc * 100, 2),
                "selective_f1_macro": round(sel_f1 * 100, 2),
                "selective_qwk": round(sel_qwk, 4),
            }
        )

    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(os.path.join(out_epoch_dir, "abstention_metrics.csv"), index=False)
    return df_rows

# =============================================================
# 5) Checkpoint list (chunked)
# =============================================================
all_ckpts = sorted(glob.glob(os.path.join(PRETRAIN_DIR, CKPT_GLOB)))

start = CHUNK_IDX * CHUNK_SIZE
end = (CHUNK_IDX + 1) * CHUNK_SIZE
ckpts = all_ckpts[start:end]

print(f"Chunk {CHUNK_IDX} → {len(ckpts)} checkpoints")
for p in ckpts:
    print(" ", os.path.basename(p))
print()

if not ckpts:
    raise SystemExit("No checkpoints to process in this chunk.")

# thresholds list
ABSTENTION_THRESHOLDS = [
    round(x, 2) for x in np.arange(THRESH_START, THRESH_END + 1e-9, THRESH_STEP)
]

# Summary CSV (append-safe)
CSV_PATH = os.path.join(RESULTS_ROOT, "summary_metrics.csv")
append_header = not os.path.exists(CSV_PATH)
records = []

# =============================================================
# 6) Main loop
# =============================================================
for ckpt in ckpts:
    run_name = os.path.splitext(os.path.basename(ckpt))[0]

    run_dir = os.path.join(
        RESULTS_ROOT,
        f"{DATASET_NAME}_{ARCH_NAME}_{SSL_TAG}",
        run_name,
    )
    os.makedirs(run_dir, exist_ok=True)

    model = TripletDownstreamModel(
        encoder_ckpt_path=ckpt,
        embedding_dim=128,
        num_classes=NUM_CLASSES,
        use_cam_loss=USE_CAM_LOSS,
        alpha=CAM_ALPHA,
    ).to(device)

    ce = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, TOTAL_DS_EPOCHS)

    for epoch in range(1, TOTAL_DS_EPOCHS + 1):
        model.train()
        running = 0.0
        nb = 0

        for imgs, labels in tqdm(train_loader, desc=f"{run_name} e{epoch:03d}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            opt.zero_grad()
            logits, lr_loss = model(imgs)

            loss = ce(logits, labels)
            if USE_CAM_LOSS and lr_loss is not None:
                loss = loss + model.alpha * lr_loss

            loss.backward()
            opt.step()

            running += float(loss.item())
            nb += 1

        sched.step()
        avg_loss = running / nb if nb else 0.0

        if epoch in EVAL_EPOCHS:
            epoch_dir = os.path.join(run_dir, f"epoch_{epoch:03d}")
            os.makedirs(epoch_dir, exist_ok=True)

            # Standard evaluation on TEST set
            acc, pr, rc, f1, qwk, cm = evaluate_standard(model, val_loader)

            # Calibration on held-out calib split (from train)
            scaler = fit_temperature(model, calib_loader)

            with open(os.path.join(epoch_dir, "temperature.json"), "w") as f:
                json.dump({"T": scaler.T}, f, indent=2)

            np.save(os.path.join(epoch_dir, "confusion_matrix.npy"), cm)

            # Abstention + manifests (TEST set)
            _ = abstention_eval_and_save_manifests(
                model=model,
                loader=val_loader,
                scaler=scaler,
                thresholds=ABSTENTION_THRESHOLDS,
                out_epoch_dir=epoch_dir,
            )

            records.append(
                {
                    "ckpt": run_name,
                    "epoch": epoch,
                    "train_avg_loss": round(avg_loss, 6),
                    "acc": round(acc * 100, 2),
                    "precision": round(pr * 100, 2),
                    "recall": round(rc * 100, 2),
                    "f1": round(f1 * 100, 2),
                    "qwk": round(qwk, 4),
                    "temperature_T": round(scaler.T, 6),
                    "abstention_csv": os.path.join(epoch_dir, "abstention_metrics.csv"),
                }
            )

    del model, opt, sched
    torch.cuda.empty_cache()

# =============================================================
# 7) Save summary (append-safe)
# =============================================================
pd.DataFrame(records).to_csv(CSV_PATH, mode="a", header=append_header, index=False)
print(f"Done. Summary appended to: {CSV_PATH}")
