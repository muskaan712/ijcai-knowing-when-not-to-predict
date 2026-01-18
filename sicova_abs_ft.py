"""
Downstream DR classification with confidence-based abstention (selective prediction)
and temperature scaling calibration.

High-level overview:
1) Load SSL-pretrained ResNet50 backbones.
2) Fine-tune downstream classifier on APTOS (supervised).
3) Evaluate at selected downstream epochs.
4) Apply confidence-based abstention using calibrated probabilities.

Important clarification (for supervision / paper):
- Downstream training is fully supervised (CrossEntropy on labels).
- Abstention is post-hoc and does NOT affect training.
- Temperature scaling is supervised calibration using a held-out split.
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
from torchvision import models
from torchvision.datasets import ImageFolder
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =============================================================
# 0) CONFIGURATION (edit here)
# =============================================================
DATASET_NAME = "aptos"
ARCH_NAME = "resnet50"
SSL_TAG = "VR1_CLAHE_Jigsaw"

# Anonymous paths - replace with your data directories
DATA_ROOT = os.getenv("DATA_ROOT", "/path/to/data")
EXPERIMENTS_ROOT = os.getenv("EXPERIMENTS_ROOT", "/path/to/experiments")

TRAIN_PATH = os.path.join(DATA_ROOT, "aptos", "train")
VAL_PATH = os.path.join(DATA_ROOT, "aptos", "test")

PRETRAIN_DIR = os.path.join(EXPERIMENTS_ROOT, "pretrain", SSL_TAG)
RESULTS_ROOT = os.path.join(EXPERIMENTS_ROOT, "aptos", "abs", "downstreamresults")

NUM_CLASSES = 5
BATCH_SIZE = 32
TOTAL_DS_EPOCHS = 50
EVAL_EPOCHS = {25, 50}

# Split config (dynamic random_split like before, but deterministic via seed)
CALIB_FRAC = 0.10
CALIB_SEED = 42

# Abstention thresholds (dense sweep)
THRESH_START = 0.50
THRESH_END = 0.95
THRESH_STEP = 0.05

LR = 0.003
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

NUM_WORKERS = os.cpu_count() or 4

# Chunking (SLURM array support)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 5))
CHUNK_IDX = int(os.getenv("CHUNK_IDX", os.getenv("SLURM_ARRAY_TASK_ID", 0)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

os.makedirs(RESULTS_ROOT, exist_ok=True)

# =============================================================
# Compatibility helper (required for loading old checkpoints)
# =============================================================
def exclude_bias_and_norm(n):
    """
    Dummy re-definition for checkpoint deserialization compatibility.

    This function was referenced during SSL pretraining and must exist
    at load time for torch.load to succeed. It is NOT used in downstream
    training or evaluation.
    
    Args:
        n (str): Parameter name from state dict.
    
    Returns:
        bool: True if name contains bias or norm layers.
    """
    return n.endswith(".bias") or "norm" in n.lower()


# =============================================================
# 1) DATA TRANSFORMS
# =============================================================
class RemoveBackgroundTransform:
    """
    Removes background from medical images using binary thresholding.
    
    Converts RGB to grayscale, applies binary threshold to identify background,
    and sets background pixels to zero.
    """
    def __init__(self, threshold: int = 10):
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
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        arr[mask == 0] = 0
        return Image.fromarray(arr)

class CLAHETransform:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Enhances local contrast in the L channel (LAB color space) to improve
    visibility of retinal features in fundus images.
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
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        lab = cv2.merge((l, a, b))
        arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(arr)

transform = T.Compose(
    [
        T.Resize((512, 512)),
        RemoveBackgroundTransform(10),
        CLAHETransform(),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ]
)

# =============================================================
# 2) DATASETS AND LOADERS (dynamic random_split, no .npz)
# =============================================================
train_full = ImageFolder(TRAIN_PATH, transform=transform)
val_ds = ImageFolder(VAL_PATH, transform=transform)

n_calib = int(len(train_full) * CALIB_FRAC)
n_train = len(train_full) - n_calib

g = torch.Generator().manual_seed(CALIB_SEED)
train_ds, calib_ds = random_split(train_full, [n_train, n_calib], generator=g)

train_loader = DataLoader(
    train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True
)
calib_loader = DataLoader(
    calib_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)
val_loader = DataLoader(
    val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

print(
    f"Dataset sizes ▶ train {len(train_ds)} | calib {len(calib_ds)} | val {len(val_ds)}"
)
print(f"Calibration split ▶ CALIB_FRAC={CALIB_FRAC}, CALIB_SEED={CALIB_SEED}\n")

# =============================================================
# 3) MODEL DEFINITIONS
# =============================================================
class VICRegNet(nn.Module):
    """
    SSL-pretrained ResNet50 backbone with feature extraction.
    
    Extracts feature maps, pooled representations, and expanded embeddings.
    Used as feature extractor for downstream classification task.
    """
    def __init__(self):
        """Initialize ResNet50 backbone and expansion head."""
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # kept only for checkpoint compatibility
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
        """
        Args:
            x (torch.Tensor): Input images (B, 3, H, W).
        
        Returns:
            tuple: (feature_maps, pooled_features, expanded_embeddings)
        """
        fmap = self.backbone(x)
        pooled = self.avgpool(fmap).view(x.size(0), -1)
        emb = self.expander(pooled)
        return fmap, pooled, emb

class LinearClassifier(nn.Module):
    """
    Simple linear classifier head on ResNet50 pooled features.
    
    Maps pooled backbone features (2048-d) to NUM_CLASSES predictions
    for downstream DR classification task.
    """
    def __init__(self, backbone: nn.Module, num_classes: int = 5):
        """
        Args:
            backbone (nn.Module): Pretrained VICRegNet backbone.
            num_classes (int): Number of output classes. Default: 5.
        """
        super().__init__()
        self.backbone = backbone
        self.cls = nn.Linear(2048, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images (B, 3, H, W).
        
        Returns:
            torch.Tensor: Logits (B, num_classes).
        """
        _, pooled, _ = self.backbone(x)
        logits = self.cls(pooled)
        return logits

# =============================================================
# 4) METRICS, CALIBRATION, ABSTENTION + MANIFESTS
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

def evaluate(model, loader):
    """
    Evaluate model on a dataset.
    
    Args:
        model (nn.Module): Model to evaluate.
        loader (DataLoader): Data loader.
    
    Returns:
        tuple: (accuracy, precision, recall, f1, qwk, confusion_matrix).
    """
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            y_true.extend(y.cpu().tolist())
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
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits_all.append(model(x))
            labels_all.append(y)

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

def abstention_eval_and_save_manifests(
    model,
    loader,
    scaler,
    thresholds,
    out_epoch_dir: str,
):
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

    global_i = 0  # relies on loader shuffle=False
    with torch.no_grad():
        for x, y in loader:
            bs = x.size(0)
            x, y = x.to(device), y.to(device)

            logits = scaler(model(x))
            probs = F.softmax(logits, dim=1)
            c, p = probs.max(dim=1)

            y_true_list.append(y.detach().cpu())
            y_pred_list.append(p.detach().cpu())
            conf_list.append(c.detach().cpu())

            for j in range(bs):
                path_list.append(val_ds.samples[global_i + j][0])
            global_i += bs

    y_true = torch.cat(y_true_list).numpy()
    y_pred = torch.cat(y_pred_list).numpy()
    conf = torch.cat(conf_list).numpy()
    n_total = len(y_true)

    # build base dataframe once (then slice per threshold)
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
# 5) CHECKPOINT SELECTION (chunk only, no exclusions)
# =============================================================
ckpt_paths = sorted(glob.glob(os.path.join(PRETRAIN_DIR, "*.pt")))

start = CHUNK_IDX * CHUNK_SIZE
end = (CHUNK_IDX + 1) * CHUNK_SIZE
ckpt_paths = ckpt_paths[start:end]

print(f"Chunk {CHUNK_IDX} → {len(ckpt_paths)} checkpoints")
for p in ckpt_paths:
    print(" ", os.path.basename(p))
print()

if not ckpt_paths:
    raise SystemExit("No checkpoints in this chunk.")

# Summary CSV (append-safe)
CSV_PATH = os.path.join(RESULTS_ROOT, "summary_metrics.csv")
append_header = not os.path.exists(CSV_PATH)
records = []

ABSTENTION_THRESHOLDS = [
    round(x, 2) for x in np.arange(THRESH_START, THRESH_END + 1e-9, THRESH_STEP)
]

# =============================================================
# 6) MAIN LOOP
# =============================================================
for ckpt in ckpt_paths:
    run_name = os.path.splitext(os.path.basename(ckpt))[0]

    run_dir = os.path.join(
        RESULTS_ROOT,
        f"{DATASET_NAME}_{ARCH_NAME}_{SSL_TAG}",
        run_name,
    )
    os.makedirs(run_dir, exist_ok=True)

    backbone = VICRegNet().to(device)
    state = torch.load(ckpt, map_location=device)
    state = state.get("model_state_dict", state)
    state = {k.replace("encoder.", "backbone."): v for k, v in state.items()}
    backbone.load_state_dict(state, strict=False)

    model = LinearClassifier(backbone, NUM_CLASSES).to(device)

    opt = optim.SGD(
        model.parameters(),
        LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, TOTAL_DS_EPOCHS)
    ce = nn.CrossEntropyLoss()

    for ep in range(1, TOTAL_DS_EPOCHS + 1):
        model.train()
        for x, y in tqdm(train_loader, leave=False, desc=f"{run_name} e{ep:03d}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = ce(model(x), y)
            loss.backward()
            opt.step()
        sched.step()

        if ep in EVAL_EPOCHS:
            epoch_dir = os.path.join(run_dir, f"epoch_{ep:03d}")
            os.makedirs(epoch_dir, exist_ok=True)

            acc, pr, rc, f1, qwk, cm = evaluate(model, val_loader)

            scaler = fit_temperature(model, calib_loader)

            with open(os.path.join(epoch_dir, "temperature.json"), "w") as f:
                json.dump({"T": scaler.T}, f, indent=2)

            np.save(os.path.join(epoch_dir, "confusion_matrix.npy"), cm)

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
                    "epoch": ep,
                    "acc": round(acc * 100, 2),
                    "precision": round(pr * 100, 2),
                    "recall": round(rc * 100, 2),
                    "f1": round(f1 * 100, 2),
                    "qwk": round(qwk, 4),
                    "temperature_T": round(scaler.T, 6),
                    "abstention_csv": os.path.join(epoch_dir, "abstention_metrics.csv"),
                }
            )

    torch.cuda.empty_cache()

# =============================================================
# 7) SAVE SUMMARY (append-safe)
# =============================================================
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
pd.DataFrame(records).to_csv(CSV_PATH, mode="a", header=append_header, index=False)
print("Done.")
