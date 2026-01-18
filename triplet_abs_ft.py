"""
Triplet-SSL APTOS downstream DR classification
with temperature scaling + confidence-based abstention (selective prediction).

What this script does (per SSL checkpoint):
1) Load triplet-pretrained ResNet50 encoder checkpoint.
2) Fine-tune downstream classifier on APTOS train split (supervised).
3) Fit temperature scaling on a held-out calibration split (from train).
4) Evaluate on APTOS test split with abstention sweep.
5) Save:
   - epoch_xxx/abstention_metrics.csv
   - epoch_xxx/temperature.json
   - epoch_xxx/confusion_matrix.npy
   - epoch_xxx/threshold_0p70/accepted.csv and rejected.csv (manifests only, no copying)
   - summary_metrics.csv (append-safe)

Notes:
- Accepted/rejected manifests are saved from the TEST set (VAL_PATH) only.
- Split into train/calib is dynamic via random_split (no .npz saved),
  but deterministic given CALIB_SEED.
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

TRAIN_PATH = "/hpcwork/ni124545/data/aptos/train"
VAL_PATH = "/hpcwork/ni124545/data/aptos/test"

PRETRAIN_DIR = "/hpcwork/ni124545/aptos/triplet_ssl/triplet_ssl_checkpoints"
RESULTS_ROOT = "/hpcwork/ni124545/aptos/abstention"

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
    def __init__(self, threshold=10):
        self.threshold = threshold

    def __call__(self, img: Image.Image) -> Image.Image:
        import cv2
        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        arr[mask == 0] = 0
        return Image.fromarray(arr)

class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
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
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, bias=False)

    def forward(self, feat_map):
        cam = F.relu(self.conv(feat_map)).squeeze(1)  # [B,H,W]
        B, H, W = cam.shape
        flat = cam.view(B, -1)
        mn, mx = flat.min(1, True)[0], flat.max(1, True)[0] + 1e-5
        return ((flat - mn) / (mx - mn)).view(B, H, W)

class RefinementCAM(nn.Module):
    def __init__(self, thresholds=(0.3, 0.4, 0.5)):
        super().__init__()
        self.thresholds = thresholds

    def forward(self, cam, feat):
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
        B, C, H, W = feat.shape
        f = feat.view(B, C, -1)
        fn = F.normalize(f, dim=1)
        sim = torch.bmm(fn.transpose(1, 2), fn)  # [B,HW,HW]
        cf = cam.view(B, -1, 1)                  # [B,HW,1]
        out = torch.bmm(sim, cf).squeeze(-1)     # [B,HW]
        mn, mx = out.min(1, True)[0], out.max(1, True)[0] + 1e-5
        return ((out - mn) / (mx - mn)).view(B, H, W)

class ResNet50TripletSelfSup(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        resnet = models.resnet50(pretrained=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # up to avgpool
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, embedding_dim)

    def forward(self, x):
        feat = self.encoder[:-1](x)              # [B,2048,7,7]
        pooled = self.encoder[-1](feat)          # [B,2048,1,1]
        pooled = self.flatten(pooled)            # [B,2048]
        emb = self.fc(pooled)                    # [B,128]
        emb = F.normalize(emb, dim=1)
        return feat, emb

class TripletDownstreamModel(nn.Module):
    """
    Downstream classifier on top of triplet-pretrained embedding.
    Returns logits (and optionally CAM refinement loss if enabled).
    """
    def __init__(self, encoder_ckpt_path, embedding_dim=128, num_classes=5, use_cam_loss=False, alpha=0.1):
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
        feat_map, emb = self.backbone(x)
        logits = self.cls_head(emb)

        if self.use_cam_loss:
            cam0 = self.cam_ext(feat_map)
            _, lr_loss = self.refiner(cam0, feat_map)
            return logits, lr_loss

        return logits, None

# =============================================================
# 4) Metrics, calibration, abstention saving (same style as SSL code)
# =============================================================
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    pr = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rc = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return acc, pr, rc, f1, qwk

def _get_logits(out):
    # out can be (logits, lr_loss) or logits
    if isinstance(out, (tuple, list)):
        return out[0]
    return out

def evaluate_standard(model, loader):
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
    def __init__(self):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        return logits / torch.exp(self.log_T)

    @property
    def T(self):
        return float(torch.exp(self.log_T).detach().cpu().item())

def fit_temperature(model, loader):
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
    return f"threshold_{t:.2f}".replace(".", "p")

def abstention_eval_and_save_manifests(model, loader, scaler, thresholds, out_epoch_dir: str):
    """
    Saves (per epoch):
      out_epoch_dir/abstention_metrics.csv
      out_epoch_dir/threshold_0p70/accepted.csv
      out_epoch_dir/threshold_0p70/rejected.csv

    Manifests are saved from the TEST set (val_loader) only.
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
