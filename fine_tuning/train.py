"""Fine-tuning classifier using pre-trained backbones."""

import os
import glob
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from transforms import build_transform
from model import VICRegNet, CAMClassification
from utils import parse_epoch, evaluate

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

transform = build_transform()

TRAIN_PATH = "/home/s13mchop/HybridML/data/aptos/train"
VAL_PATH = "/home/s13mchop/HybridML/data/aptos/test"
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count() or 4

train_ds = ImageFolder(TRAIN_PATH, transform)
val_ds = ImageFolder(VAL_PATH, transform)

train_loader = DataLoader(
    train_ds,
    BATCH_SIZE,
    True,
    num_workers=NUM_WORKERS,
    drop_last=True,
    pin_memory=True,
)
val_loader = DataLoader(
    val_ds,
    BATCH_SIZE,
    False,
    num_workers=NUM_WORKERS,
    drop_last=False,
    pin_memory=True,
)

print(f"Dataset sizes  ▶  train: {len(train_ds)}   val: {len(val_ds)}")

PRETRAIN_DIR = "/home/s13mchop/HybridML/experiments/pretrain/VR1_CLAHE_Jigsaw"
RESULTS_ROOT = "/home/s13mchop/HybridML/experiments/aptos/1best_run/augs/downstream_results"
TOTAL_DS_EPOCHS = 200
EVAL_EPOCHS = {100, 200}
NUM_CLASSES = 5
PARTIAL_SAVE_EVERY = 10

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 5))
CHUNK_IDX = int(os.getenv("CHUNK_IDX", os.getenv("SLURM_ARRAY_TASK_ID", 0)))

os.makedirs(RESULTS_ROOT, exist_ok=True)

ckpt_paths = sorted(glob.glob(os.path.join(PRETRAIN_DIR, "*.pt")))
if not ckpt_paths:
    raise FileNotFoundError(f"No checkpoints found in {PRETRAIN_DIR}")

start, end = CHUNK_IDX * CHUNK_SIZE, (CHUNK_IDX + 1) * CHUNK_SIZE
ckpt_paths = ckpt_paths[start:end]
print(f"Chunk {CHUNK_IDX}: {len(ckpt_paths)} checkpoints → {ckpt_paths}")

if not ckpt_paths:
    print("Nothing to do for this chunk — exiting.")
    exit(0)

CSV_PATH = os.path.join(RESULTS_ROOT, "summary_metrics.csv")
append_header = not os.path.exists(CSV_PATH)

records = []

for ckpt_path in ckpt_paths:
    pre_ep = parse_epoch(ckpt_path)
    run_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    run_dir = os.path.join(RESULTS_ROOT, run_name)
    os.makedirs(run_dir, exist_ok=True)

    backbone = VICRegNet().to(device)
    raw = torch.load(ckpt_path, map_location=device)
    state = raw.get("model_state_dict", raw)
    mapped = {k.replace("encoder.", "backbone."): v for k, v in state.items()}
    _ = backbone.load_state_dict(mapped, strict=False)

    model = CAMClassification(backbone, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(model.parameters(), lr=0.003, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, TOTAL_DS_EPOCHS)

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

        if epoch % PARTIAL_SAVE_EVERY == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optimiser.state_dict(),
                },
                os.path.join(run_dir, "latest.pt"),
            )

        if epoch in EVAL_EPOCHS:
            acc, prec, rec, f1, cm = evaluate(model, val_loader, device)
            print(
                f"Eval ▶ pre‑ep {pre_ep:3d}  ds‑ep {epoch:3d}  "
                f"acc {acc:.4f}  prec {prec:.4f}  rec {rec:.4f}  f1 {f1:.4f}"
            )

            torch.save(
                {
                    "pretrain_ckpt": ckpt_path,
                    "downstream_epoch": epoch,
                    "model_state_dict": model.state_dict(),
                },
                os.path.join(run_dir, f"downstream_epoch{epoch}.pt"),
            )
            np.save(os.path.join(run_dir, f"confusion_matrix_epoch{epoch}.npy"), cm)

            records.append(
                {
                    "pretrain_epoch": pre_ep,
                    "downstream_epoch": epoch,
                    "accuracy": round(acc * 100, 2),
                    "precision": round(prec * 100, 2),
                    "recall": round(rec * 100, 2),
                    "f1_score": round(f1 * 100, 2),
                }
            )

    del model, backbone, optimiser, scheduler
    torch.cuda.empty_cache()

pd.DataFrame(records).to_csv(CSV_PATH, mode="a", header=append_header, index=False)
print(f"All done for chunk {CHUNK_IDX}. Results appended to {CSV_PATH}.")
