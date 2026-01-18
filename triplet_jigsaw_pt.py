"""
Self-supervised pretraining using Triplet Loss with multi-stage jigsaw augmentation.

High-level overview:
1) Load unlabeled medical images from disk (organized in torchvision.ImageFolder format).
2) Apply multi-stage contrastive augmentations (CLAHE, jigsaw puzzle, color jittering, etc.).
3) Train ResNet50 encoder with labeled triplet loss (all-triplet mining strategy).
4) Save encoder checkpoints at regular intervals for downstream fine-tuning.
"""
import os
import numpy as np

import torch
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import ResNet50TripletSelfSup
from src.transforms import TwoViewTransform, get_pretraining_transform
from src.losses import LabeledTripletLoss

# Anonymous paths - replace with your data directories
DATA_ROOT = os.getenv("DATA_ROOT", "/path/to/data")
EXPERIMENTS_ROOT = os.getenv("EXPERIMENTS_ROOT", "/path/to/experiments")



# =============================================================================
# Setup DataLoader, Model, Loss, Optimizer
# =============================================================================
# =============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = os.path.join(DATA_ROOT, "eyepacs", "train")

train_dataset = datasets.ImageFolder(
    root=data_dir,
    transform=TwoViewTransform(get_pretraining_transform(resize=300, crop_size=256, use_jigsaw=True))
)
train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=os.cpu_count(),
    pin_memory=True
)

# =============================================================================
# Initialize Model, Loss, Optimizer
# =============================================================================
model = ResNet50TripletSelfSup(embedding_dim=128).to(device)
loss_fn = LabeledTripletLoss(device=device, margin=1.0, gamma=1.0)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# =============================================================================
# Training Loop with Encoder Checkpoints Every 10 Epochs
# =============================================================================
num_epochs = 200
checkpoint_dir = os.path.join(EXPERIMENTS_ROOT, "triplet_ssl_checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

for epoch in range(100, num_epochs + 1):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

    for (x_i, x_j), _ in pbar:
        x_i, x_j = x_i.to(device), x_j.to(device)
        inputs = torch.cat([x_i, x_j], dim=0)
        B = x_i.size(0)
        labels = torch.arange(B, device=device).repeat(2)

        optimizer.zero_grad()
        embeddings = model(inputs)
        loss = loss_fn(embeddings, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch}/{num_epochs} — Avg Triplet Loss: {avg_loss:.4f}")

    # Save encoder-only checkpoint every 10 epochs
    if epoch % 10 == 0:
        ckpt_path = os.path.join(checkpoint_dir, f"encoder_epoch_{epoch}.pth")
        torch.save(model.encoder.state_dict(), ckpt_path)
        print(f"Saved encoder checkpoint: {ckpt_path}")

# =============================================================================
# Save Final Full Model Checkpoint
# =============================================================================
final_ckpt_path = os.path.join(checkpoint_dir, "resnet50_triplet_selfsup_final_200.pth")
torch.save(model.state_dict(), final_ckpt_path)
print(f"Saved final self-supervised triplet checkpoint: {final_ckpt_path}")
