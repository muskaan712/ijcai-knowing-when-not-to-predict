"""
Self-supervised pretraining using VICReg loss with multi-stage jigsaw augmentation.

High-level overview:
1) Load unlabeled medical images from disk.
2) Apply multi-stage contrastive augmentations (CLAHE, jigsaw puzzle, color jittering, etc.).
3) Train ResNet50 encoder with VICReg loss (variance, invariance, covariance regularization).
4) Save checkpoints at regular intervals for downstream fine-tuning.
"""
import os
import glob
import math
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import SiCoVa
from src.transforms import ContrastiveLearningViewGenerator, get_pretraining_transform
from src.datasets import CustomDataset
from src.losses import VICRegLoss
from src.optimizers import LARS, adjust_learning_rate, exclude_bias_and_norm

warnings.filterwarnings('ignore')

print(f"Torch-Version {torch.__version__}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")

# Anonymous paths - replace with your data directories
DATA_ROOT = os.getenv("DATA_ROOT", "/path/to/data")
EXPERIMENTS_ROOT = os.getenv("EXPERIMENTS_ROOT", "/path/to/experiments")


# =============================================================================
# Training Loop
# =============================================================================

def train_loop(model, optimizer, trainn_dl, loss_fn, device, epoch):
    """
    Execute one training epoch with gradient accumulation and checkpoint saving.
    
    Computes VICReg loss, accumulates gradients over 8 batches for 
    effective batch size ~2048, and saves checkpoints every 10 epochs.
    
    Args:
        model (nn.Module): SiCoVa model to train.
        optimizer (torch.optim.Optimizer): LARS optimizer.
        trainn_dl (DataLoader): Training data loader.
        loss_fn (VICRegLoss): VICReg loss function.
        device (torch.device): GPU/CPU device.
        epoch (int): Current epoch number (0-indexed).
    """
    tk0 = tqdm(trainn_dl, desc=f"Epoch {epoch+101}")
    train_loss = []

    # For batch_size=256, accumulate gradients for 8 steps to get effective batch size ~2048.
    accumulation_steps = 8

    optimizer.zero_grad()
    step_count = 0

    for i, (x, x1) in enumerate(tk0):
        adjust_learning_rate(optimizer, trainn_dl, step_count, 
                           total_epochs=200, warmup_epochs=10, base_lr=6.4)
        step_count += 1

        x, x1 = x.to(device), x1.to(device)

        fx = model(x)
        fx1 = model(x1)

        loss = loss_fn(fx, fx1)
        train_loss.append(loss.item())

        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Save checkpoints every 10 epochs without the expander.
    if (epoch + 101) % 10 == 0:
        # Save encoder-only checkpoint
        checkpoint_encoder_name = f"SiCoVa_no_expander_{epoch+101}.pt"
        state_dict = model.state_dict()
        encoder_only = {k: v for k, v in state_dict.items() if not k.startswith("expander")}
        torch.save(encoder_only, checkpoint_encoder_name)
        print(f"Encoder-only checkpoint saved at {checkpoint_encoder_name}")

        # Additionally, save the full checkpoint at epochs 100 and 200.
        if (epoch + 101) in [100, 200]:
            checkpoint_name = f"SiCoVa_full_{epoch+101}.pt"
            torch.save({
                'epoch': epoch + 101,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_name)
            print(f"Checkpoint saved => {checkpoint_name}")

    print(f"Completed Epoch {epoch+101} => Mean Loss: {np.mean(train_loss):.5f}")

# =============================================================
# 12) Main Pretraining
# =============================================================
if __name__ == "__main__":
    # Wrap the base transform to produce 2 contrastive views per sample.
    custom_transform = ContrastiveLearningViewGenerator(
        base_transform=get_pretraining_transform(resize=300, crop_size=256, use_jigsaw=True),
        n_views=2
    )

    train_image_paths = glob.glob(
        os.path.join(DATA_ROOT, "eyepacs", "train", "**", "*.jpeg"),
        recursive=True
    )
    print(f"Number of training samples: {len(train_image_paths)}")

    trainn_ds = CustomDataset(
        list_images=train_image_paths,
        transform=custom_transform
    )

    # DataLoader with batch_size=256.
    trainn_dl = DataLoader(
        trainn_ds,
        batch_size=256,
        shuffle=True,
        num_workers=os.cpu_count(),
        drop_last=True,
        pin_memory=True
    )

    # Initialize model.
    model = SiCoVa().to(DEVICE)

    # Using an initial learning rate that gives an effective LR of 3.2
    batch_size = 2048
    offset_bs = 256
    base_lr = 0.1
    initial_lr = base_lr * batch_size / offset_bs
    optimizer = LARS(
        model.parameters(),
        learning_rate=initial_lr,
        weight_decay=1e-6,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm
    )
    
    # Loss function
    loss_fn = VICRegLoss(lambda_var=25.0, lambda_invar=25.0, lambda_cov=1.0, lambda_cross=5e-3)
    
    # Optionally resume from checkpoint at epoch 100 if available.
    checkpoint_path = os.path.join(
        EXPERIMENTS_ROOT,
        "pretrain",
        "VR1_CLAHE_Jigsaw",
        "SiCoVa_full_100.pt"
    )
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("No checkpoint found, starting from scratch.")
        
    epochs = 100
    for epoch in range(epochs):
        train_loop(
            model=model,
            optimizer=optimizer,
            trainn_dl=trainn_dl,
            loss_fn=loss_fn,
            device=DEVICE,
            epoch=epoch
        )
