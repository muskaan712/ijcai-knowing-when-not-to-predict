"""Pretraining script for SiCoVa with jigsaw augmentation."""

import os
import glob
import math
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.pretrain import (
    get_pretrain_transform,
    ContrastiveLearningViewGenerator,
    CustomDataset,
    var_loss,
    invar_loss,
    cov_loss,
    cross_corr_loss,
    SiCoVa,
    LARS,
    exclude_bias_and_norm,
)

warnings.filterwarnings("ignore")

print(f"Torch-Version {torch.__version__}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")


def adjust_learning_rate(optimizer, loader, step):
    """Cosine learning rate schedule with warmup."""

    total_epoch = 200
    effective_batch_size = 2048
    warmup_steps = 10 * len(loader)
    base_lr = 0.2 * effective_batch_size / 64
    max_steps = total_epoch * len(loader)
    if step < warmup_steps:
        lr = 0.2 * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]["learning_rate"] = lr


def train_loop(
    model,
    optimizer,
    trainn_dl,
    var_loss_fn,
    invar_loss_fn,
    cov_loss_fn,
    cross_corr_loss_fn,
    device,
    epoch,
):
    """Run a single training epoch for SiCoVa."""

    tk0 = tqdm(trainn_dl, desc=f"Epoch {epoch+101}")
    train_loss = []
    lmbd = 25
    u = 25
    v = 1
    accumulation_steps = 8
    optimizer.zero_grad()
    step_count = 0
    for i, (x, x1) in enumerate(tk0):
        adjust_learning_rate(optimizer, trainn_dl, step_count)
        step_count += 1
        x, x1 = x.to(device), x1.to(device)
        fx = model(x)
        fx1 = model(x1)
        variance_loss = var_loss_fn(fx, fx1)
        invariance_loss = invar_loss_fn(fx, fx1)
        covariance_loss = cov_loss_fn(fx, fx1)
        cross_correlation_loss = cross_corr_loss_fn(fx, fx1)
        loss = (
            lmbd * variance_loss
            + u * invariance_loss
            + v * covariance_loss
            + cross_correlation_loss
        )
        train_loss.append(loss.item())
        loss = loss / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    if (epoch + 101) % 10 == 0:
        checkpoint_encoder_name = f"SiCoVa_no_expander_{epoch+101}.pt"
        state_dict = model.state_dict()
        encoder_only = {k: v for k, v in state_dict.items() if not k.startswith("expander")}
        torch.save(encoder_only, checkpoint_encoder_name)
        print(f"Encoder-only checkpoint saved at {checkpoint_encoder_name}")
        if (epoch + 101) in [100, 200]:
            checkpoint_name = f"SiCoVa_full_{epoch+101}.pt"
            torch.save(
                {
                    "epoch": epoch + 101,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_name,
            )
            print(f"Checkpoint saved => {checkpoint_name}")
    print(f"Completed Epoch {epoch+101} => Mean Loss: {np.mean(train_loss):.5f}")


if __name__ == "__main__":
    custom_transform = ContrastiveLearningViewGenerator(
        base_transform=get_pretrain_transform(use_jigsaw=True),
        n_views=2
    )
    train_image_paths = glob.glob("/home/s13mchop/HybridML/data/eyepacs/train/**/*.jpeg", recursive=True)
    print(f"Number of training samples: {len(train_image_paths)}")
    trainn_ds = CustomDataset(
        list_images=train_image_paths,
        transform=custom_transform
    )
    trainn_dl = DataLoader(
        trainn_ds,
        batch_size=256,
        shuffle=True,
        num_workers=os.cpu_count(),
        drop_last=True,
        pin_memory=True
    )
    model = SiCoVa().to(DEVICE)
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
    checkpoint_path = "/home/s13mchop/HybridML/experiments/pretrain/VR1_CLAHE_Jigsaw/SiCoVa_full_100.pt"
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
            var_loss_fn=var_loss,
            invar_loss_fn=invar_loss,
            cov_loss_fn=cov_loss,
            cross_corr_loss_fn=cross_corr_loss,
            device=DEVICE,
            epoch=epoch
        )
