import os
import glob
import math
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

from skimage import io
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm

warnings.filterwarnings('ignore')

print(f"Torch-Version {torch.__version__}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")

# ---------------------------------------------------------------------
# 1) RandomCropWithFallback
# ---------------------------------------------------------------------
class RandomCropWithFallback:
    def __init__(self, size):
        self.size = size
        self.center_crop = T.CenterCrop(size)
        
    def __call__(self, img):
        width, height = img.size  # PIL image: (width, height)
        if width < self.size or height < self.size:
            return self.center_crop(img)
        left = np.random.randint(0, width - self.size + 1)
        top = np.random.randint(0, height - self.size + 1)
        return img.crop((left, top, left + self.size, top + self.size))

# ---------------------------------------------------------------------
# 2) CLAHETransform
# ---------------------------------------------------------------------
class CLAHETransform:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance image contrast.
    For color images, applies on the L channel in LAB space.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8,8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
    def __call__(self, img):
        np_img = np.array(img)
        if len(np_img.shape) == 3 and np_img.shape[2] == 3:
            lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge((l, a, b))
            img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            img_clahe = self.clahe.apply(np_img)
        return Image.fromarray(img_clahe)

# ---------------------------------------------------------------------
# 3) Multi-Stage Jigsaw Puzzle Transform
# ---------------------------------------------------------------------
def jigsaw_puzzle(np_img, grid_size=4):
    """
    Splits np_img (H, W, 3) into grid_size x grid_size tiles,
    shuffles them, and reassembles into a single scrambled image.
    """
    H, W, C = np_img.shape
    tile_w = W // grid_size
    tile_h = H // grid_size

    tiles = []
    for gx in range(grid_size):
        for gy in range(grid_size):
            left   = gx * tile_w
            right  = (gx + 1) * tile_w
            top    = gy * tile_h
            bottom = (gy + 1) * tile_h
            tile = np_img[top:bottom, left:right, :]
            tiles.append(tile)

    random.shuffle(tiles)

    new_img = np.zeros_like(np_img)
    idx = 0
    for gx in range(grid_size):
        for gy in range(grid_size):
            left   = gx * tile_w
            right  = (gx + 1) * tile_w
            top    = gy * tile_h
            bottom = (gy + 1) * tile_h
            new_img[top:bottom, left:right, :] = tiles[idx]
            idx += 1

    return new_img

class MultiStageRandomChoiceJigsaw:
    """
    Randomly picks one grid size from 'puzzle_sizes' (e.g., [8, 4, 2, 1])
    and applies the jigsaw puzzle transformation. With probability p the transform is applied.
    """
    def __init__(self, puzzle_sizes=[8,4,2,1], p=1.0):
        self.puzzle_sizes = puzzle_sizes
        self.prob = p

    def __call__(self, pil_img):
        if random.random() > self.prob:
            return pil_img
        grid_size = random.choice(self.puzzle_sizes)
        np_img = np.array(pil_img)
        puzzle_np = jigsaw_puzzle(np_img, grid_size=grid_size)
        return Image.fromarray(puzzle_np)

# ---------------------------------------------------------------------
# 4) Full Transform Pipeline (Updated with Multi-Stage Jigsaw)
# ---------------------------------------------------------------------
def get_complete_transform_updated():
    """
    Creates a transformation pipeline for contrastive views.
    The image is resized, enhanced with CLAHE, randomly cropped and flipped,
    jittered in color and grayscale, blurred, and then augmented with a random jigsaw puzzle.
    Finally, the image is converted to a tensor and normalized.
    """
    return T.Compose([
        T.Resize(300),
        CLAHETransform(clip_limit=2.0, tile_grid_size=(8,8)),
        RandomCropWithFallback(256),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(0.4, 0.4, 0.2, 0.1),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=(23,23), sigma=(0.1, 2.0)),
        # Apply multi-stage jigsaw augmentation:
        MultiStageRandomChoiceJigsaw(puzzle_sizes=[8,4,2,1], p=1.0),
        T.ToTensor(),
        T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])

# ---------------------------------------------------------------------
# 5) ContrastiveLearningViewGenerator
# ---------------------------------------------------------------------
class ContrastiveLearningViewGenerator:
    """
    Wraps a base transform to produce 'n_views' augmented versions of the same image.
    """
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.n_views)]

# ---------------------------------------------------------------------
# 6) CustomDataset
# ---------------------------------------------------------------------
class CustomDataset(Dataset):
    """
    Loads images from file paths, applying a transform if provided.
    """
    def __init__(self, list_images, transform=None):
        self.list_images = list_images
        self.transform = transform

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        img_name = self.list_images[idx]
        image = io.imread(img_name)
        if image.dtype != np.uint8:
            if image.max() <= 1:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        pil_img = Image.fromarray(image)
        if self.transform:
            pil_img = self.transform(pil_img)
        return pil_img

# ---------------------------------------------------------------------
# 7) Loss Functions
# ---------------------------------------------------------------------
def off_diagonal(x):
    n, m = x.shape
    assert n == m, "Input must be a square matrix."
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def var_loss(x, y, epsilon=1e-3):
    x0 = x - x.mean(dim=0)
    y0 = y - y.mean(dim=0)
    std_x = torch.sqrt(x0.var(dim=0) + epsilon)
    std_y = torch.sqrt(y0.var(dim=0) + epsilon)
    var_l = (torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))) / 2
    return var_l

def invar_loss(x, y):
    return F.mse_loss(x, y)

def cov_loss(x, y):
    bs = x.size(0)
    emb = x.size(1)
    x1 = x - x.mean(0)
    y1 = y - y.mean(0)
    cov_x = (x1.T @ x1) / (bs - 1)
    cov_y = (y1.T @ y1) / (bs - 1)
    cov_l = off_diagonal(cov_x).pow(2).sum().div(emb) + off_diagonal(cov_y).pow(2).sum().div(emb)
    return cov_l

def cross_corr_loss(x, y, lmbda=5e-3):
    bs = x.size(0)
    emb = x.size(1)
    x_norm = (x - x.mean(0)) / x.std(0)
    y_norm = (y - y.mean(0)) / y.std(0)
    cross_cor_mat = (x_norm.T @ y_norm) / bs
    cross_l = ((cross_cor_mat * lmbda - torch.eye(emb, device=x.device) * lmbda).pow(2)).sum()
    return cross_l

# ---------------------------------------------------------------------
# 8) SiCoVa Model
# ---------------------------------------------------------------------
class SiCoVa(nn.Module):
    """
    Basic model for self-supervised learning:
    - Uses a ResNet50 encoder (pretrained on ImageNet)
    - An expander (3-layer MLP) to map features to high-dimensional embeddings.
    """
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(
            *(list(self.encoder.children())[:-1]),
            nn.Flatten()
        )
        self.expander = nn.Sequential(
            nn.Linear(2048, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Linear(8192, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Linear(8192, 8192)
        )

    def forward(self, x):
        features = self.encoder(x)    # shape: [B, 2048]
        embeds = self.expander(features)  # shape: [B, 8192]
        return embeds

# ---------------------------------------------------------------------
# 9) LARS Optimizer
# ---------------------------------------------------------------------
class LARS(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        learning_rate,  # using 'learning_rate'
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad
                if dp is None:
                    continue
                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])
                if g["lars_adaptation_filter"] is None or not g["lars_adaptation_filter"](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(update_norm > 0, (g["eta"] * param_norm / update_norm), one),
                        one,
                    )
                    dp = dp.mul(q)
                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)
                p.add_(mu, alpha=-g["learning_rate"])

def exclude_bias_and_norm(p):
    return p.ndim == 1

# ---------------------------------------------------------------------
# 10) LR Scheduling
# ---------------------------------------------------------------------
def adjust_learning_rate(optimizer, loader, step):
    total_epoch = 200
    effective_batch_size = 2048
    warmup_steps = 10 * len(loader)
    base_lr = 0.2 * effective_batch_size / 64  # For effective batch size 2048, offset_bs=64 => base_lr=6.4
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

# ---------------------------------------------------------------------
# 11) Training Loop
# ---------------------------------------------------------------------
def train_loop(model, optimizer, trainn_dl, var_loss_fn, invar_loss_fn, cov_loss_fn, cross_corr_loss_fn, device, epoch):
    tk0 = tqdm(trainn_dl, desc=f"Epoch {epoch+101}")
    train_loss = []
    lmbd = 25
    u = 25
    v = 1

    # For batch_size=64, accumulate gradients for 32 steps to get effective batch size ~2048.
    accumulation_steps = 8

    optimizer.zero_grad()
    step_count = 0

    for i, (x, x1) in enumerate(tk0):
        adjust_learning_rate(optimizer, trainn_dl, step_count)
        step_count += 1

        x, x1 = x.to(device), x1.to(device)

        fx  = model(x)
        fx1 = model(x1)

        variance_loss          = var_loss_fn(fx, fx1)
        invariance_loss        = invar_loss_fn(fx, fx1)
        covariance_loss        = cov_loss_fn(fx, fx1)
        cross_correlation_loss = cross_corr_loss_fn(fx, fx1)

        loss = (lmbd * variance_loss +
                u    * invariance_loss +
                v    * covariance_loss +
                cross_correlation_loss)
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

# ---------------------------------------------------------------------
# 12) Main Pretraining
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    # Wrap the base transform to produce 2 contrastive views per sample.
    custom_transform = ContrastiveLearningViewGenerator(
        base_transform=get_complete_transform_updated(),
        n_views=2
    )

    train_image_paths = glob.glob("/home/s13mchop/HybridML/data/eyepacs/train/**/*.jpeg", recursive=True)
    print(f"Number of training samples: {len(train_image_paths)}")

    trainn_ds = CustomDataset(
        list_images=train_image_paths,
        transform=custom_transform
    )

    # DataLoader with batch_size=64.
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
        learning_rate=initial_lr,  # note: using 'learning_rate'
        weight_decay=1e-6,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm
    )
    #Optionally resume from checkpoint at epoch 100 if available.
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
