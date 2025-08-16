"""Shared utilities and model definitions for pretraining scripts."""

import random
from typing import List

import cv2
import numpy as np
from PIL import Image
from skimage import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset


class RandomCropWithFallback:
    """Randomly crop an image, falling back to center crop if too small."""

    def __init__(self, size: int):
        """Store crop size and create a center-crop transform."""
        self.size = size
        self.center_crop = T.CenterCrop(size)

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply the random crop or use center crop for small images."""
        width, height = img.size
        if width < self.size or height < self.size:
            return self.center_crop(img)
        left = np.random.randint(0, width - self.size + 1)
        top = np.random.randint(0, height - self.size + 1)
        return img.crop((left, top, left + self.size, top + self.size))


class CLAHETransform:
    """Contrast-limited adaptive histogram equalization."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size=(8, 8)):
        """Initialize the CLAHE operator."""
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply CLAHE to an RGB or grayscale image."""
        np_img = np.array(img)
        if np_img.ndim == 3 and np_img.shape[2] == 3:
            lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge((l, a, b))
            img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            img_clahe = self.clahe.apply(np_img)
        return Image.fromarray(img_clahe)


def jigsaw_puzzle(np_img: np.ndarray, grid_size: int = 4) -> np.ndarray:
    """Shuffle tiles of an image to form a jigsaw puzzle."""

    H, W, C = np_img.shape
    tile_w, tile_h = W // grid_size, H // grid_size
    tiles: List[np.ndarray] = []
    for i in range(grid_size):
        for j in range(grid_size):
            left, top = i * tile_w, j * tile_h
            tiles.append(np_img[top:top + tile_h, left:left + tile_w, :])
    random.shuffle(tiles)
    new_img = np.zeros_like(np_img)
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            left, top = i * tile_w, j * tile_h
            new_img[top:top + tile_h, left:left + tile_w, :] = tiles[idx]
            idx += 1
    return new_img


class MultiStageRandomChoiceJigsaw:
    """Apply jigsaw augmentations with varying grid sizes."""

    def __init__(self, puzzle_sizes=None, p: float = 1.0):
        """Initialize with possible ``puzzle_sizes`` and probability ``p``."""
        self.puzzle_sizes = puzzle_sizes or [8, 4, 2, 1]
        self.prob = p

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        """Apply a random jigsaw puzzle transformation."""
        if random.random() > self.prob:
            return pil_img
        grid_size = random.choice(self.puzzle_sizes)
        np_img = np.array(pil_img)
        puzzle_np = jigsaw_puzzle(np_img, grid_size=grid_size)
        return Image.fromarray(puzzle_np)


def get_pretrain_transform(use_jigsaw: bool = False) -> T.Compose:
    """Build the image augmentation pipeline for pretraining."""

    base = [
        T.Resize(300),
        CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
        RandomCropWithFallback(256),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(0.4, 0.4, 0.2, 0.1),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=(23, 23), sigma=(0.1, 2.0)),
    ]
    if use_jigsaw:
        base.append(MultiStageRandomChoiceJigsaw(puzzle_sizes=[8, 4, 2, 1], p=1.0))
    base.extend([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return T.Compose(base)


class ContrastiveLearningViewGenerator:
    """Generate multiple augmented views for contrastive learning."""

    def __init__(self, base_transform, n_views: int = 2):
        """Store base transform and number of views."""
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        """Return a list with ``n_views`` transformed versions of ``x``."""
        return [self.base_transform(x) for _ in range(self.n_views)]


class TwoViewTransform:
    """Return two independent views of an image."""

    def __init__(self, base_transform):
        """Store the transform applied to both views."""
        self.base_transform = base_transform

    def __call__(self, img):
        """Return a tuple with two transformed images."""
        return self.base_transform(img), self.base_transform(img)


class CustomDataset(Dataset):
    """Dataset that loads images from file paths."""

    def __init__(self, list_images, transform=None):
        """Initialize with a list of image paths and an optional transform."""
        self.list_images = list_images
        self.transform = transform

    def __len__(self):
        """Return the dataset length."""
        return len(self.list_images)

    def __getitem__(self, idx):
        """Load and transform an image by index."""
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


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Return flattened off-diagonal elements of a square matrix."""

    n, m = x.shape
    assert n == m, "Input must be a square matrix."
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def var_loss(x: torch.Tensor, y: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    """Variance regularization encouraging non-collapse."""

    x0 = x - x.mean(dim=0)
    y0 = y - y.mean(dim=0)
    std_x = torch.sqrt(x0.var(dim=0) + epsilon)
    std_y = torch.sqrt(y0.var(dim=0) + epsilon)
    return (torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))) / 2


def invar_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Invariance loss measuring similarity between views."""

    return F.mse_loss(x, y)


def cov_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cross-covariance loss to encourage decorrelated features."""

    bs = x.size(0)
    emb = x.size(1)
    x1 = x - x.mean(0)
    y1 = y - y.mean(0)
    cov_x = (x1.T @ x1) / (bs - 1)
    cov_y = (y1.T @ y1) / (bs - 1)
    cov_l = off_diagonal(cov_x).pow(2).sum().div(emb) + off_diagonal(cov_y).pow(2).sum().div(emb)
    return cov_l


def cross_corr_loss(x: torch.Tensor, y: torch.Tensor, lmbda: float = 5e-3) -> torch.Tensor:
    """Cross-correlation penalty inspired by Barlow Twins."""

    bs = x.size(0)
    emb = x.size(1)
    x_norm = (x - x.mean(0)) / x.std(0)
    y_norm = (y - y.mean(0)) / y.std(0)
    cross_cor_mat = (x_norm.T @ y_norm) / bs
    cross_l = ((cross_cor_mat * lmbda - torch.eye(emb, device=x.device) * lmbda).pow(2)).sum()
    return cross_l


class SiCoVa(nn.Module):
    """ResNet50 encoder with expander MLP for SiCoVa training."""

    def __init__(self):
        """Load pretrained ResNet50 and append expander head."""
        super().__init__()
        base = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*(list(base.children())[:-1]), nn.Flatten())
        self.expander = nn.Sequential(
            nn.Linear(2048, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Linear(8192, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Linear(8192, 8192),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return embedding vectors for input batch ``x``."""
        features = self.encoder(x)
        embeds = self.expander(features)
        return embeds


class LARS(torch.optim.Optimizer):
    """Layer-wise adaptive rate scaling optimizer."""

    def __init__(
        self,
        params,
        learning_rate,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        """Initialize optimizer with LARS hyperparameters."""

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
        """Apply one optimization step."""
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
    """Check if parameter ``p`` is a bias or norm parameter."""

    return p.ndim == 1


class ResNet50TripletSelfSup(nn.Module):
    """ResNet50 encoder for self-supervised triplet learning."""

    def __init__(self, embedding_dim: int = 128):
        """Initialize encoder and projection head."""
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return normalized embeddings for inputs."""
        h = self.encoder(x)
        h = self.flatten(h)
        z = self.fc(h)
        return F.normalize(z, dim=1)


class LabeledTripletLoss(nn.Module):
    """Triplet loss variant that uses labels to form positives/negatives."""

    def __init__(self, device: torch.device, margin: float = 1.0, gamma: float = 1.0):
        """Configure loss with margin and exponent ``gamma``."""
        super().__init__()
        self.device = device
        self.margin = margin
        self.gamma = gamma

    def get_distance_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distances between embeddings."""
        dot = embeddings @ embeddings.T
        norm = torch.diag(dot)
        dists = norm.view(1, -1) - 2 * dot + norm.view(-1, 1)
        return torch.sqrt(F.relu(dists) + 1e-16)

    def get_triplet_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """Return mask for valid triplets based on labels."""
        B = labels.size(0)
        idx_eq = torch.eye(B, device=self.device).bool()
        neq = ~idx_eq
        i_ne_j = neq.view(B, B, 1)
        i_ne_k = neq.view(B, 1, B)
        j_ne_k = neq.view(1, B, B)
        distinct = i_ne_j & i_ne_k & j_ne_k
        lbl_eq = labels.view(1, B) == labels.view(B, 1)
        i_eq_j = lbl_eq.view(B, B, 1)
        i_ne_k2 = (~lbl_eq).view(B, 1, B)
        valid = i_eq_j & i_ne_k2
        return distinct & valid

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute labeled triplet loss for a batch of embeddings."""
        B = embeddings.size(0)
        dist_mat = self.get_distance_matrix(embeddings)
        dij = dist_mat.view(B, B, 1)
        dik = dist_mat.view(B, 1, B)
        loss_un = dij ** self.gamma - dik ** self.gamma + self.margin
        mask = self.get_triplet_mask(labels)
        triplet_losses = F.relu(loss_un[mask])
        if triplet_losses.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        return triplet_losses.mean()
