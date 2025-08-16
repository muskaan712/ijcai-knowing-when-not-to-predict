import os
import random
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as T
import torchvision.datasets as datasets
import torchvision.models as models

from torch.utils.data import DataLoader
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1) RandomCropWithFallback
# -----------------------------------------------------------------------------
class RandomCropWithFallback:
    def __init__(self, size):
        self.size = size
        self.center_crop = T.CenterCrop(size)
        
    def __call__(self, img):
        width, height = img.size
        if width < self.size or height < self.size:
            return self.center_crop(img)
        left = np.random.randint(0, width - self.size + 1)
        top = np.random.randint(0, height - self.size + 1)
        return img.crop((left, top, left + self.size, top + self.size))

# -----------------------------------------------------------------------------
# 2) CLAHETransform
# -----------------------------------------------------------------------------
class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8,8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
    def __call__(self, img):
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

# -----------------------------------------------------------------------------
# 3) Full Transform Pipeline (without jigsaw)
# -----------------------------------------------------------------------------
def get_complete_transform_updated():
    return T.Compose([
        T.Resize(300),
        CLAHETransform(clip_limit=2.0, tile_grid_size=(8,8)),
        RandomCropWithFallback(256),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(0.4, 0.4, 0.2, 0.1),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=(23,23), sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])

# -----------------------------------------------------------------------------
# 4) Two-View Wrapper
# -----------------------------------------------------------------------------
class TwoViewTransform:
    """Generate two independent augmentations of the same image."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, img):
        return self.base_transform(img), self.base_transform(img)

# -----------------------------------------------------------------------------
# 5) Model: ResNet50 + Triplet Embedding Head
# -----------------------------------------------------------------------------
class ResNet50TripletSelfSup(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, embedding_dim)

    def forward(self, x):
        h = self.encoder(x)
        h = self.flatten(h)
        z = self.fc(h)
        return F.normalize(z, dim=1)

# -----------------------------------------------------------------------------
# 6) Triplet Loss with All-Triplet Mining
# -----------------------------------------------------------------------------
class LabeledTripletLoss(nn.Module):
    def __init__(self, device: torch.device, margin: float = 1.0, gamma: float = 1.0):
        super().__init__()
        self.device = device
        self.margin = margin
        self.gamma = gamma

    def get_distance_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        dot = embeddings @ embeddings.T
        norm = torch.diag(dot)
        dists = norm.view(1, -1) - 2*dot + norm.view(-1, 1)
        return torch.sqrt(F.relu(dists) + 1e-16)

    def get_triplet_mask(self, labels: torch.Tensor) -> torch.Tensor:
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
        B2 = embeddings.size(0)
        dist_mat = self.get_distance_matrix(embeddings)
        dij = dist_mat.view(B2, B2, 1)
        dik = dist_mat.view(B2, 1, B2)
        loss_un = dij**self.gamma - dik**self.gamma + self.margin

        mask = self.get_triplet_mask(labels)
        triplet_losses = F.relu(loss_un[mask])
        if triplet_losses.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        return triplet_losses.mean()

# -----------------------------------------------------------------------------
# 7) Setup DataLoader
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "/hpcwork/ni124545/data/eyepacs/train/"

train_dataset = datasets.ImageFolder(
    root=data_dir,
    transform=TwoViewTransform(get_complete_transform_updated())
)
train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=os.cpu_count(),
    pin_memory=True
)

# -----------------------------------------------------------------------------
# 8) Initialize Model, Loss, Optimizer
# -----------------------------------------------------------------------------
model = ResNet50TripletSelfSup(embedding_dim=128).to(device)
loss_fn = LabeledTripletLoss(device=device, margin=1.0, gamma=1.0)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# -----------------------------------------------------------------------------
# 9) Training Loop with Encoder Checkpoints Every 10 Epochs
# -----------------------------------------------------------------------------
num_epochs = 200

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
        os.makedirs("/hpcwork/ni124545/triplet_ssl_wojigsaw_checkpoints", exist_ok=True)
        ckpt_path = f"/hpcwork/ni124545/triplet_ssl_wojigsaw_checkpoints/encoder_epoch_{epoch}.pth"
        torch.save(model.encoder.state_dict(), ckpt_path)
        print(f"Saved encoder checkpoint: {ckpt_path}")

# -----------------------------------------------------------------------------
# 10) Save Final Full Model Checkpoint
# -----------------------------------------------------------------------------
torch.save(model.state_dict(), "resnet50_triplet_selfsup_final_200.pth")
print("Saved final self-supervised triplet checkpoint.")
