"""Self-supervised triplet pretraining without jigsaw augmentation."""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from tqdm import tqdm

from utils.pretrain import (
    get_pretrain_transform,
    TwoViewTransform,
    ResNet50TripletSelfSup,
    LabeledTripletLoss,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "/hpcwork/ni124545/data/eyepacs/train/"
train_dataset = datasets.ImageFolder(
    root=data_dir,
    transform=TwoViewTransform(get_pretrain_transform(use_jigsaw=False))
)
train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=os.cpu_count(),
    pin_memory=True
)

model = ResNet50TripletSelfSup(embedding_dim=128).to(device)
loss_fn = LabeledTripletLoss(device=device, margin=1.0, gamma=1.0)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

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
    if epoch % 10 == 0:
        os.makedirs("/hpcwork/ni124545/triplet_ssl_checkpoints", exist_ok=True)
        ckpt_path = f"/hpcwork/ni124545/triplet_ssl_checkpoints/encoder_epoch_{epoch}.pth"
        torch.save(model.encoder.state_dict(), ckpt_path)
        print(f"Saved encoder checkpoint: {ckpt_path}")

torch.save(model.state_dict(), "resnet50_triplet_selfsup_final_200.pth")
print("Saved final self-supervised triplet checkpoint.")
