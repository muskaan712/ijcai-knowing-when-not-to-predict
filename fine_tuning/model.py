"""Model components used for downstream fine-tuning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CAMExtractor(nn.Module):
    """Generate class activation maps from feature maps."""

    def __init__(self, in_channels: int = 2048):
        """Create a 1x1 convolution for CAM extraction."""
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1, bias=False)

    def forward(self, feat_map):
        """Produce normalized activation maps."""
        cam = F.relu(self.conv(feat_map)).squeeze(1)
        B, H, W = cam.shape
        flat = cam.view(B, -1)
        cam = (flat - flat.min(1, keepdim=True)[0]) / (flat.max(1, keepdim=True)[0] + 1e-5)
        return cam.view(B, H, W)


class RefinementCAM(nn.Module):
    """Refine CAMs via self-attention with thresholded masks."""

    def __init__(self, thresholds=(0.3, 0.4, 0.5)):
        """Store thresholds used to create masks."""
        super().__init__()
        self.thresholds = thresholds

    def forward(self, cam, feat):
        """Refine CAM using self-attention and compute refinement loss."""
        masks = [(cam >= t).float() for t in self.thresholds]
        mask = torch.stack(masks, 1).mean(1).unsqueeze(1)
        if mask.shape[-2:] != feat.shape[-2:]:
            raise RuntimeError("CAM/feature size mismatch, check extractor.")
        masked = feat * mask
        refined = self.self_attention(cam, masked)
        loss = F.l1_loss(refined, cam.detach())
        return refined, loss

    @staticmethod
    def self_attention(cam, feat):
        """Perform self-attention weighting of features by CAM."""
        B, C, H, W = feat.shape
        feat_flat = F.normalize(feat.view(B, C, -1), dim=1)
        sim = torch.bmm(feat_flat.transpose(1, 2), feat_flat)
        cam_flat = cam.view(B, -1, 1)
        refined = torch.bmm(sim, cam_flat).squeeze(-1)
        refined = (refined - refined.min(1, keepdim=True)[0]) / (
            refined.max(1, keepdim=True)[0] + 1e-5
        )
        return refined.view(B, H, W)


class VICRegNet(nn.Module):
    """ResNet backbone with expander head for VICReg-style training."""

    def __init__(self):
        """Load pretrained backbone and build expander MLP."""
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
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
        """Return feature map, pooled features and expanded embedding."""
        fmap = self.backbone(x)
        pooled = self.avgpool(fmap).view(x.size(0), -1)
        embeds = self.expander(pooled)
        return fmap, pooled, embeds


class CAMClassification(nn.Module):
    """Classifier with CAM refinement regularization."""

    def __init__(self, backbone, num_classes: int = 5, alpha: float = 0.1):
        """Wrap a backbone with classification and CAM components."""
        super().__init__()
        self.backbone = backbone
        self.cls_head = nn.Linear(2048, num_classes)
        self.cam_extractor = CAMExtractor(2048)
        self.refiner = RefinementCAM()
        self.alpha = alpha

    def forward(self, x):
        """Return logits, refined CAM and refinement loss."""
        fmap, pooled, _ = self.backbone(x)
        logits = self.cls_head(pooled)
        cam0 = self.cam_extractor(fmap)
        cam, loss_ref = self.refiner(cam0, fmap)
        return logits, cam, loss_ref
