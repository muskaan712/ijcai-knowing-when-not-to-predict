"""
Model architectures for self-supervised pretraining and downstream fine-tuning.

Implements ResNet50-based encoders for VICReg and Triplet loss training,
as well as downstream classifiers with optional CAM refinement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# =============================================================================
# VICReg Models
# =============================================================================

class SiCoVa(nn.Module):
    """
    Self-supervised learning model with ResNet50 encoder and MLP expander.
    
    Implements VICReg (Variance-Invariance-Covariance Regularization).
    Uses ImageNet-pretrained ResNet50 backbone for feature extraction,
    followed by a 3-layer MLP to map to high-dimensional embeddings
    for VICReg loss computation.
    
    Reference: Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization
    for Self-Supervised Learning" (ICLR 2022).
    """
    def __init__(self, embedding_dim: int = 8192):
        """
        Args:
            embedding_dim (int): Dimension of output embeddings. Default: 8192.
        """
        super().__init__()
        self.encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
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
            nn.Linear(8192, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input images (B, 3, H, W).
        
        Returns:
            torch.Tensor: Embeddings (B, embedding_dim).
        """
        features = self.encoder(x)
        embeds = self.expander(features)
        return embeds


class VICRegNet(nn.Module):
    """
    SSL-pretrained ResNet50 backbone with feature extraction.
    
    Extracts feature maps, pooled representations, and expanded embeddings.
    Used as feature extractor for downstream classification task.
    """
    def __init__(self, embedding_dim: int = 8192):
        """
        Args:
            embedding_dim (int): Dimension of embeddings. Default: 8192.
        """
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # MLP expander - kept for checkpoint compatibility
        self.expander = nn.Sequential(
            nn.Linear(2048, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x (torch.Tensor): Input images (B, 3, H, W).
        
        Returns:
            tuple: (feature_maps, pooled_features, expanded_embeddings)
        """
        fmap = self.backbone(x)
        pooled = self.avgpool(fmap).view(x.size(0), -1)
        emb = self.expander(pooled)
        return fmap, pooled, emb


class LinearClassifier(nn.Module):
    """
    Simple linear classifier head on ResNet50 pooled features.
    
    Maps pooled backbone features (2048-d) to NUM_CLASSES predictions
    for downstream DR classification task.
    """
    def __init__(self, backbone: nn.Module, num_classes: int = 5):
        """
        Args:
            backbone (nn.Module): Pretrained VICRegNet backbone.
            num_classes (int): Number of output classes. Default: 5.
        """
        super().__init__()
        self.backbone = backbone
        self.cls = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input images (B, 3, H, W).
        
        Returns:
            torch.Tensor: Logits (B, num_classes).
        """
        _, pooled, _ = self.backbone(x)
        logits = self.cls(pooled)
        return logits


# =============================================================================
# Triplet Loss Models
# =============================================================================

class ResNet50TripletSelfSup(nn.Module):
    """
    ResNet50 encoder trained with triplet loss for self-supervised learning.
    
    Uses ImageNet-pretrained ResNet50 for feature extraction,
    followed by a linear projection to embedding space and L2 normalization.
    """
    def __init__(self, embedding_dim: int = 128):
        """
        Args:
            embedding_dim (int): Dimension of normalized embeddings. Default: 128.
        """
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input images (B, 3, H, W).
        
        Returns:
            torch.Tensor: L2-normalized embeddings (B, embedding_dim).
        """
        h = self.encoder(x)
        h = self.flatten(h)
        z = self.fc(h)
        return F.normalize(z, dim=1)


# =============================================================================
# CAM-based Refinement Modules
# =============================================================================

class CAMExtractor(nn.Module):
    """
    Extract Class Activation Map (CAM) from feature maps.
    
    Applies a 1x1 convolution to generate channel-wise attention,
    then normalizes per sample for interpretability.
    """
    def __init__(self, in_ch: int):
        """
        Args:
            in_ch (int): Number of input channels (feature map channels).
        """
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, bias=False)

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_map (torch.Tensor): Feature maps (B, C, H, W).
        
        Returns:
            torch.Tensor: Normalized CAM (B, H, W).
        """
        cam = F.relu(self.conv(feat_map)).squeeze(1)
        B, H, W = cam.shape
        flat = cam.view(B, -1)
        mn, mx = flat.min(1, True)[0], flat.max(1, True)[0] + 1e-5
        return ((flat - mn) / (mx - mn)).view(B, H, W)


class RefinementCAM(nn.Module):
    """
    Refine CAM using multi-threshold masking and self-attention.
    
    Creates soft masks at multiple confidence thresholds, applies spatial
    attention to feature maps, and regularizes refined CAM with L1 loss.
    """
    def __init__(self, thresholds: tuple = (0.3, 0.4, 0.5)):
        """
        Args:
            thresholds (tuple): CAM confidence thresholds for multi-scale masking. 
                Default: (0.3, 0.4, 0.5).
        """
        super().__init__()
        self.thresholds = thresholds

    def forward(
        self,
        cam: torch.Tensor,
        feat: torch.Tensor
    ) -> tuple:
        """
        Args:
            cam (torch.Tensor): Input CAM (B, H, W).
            feat (torch.Tensor): Feature maps (B, C, H, W).
        
        Returns:
            tuple: (refined_cam, refinement_loss).
        """
        masks = [(cam >= t).float() for t in self.thresholds]
        m = torch.stack(masks, 1).mean(1).unsqueeze(1)
        if m.shape[-2:] != feat.shape[-2:]:
            raise RuntimeError("CAM/feat map size mismatch")
        masked = feat * m
        ref = self.self_att(cam, masked)
        loss = F.l1_loss(ref, cam.detach())
        return ref, loss

    @staticmethod
    def self_att(cam: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention over feature maps weighted by CAM.
        
        Computes pairwise feature similarity and aggregates using CAM as weights.
        
        Args:
            cam (torch.Tensor): CAM (B, H, W).
            feat (torch.Tensor): Masked features (B, C, H, W).
        
        Returns:
            torch.Tensor: Refined CAM (B, H, W).
        """
        B, C, H, W = feat.shape
        f = feat.view(B, C, -1)
        fn = F.normalize(f, dim=1)
        sim = torch.bmm(fn.transpose(1, 2), fn)
        cf = cam.view(B, -1, 1)
        out = torch.bmm(sim, cf).squeeze(-1)
        mn, mx = out.min(1, True)[0], out.max(1, True)[0] + 1e-5
        return ((out - mn) / (mx - mn)).view(B, H, W)


# =============================================================================
# Downstream Fine-tuning Models
# =============================================================================

class CAMFinetuneVICReg(nn.Module):
    """
    Downstream classifier on top of VICReg-pretrained backbone with CAM.
    
    Loads VICReg SSL backbone, adds classification head, and includes
    CAM refinement loss for interpretability and feature localization.
    """
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 5,
        alpha: float = 0.1
    ):
        """
        Args:
            backbone (nn.Module): Pretrained VICRegNet backbone.
            num_classes (int): Number of output classes. Default: 5.
            alpha (float): Weight for CAM refinement loss. Default: 0.1.
        """
        super().__init__()
        self.backbone = backbone
        self.alpha = alpha
        self.cls_head = nn.Linear(2048, num_classes)
        self.cam_ext = CAMExtractor(in_ch=2048)
        self.refiner = RefinementCAM()

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x (torch.Tensor): Input images (B, 3, H, W).
        
        Returns:
            tuple: (logits, cam, refinement_loss).
        """
        feat_map, pooled, _ = self.backbone(x)
        logits = self.cls_head(pooled)
        cam0 = self.cam_ext(feat_map)
        cam, lr_loss = self.refiner(cam0, feat_map)
        return logits, cam, lr_loss


class CAMFinetuneTriplet(nn.Module):
    """
    Downstream classifier on top of triplet-pretrained embedding with CAM.
    
    Loads triplet SSL encoder, adds classification head, and includes
    CAM refinement loss for interpretability and feature localization.
    """
    def __init__(
        self,
        encoder_ckpt_path: str,
        embedding_dim: int = 128,
        num_classes: int = 5,
        alpha: float = 0.1
    ):
        """
        Args:
            encoder_ckpt_path (str): Path to triplet SSL encoder checkpoint.
            embedding_dim (int): Embedding dimension. Default: 128.
            num_classes (int): Number of output classes. Default: 5.
            alpha (float): Weight for CAM refinement loss. Default: 0.1.
        """
        super().__init__()
        # Load triplet-trained backbone
        self.backbone = ResNet50TripletSelfSup(embedding_dim=embedding_dim)
        state = torch.load(encoder_ckpt_path, map_location="cpu")
        self.backbone.encoder.load_state_dict(state, strict=True)
        self.alpha = alpha
        
        # Classification head on embedding
        self.cls_head = nn.Linear(embedding_dim, num_classes)
        
        # CAM & refinement
        self.cam_ext = CAMExtractor(in_ch=2048)
        self.refiner = RefinementCAM()

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x (torch.Tensor): Input images (B, 3, 224, 224).
        
        Returns:
            tuple: (logits, cam, refinement_loss).
        """
        # Extract feature maps before embedding projection
        feat_map = self.backbone.encoder[:-1](x)
        pooled = self.backbone.encoder[-1](feat_map)
        h = self.backbone.flatten(pooled)
        emb = self.backbone.fc(h)
        
        logits = self.cls_head(emb)
        cam0 = self.cam_ext(feat_map)
        cam, lr_loss = self.refiner(cam0, feat_map)
        return logits, cam, lr_loss


class ABSFinetuneVICReg(nn.Module):
    """
    Downstream classifier with confidence-based abstention (selective prediction).
    
    Fine-tunes VICReg backbone on labeled data with temperature scaling
    calibration for uncertainty estimation and selective prediction.
    """
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 5
    ):
        """
        Args:
            backbone (nn.Module): Pretrained VICRegNet backbone.
            num_classes (int): Number of output classes. Default: 5.
        """
        super().__init__()
        self.backbone = backbone
        self.cls = nn.Linear(2048, num_classes)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input images (B, 3, H, W).
        
        Returns:
            torch.Tensor: Logits (B, num_classes).
        """
        _, pooled, _ = self.backbone(x)
        logits = self.cls(pooled)
        return logits
    
    def get_calibrated_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Get temperature-calibrated probabilities.
        
        Args:
            logits (torch.Tensor): Model logits.
        
        Returns:
            torch.Tensor: Calibrated probabilities.
        """
        return F.softmax(logits / self.temperature, dim=1)


class ABSFinetuneTriplet(nn.Module):
    """
    Triplet-SSL downstream classifier with confidence-based abstention.
    
    Fine-tunes triplet pretrained encoder on labeled data with
    temperature scaling for calibrated uncertainty.
    """
    def __init__(
        self,
        encoder_ckpt_path: str,
        embedding_dim: int = 128,
        num_classes: int = 5
    ):
        """
        Args:
            encoder_ckpt_path (str): Path to triplet SSL encoder checkpoint.
            embedding_dim (int): Embedding dimension. Default: 128.
            num_classes (int): Number of output classes. Default: 5.
        """
        super().__init__()
        self.backbone = ResNet50TripletSelfSup(embedding_dim=embedding_dim)
        state = torch.load(encoder_ckpt_path, map_location="cpu")
        self.backbone.encoder.load_state_dict(state, strict=True)
        
        self.cls_head = nn.Linear(embedding_dim, num_classes)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input images (B, 3, 224, 224).
        
        Returns:
            torch.Tensor: Logits (B, num_classes).
        """
        emb = self.backbone(x)
        logits = self.cls_head(emb)
        return logits
    
    def get_calibrated_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Get temperature-calibrated probabilities.
        
        Args:
            logits (torch.Tensor): Model logits.
        
        Returns:
            torch.Tensor: Calibrated probabilities.
        """
        return F.softmax(logits / self.temperature, dim=1)
