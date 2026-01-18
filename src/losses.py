"""
Loss functions for self-supervised learning.

Implements VICReg loss components (variance, invariance, covariance, cross-correlation)
and labeled triplet loss with all-triplet mining strategy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# VICReg Loss Components
# =============================================================================

def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """
    Extract off-diagonal elements from a square matrix.
    
    Args:
        x (torch.Tensor): Square matrix (N, N).
    
    Returns:
        torch.Tensor: Flattened off-diagonal elements.
    """
    n, m = x.shape
    assert n == m, "Input must be a square matrix."
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def var_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 1e-3
) -> torch.Tensor:
    """
    Compute variance regularization loss (VICReg).
    
    Encourages each dimension of the embedding to have sufficient variance.
    Uses ReLU to penalize dimensions with variance < 1.
    
    Args:
        x (torch.Tensor): Embeddings from first view (B, D).
        y (torch.Tensor): Embeddings from second view (B, D).
        epsilon (float): Small value for numerical stability. Default: 1e-3.
    
    Returns:
        torch.Tensor: Scalar variance loss.
    """
    x0 = x - x.mean(dim=0)
    y0 = y - y.mean(dim=0)
    std_x = torch.sqrt(x0.var(dim=0) + epsilon)
    std_y = torch.sqrt(y0.var(dim=0) + epsilon)
    var_l = (torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))) / 2
    return var_l


def invar_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute invariance loss (VICReg).
    
    Minimizes MSE between embeddings of the same image under different augmentations.
    Encourages the model to produce similar representations for augmented views.
    
    Args:
        x (torch.Tensor): Embeddings from first view (B, D).
        y (torch.Tensor): Embeddings from second view (B, D).
    
    Returns:
        torch.Tensor: Scalar invariance loss.
    """
    return F.mse_loss(x, y)


def cov_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute covariance regularization loss (VICReg).
    
    Penalizes correlation between dimensions. Off-diagonal elements of the
    covariance matrix are regularized to zero, encouraging decorrelation.
    
    Args:
        x (torch.Tensor): Embeddings from first view (B, D).
        y (torch.Tensor): Embeddings from second view (B, D).
    
    Returns:
        torch.Tensor: Scalar covariance loss.
    """
    bs = x.size(0)
    emb = x.size(1)
    x1 = x - x.mean(0)
    y1 = y - y.mean(0)
    cov_x = (x1.T @ x1) / (bs - 1)
    cov_y = (y1.T @ y1) / (bs - 1)
    cov_l = off_diagonal(cov_x).pow(2).sum().div(emb) + off_diagonal(cov_y).pow(2).sum().div(emb)
    return cov_l


def cross_corr_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    lmbda: float = 5e-3
) -> torch.Tensor:
    """
    Compute cross-correlation loss between embedding dimensions.
    
    Encourages decorrelation between embeddings from different views
    while maintaining dimensional structure.
    
    Args:
        x (torch.Tensor): Embeddings from first view (B, D).
        y (torch.Tensor): Embeddings from second view (B, D).
        lmbda (float): Scaling factor. Default: 5e-3.
    
    Returns:
        torch.Tensor: Scalar cross-correlation loss.
    """
    bs = x.size(0)
    emb = x.size(1)
    x_norm = (x - x.mean(0)) / (x.std(0) + 1e-8)
    y_norm = (y - y.mean(0)) / (y.std(0) + 1e-8)
    cross_cor_mat = (x_norm.T @ y_norm) / bs
    cross_l = ((cross_cor_mat * lmbda - torch.eye(emb, device=x.device) * lmbda).pow(2)).sum()
    return cross_l


class VICRegLoss(nn.Module):
    """
    VICReg loss: Variance-Invariance-Covariance Regularization.
    
    Combines four loss terms:
    - Variance: encourages sufficient variance in each dimension
    - Invariance: encourages similar embeddings for augmented views
    - Covariance: decorrelates dimensions
    - Cross-correlation: decorrelates cross-view embeddings
    """
    def __init__(
        self,
        lambda_var: float = 25.0,
        lambda_invar: float = 25.0,
        lambda_cov: float = 1.0,
        lambda_cross: float = 5e-3
    ):
        """
        Args:
            lambda_var (float): Weight for variance loss. Default: 25.0.
            lambda_invar (float): Weight for invariance loss. Default: 25.0.
            lambda_cov (float): Weight for covariance loss. Default: 1.0.
            lambda_cross (float): Weight for cross-correlation loss. Default: 5e-3.
        """
        super().__init__()
        self.lambda_var = lambda_var
        self.lambda_invar = lambda_invar
        self.lambda_cov = lambda_cov
        self.lambda_cross = lambda_cross

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute total VICReg loss.
        
        Args:
            z1 (torch.Tensor): Embeddings from first view (B, D).
            z2 (torch.Tensor): Embeddings from second view (B, D).
        
        Returns:
            torch.Tensor: Scalar total loss.
        """
        loss = (
            self.lambda_var * var_loss(z1, z2) +
            self.lambda_invar * invar_loss(z1, z2) +
            self.lambda_cov * cov_loss(z1, z2) +
            self.lambda_cross * cross_corr_loss(z1, z2)
        )
        return loss


# =============================================================================
# Triplet Loss with All-Triplet Mining
# =============================================================================

class LabeledTripletLoss(nn.Module):
    """
    Labeled triplet loss with all-triplet mining strategy.
    
    Mines all valid triplets from a batch based on labels.
    For each anchor-positive pair, considers all negatives with different labels.
    Uses distance-based margin and supports power scaling (gamma).
    
    Reference: Schroff et al., "FaceNet: A Unified Embedding for Face Recognition
    and Clustering" (CVPR 2015).
    """
    def __init__(
        self,
        device: torch.device,
        margin: float = 1.0,
        gamma: float = 1.0
    ):
        """
        Args:
            device (torch.device): GPU/CPU device.
            margin (float): Triplet margin. Default: 1.0.
            gamma (float): Power scaling for distances. Default: 1.0.
        """
        super().__init__()
        self.device = device
        self.margin = margin
        self.gamma = gamma

    def get_distance_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise L2 distances between embeddings.
        
        Uses efficient batch matrix multiplication: d(i,j)^2 = ||z_i||^2 + ||z_j||^2 - 2*z_i^T*z_j
        Since embeddings are L2-normalized, ||z||^2 = 1, so: d(i,j)^2 = 2 - 2*z_i^T*z_j
        
        Args:
            embeddings (torch.Tensor): Normalized embeddings (B, D).
        
        Returns:
            torch.Tensor: Pairwise distance matrix (B, B).
        """
        dot = embeddings @ embeddings.T
        norm = torch.diag(dot)
        dists = norm.view(1, -1) - 2 * dot + norm.view(-1, 1)
        return torch.sqrt(F.relu(dists) + 1e-16)

    def get_triplet_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Generate mask for valid triplets (anchor, positive, negative).
        
        Valid triplets satisfy:
        - anchor != positive (different indices)
        - anchor != negative (different indices)
        - positive != negative (different indices)
        - anchor and positive have same label
        - anchor and negative have different labels
        
        Args:
            labels (torch.Tensor): Class labels (B,).
        
        Returns:
            torch.Tensor: Boolean mask (B, B, B) where True indicates valid triplet.
        """
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

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss over all valid triplets in batch.
        
        Args:
            embeddings (torch.Tensor): Normalized embeddings (B, D).
            labels (torch.Tensor): Class labels (B,).
        
        Returns:
            torch.Tensor: Scalar triplet loss (mean over valid triplets).
        """
        B2 = embeddings.size(0)
        dist_mat = self.get_distance_matrix(embeddings)
        dij = dist_mat.view(B2, B2, 1)
        dik = dist_mat.view(B2, 1, B2)
        loss_un = dij ** self.gamma - dik ** self.gamma + self.margin

        mask = self.get_triplet_mask(labels)
        triplet_losses = F.relu(loss_un[mask])
        if triplet_losses.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        return triplet_losses.mean()
