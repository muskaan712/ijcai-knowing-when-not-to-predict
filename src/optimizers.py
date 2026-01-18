"""
Optimizers and learning rate scheduling for self-supervised learning.

Implements LARS optimizer for large-batch training and cosine annealing
learning rate schedules with warmup.
"""

import math
import torch
import torch.optim as optim


class LARS(optim.Optimizer):
    """
    Layer-wise Adaptive Rate Scaling (LARS) optimizer.
    
    Adapts learning rate per layer based on the ratio of weight and
    gradient norms, enabling large batch training. Particularly useful
    for self-supervised learning with very large batch sizes (2048+).
    
    Reference: You et al., "Large Batch Optimization for Deep Learning:
    Training BERT in 76 minutes" (ICLR 2020).
    """
    def __init__(
        self,
        params,
        learning_rate: float,
        weight_decay: float = 0,
        momentum: float = 0.9,
        eta: float = 0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        """
        Args:
            params: Model parameters to optimize.
            learning_rate (float): Base learning rate.
            weight_decay (float): L2 regularization. Default: 0.
            momentum (float): Momentum factor. Default: 0.9.
            eta (float): LARS coefficient. Default: 0.001.
            weight_decay_filter (callable): Function to exclude parameters from decay. Default: None.
            lars_adaptation_filter (callable): Function to exclude parameters from LARS. Default: None.
        """
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
        """Perform single optimization step with LARS adaptation."""
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


def exclude_bias_and_norm(p: torch.nn.Parameter) -> bool:
    """
    Check if parameter should be excluded from weight decay and LARS.
    
    Bias and normalization parameters (1-D tensors) are typically
    excluded from L2 regularization.
    
    Args:
        p (torch.nn.Parameter): Parameter tensor.
    
    Returns:
        bool: True if parameter is 1-D (bias or normalization).
    """
    return p.ndim == 1


def adjust_learning_rate(
    optimizer: optim.Optimizer,
    loader_len: int,
    step: int,
    total_epochs: int = 200,
    warmup_epochs: int = 10,
    base_lr: float = 6.4,
    min_lr: float = 0.001
):
    """
    Adjust learning rate with cosine annealing and warmup schedule.
    
    Linear warmup for first `warmup_epochs`, then cosine annealing decay
    over remaining epochs. Designed for large batch training.
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer to update.
        loader_len (int): Number of batches per epoch.
        step (int): Current step count (global step, not per-epoch).
        total_epochs (int): Total training epochs. Default: 200.
        warmup_epochs (int): Number of warmup epochs. Default: 10.
        base_lr (float): Base learning rate. Default: 6.4 (for batch 2048).
        min_lr (float): Minimum learning rate. Default: 0.001.
    """
    warmup_steps = warmup_epochs * loader_len
    max_steps = total_epochs * loader_len

    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * min_lr
        lr = base_lr * q + end_lr * (1 - q)

    optimizer.param_groups[0]["learning_rate"] = lr
