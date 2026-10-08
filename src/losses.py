"""
Mathematical Loss Formulations for RSNA Knee Abnormality Detection:
1. PolyLoss (Multi-label Poly-1 BCE)
2. Asymmetric Focal Loss (ASL)
3. Logit-Adjusted Loss
4. Homoscedastic Task Uncertainty Weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Poly1Loss(nn.Module):
    """
    PolyLoss: Perturbing the leading polynomial expansion coefficient of Cross-Entropy
    L_Poly1 = L_BCE + epsilon * (1 - P_t)
    where P_t is the model's probability assigned to the ground truth label.
    """
    def __init__(self, epsilon: float = 1.0, reduction: str = 'mean', pos_weight: torch.Tensor = None):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        
        # Probabilities
        probs = torch.sigmoid(logits)
        # P_t = y * p + (1 - y) * (1 - p)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        
        # Leading polynomial term
        poly_loss = bce_loss + self.epsilon * (1.0 - p_t)
        
        if self.reduction == 'mean':
            return poly_loss.mean()
        elif self.reduction == 'sum':
            return poly_loss.sum()
        return poly_loss

class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss (ASL) for multi-label classification.
    Decouples focusing parameters gamma_pos and gamma_neg,
    and applies probability margin shifting to suppress negative gradient noise.
    """
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Positive loss
        pos_p = probs
        pos_loss = - targets * torch.pow(1.0 - pos_p, self.gamma_pos) * torch.log(pos_p.clamp(min=self.eps))
        
        # Negative loss with asymmetric probability margin shifting
        neg_p = probs
        if self.clip > 0:
            neg_p = (neg_p - self.clip).clamp(min=0.0)
        neg_loss = - (1.0 - targets) * torch.pow(neg_p, self.gamma_neg) * torch.log((1.0 - neg_p).clamp(min=self.eps))
        
        loss = pos_loss + neg_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class LogitAdjustedLoss(nn.Module):
    """
    Logit-Adjusted Loss (Menon et al., 2021)
    Mathematically compensates for extreme class imbalance by shifting logits
    by tau * log(pi_c / (1 - pi_c)) where pi_c is the empirical class prior.
    """
    def __init__(self, class_priors: torch.Tensor, tau: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        # Clamp priors to prevent log(0)
        priors = torch.clamp(class_priors, min=1e-4, max=1.0 - 1e-4)
        # Compute adjustment delta: tau * (log(pi) - log(1 - pi))
        self.register_buffer('delta', tau * (torch.log(priors) - torch.log(1.0 - priors)))
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adjusted_logits = logits + self.delta
        return F.binary_cross_entropy_with_logits(adjusted_logits, targets, reduction=self.reduction)

class HomoscedasticTaskUncertaintyLoss(nn.Module):
    """
    Multi-Task Homoscedastic Uncertainty Weighting (Kendall et al., CVPR 2018)
    Dynamically learns task observation noise log(sigma_i^2) to balance gradients
    across all 12 correlated knee abnormality targets:
    L_total = sum_i (0.5 * exp(-s_i) * L_i + 0.5 * s_i)
    where s_i = log(sigma_i^2) is a learned parameter.
    """
    def __init__(self, num_tasks: int = 12, base_loss_fn = None):
        super().__init__()
        self.num_tasks = num_tasks
        # Initialize log_vars to 0 (sigma = 1)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.base_loss_fn = base_loss_fn or nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Per-task loss: shape (batch_size, num_tasks)
        task_losses = self.base_loss_fn(logits, targets)
        mean_task_losses = task_losses.mean(dim=0) # shape (num_tasks,)
        
        # Weighted uncertainty sum
        precision = torch.exp(-self.log_vars)
        weighted_loss = 0.5 * precision * mean_task_losses + 0.5 * self.log_vars
        return weighted_loss.sum()
