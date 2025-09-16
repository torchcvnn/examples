import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, ignore_index=-100, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma
        # Convert alpha to a tensor if it's provided as a list/array.
        if alpha is not None:
            self.alpha = alpha.clone().detach()
        else:
            self.alpha = None

    def forward(self, softmax_probs, targets):
        """
        Args:
            softmax_probs: Tensor of precomputed softmax probabilities.
            targets: Ground truth labels.
        """
        # Prevent log(0) issues
        softmax_probs = torch.clamp(softmax_probs, min=1e-10, max=1.0)
        log_probs = torch.log(softmax_probs)

        # Ensure targets are int64 (required for indexing and loss functions)
        targets = targets.type(torch.int64)

        # Compute cross-entropy loss (per example) with ignore_index handling
        ce_loss = F.nll_loss(
            log_probs, targets, reduction="none", ignore_index=self.ignore_index
        )

        # Recover the probability for the true class
        pt = torch.exp(-ce_loss)

        # Compute the focal modulation term
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss

        # If alpha is provided, apply per-class weighting
        if self.alpha is not None:
            # Ensure alpha is on the same device as softmax_probs
            alpha = self.alpha.to(softmax_probs.device)
            # Index into alpha using the target labels
            alpha_weights = alpha[targets]
            loss = alpha_weights * loss

        # Create a mask to only average over non-ignored indices
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            loss = loss[valid_mask]

        return loss.mean()
