import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        at = self.alpha.gather(0, targets) if self.alpha is not None else 1.0
        fl = at * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return fl.mean()
        elif self.reduction == 'sum':
            return fl.sum()
        else:
            return fl