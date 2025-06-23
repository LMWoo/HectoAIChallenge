import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SoftTargetFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        print('Using Loss : SoftTargetFocalLoss')

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        if targets.dtype == torch.float32 and targets.dim() == 2:
            if self.alpha is not None:
                alpha_t = self.alpha.unsqueeze(0)
                alpha_factor = targets * alpha_t
            else:
                alpha_factor = targets

            focal_weight = (1.0 - probs) ** self.gamma
            loss = -alpha_factor * focal_weight * log_probs
            loss = loss.sum(dim=1)
        else:
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            at = self.alpha.gather(0, targets) if self.alpha is not None else 1.0
            loss = at * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

        # ce_loss = F.cross_entropy(logits, targets, reduction='none')
        # pt = torch.exp(-ce_loss)
        # at = self.alpha.gather(0, targets) if self.alpha is not None else 1.0
        # fl = at * (1 - pt) ** self.gamma * ce_loss
        # if self.reduction == 'mean':
        #     return fl.mean()
        # elif self.reduction == 'sum':
        #     return fl.sum()
        # else:
        #     return fl