import torch
import torch.nn.functional as F
import torch.nn as nn


class FocalLoss(nn.Module):
    """
        Return a FocalLoss object to calculate the following loss function:
        FL(p_t) = −(1 − p_t)^γ log(p_t)
        
    """
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
            inputs: batch_size * dim
            targets: (batch,)
        """

        ce_loss = F.cross_entropy(inputs, targets)
        loss = self.alpha * (1 - torch.exp(-ce_loss)) ** self.gamma * ce_loss
        return loss