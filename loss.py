import numpy as np
import torch


class compute_loss_multi_mse(torch.nn.Module):
    # Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
    def __init__(self, ntask=4):
        super(compute_loss_multi_mse, self).__init__()
        
        self.ntask = ntask
        self.sigma = torch.nn.Parameter(torch.zeros(ntask), requires_grad=True)

    def forward(self, pred, targets):
        
        precision = torch.exp(-self.sigma)

        targets = targets.contiguous()
        
        if self.ntask == 1:
            loss = torch.nn.functional.mse_loss(pred[:,0].contiguous(), targets.float())
        else:
            loss = 0.0
            for i in range(0, self.ntask):
                loss = loss + precision[i] * torch.nn.functional.mse_loss(pred[:,i].contiguous(), targets[:,i].float()) + self.sigma[i]

        loss = loss/targets.shape[1]

        return loss