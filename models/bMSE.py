import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.distributions import MultivariateNormal as MVN
class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def bmc_loss_md(self, pred, target, noise_var):
        """Compute the Multidimensional Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
        Args:
        pred: A float tensor of size [batch, d].
        target: A float tensor of size [batch, d].
        noise_var: A float number or tensor.
        Returns:
        loss: A float tensor. Balanced MSE Loss.
        """
        I = torch.eye(pred.shape[-1], device=pred.device)
        logits = MVN(pred.unsqueeze(1), noise_var*I).log_prob(target.unsqueeze(0))  # logit size: [batch, batch]
        loss = F.cross_entropy(logits, torch.arange(pred.shape[0],device=pred.device))     # contrastive-like loss
        loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable 
        return loss
        
    def forward(self, pred, target):
        noise_var = (self.noise_sigma ** 2).to(pred.device)
        return self.bmc_loss_md(pred[:,:,-1], target[:,:,-1], noise_var)

