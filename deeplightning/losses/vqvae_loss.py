import torch.nn.functional as F
from einops import rearrange
import torch
import torch.nn as nn


class VQVAE_Loss(nn.Module):
    
    def __init__(self, smooth_l1_loss, num_tokens, kl_div_loss_weight):
        super(VQVAE_Loss, self).__init__()
        self.num_tokens = num_tokens
        self.kl_div_loss_weight = kl_div_loss_weight
        self.recon_loss_fn =  F.smooth_l1_loss if smooth_l1_loss else F.mse_loss

    def kl_divergence(self, logits, device):
        logits = rearrange(logits, 'b n h w -> b (h w) n')
        log_qy = F.log_softmax(logits, dim = -1)
        log_uniform = torch.log(torch.tensor([1. / self.num_tokens], device = device))
        kl_div = F.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target = True)
        return kl_div * self.kl_div_loss_weight

    def forward(self, x, x_recon, logits=None):
        loss = self.recon_loss_fn(x, x_recon)
        if self.kl_div_loss_weight > 0.0:
            loss += self.kl_divergence(logits, x.device)
        return loss