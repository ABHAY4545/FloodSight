import torch
import torch.nn.functional as F

def focal_loss(pred, mask, alpha, gamma=2):
    alpha = alpha.to(mask.device)
    alpha_t = alpha[mask]
    ce = F.cross_entropy(pred,mask, reduction="none")
    loss = alpha_t * ((1 - torch.exp(-ce))** gamma) * ce
    return loss.mean()


def dice_loss(pred, mask):
    pass