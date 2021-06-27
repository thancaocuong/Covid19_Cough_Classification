import torch.nn.functional as F
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target)

def bce_with_logits_loss(output, target):
    output = output.view(-1).float().cuda()
    target = target.view(-1).float().cuda()
    return F.binary_cross_entropy_with_logits(output, target)