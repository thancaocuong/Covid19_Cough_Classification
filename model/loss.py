import torch.nn.functional as F
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target)
def focal_binary_cross_entropy(logits, targets, gamma=2):
    num_label = targets.size(1)
    l = logits.reshape(-1)
    t = targets.reshape(-1)
    p = torch.sigmoid(l)
    p = torch.where(t >= 0.5, p, 1-p)
    logp = - torch.log(torch.clamp(p, 1e-4, 1-1e-4))
    loss = logp*((1-p)**gamma)
    loss = num_label*loss.mean()
    return loss
def bce_with_logits_loss(output, target):
    output = output.view(-1).float().cuda()
    target = target.view(-1).float().cuda()
    return F.binary_cross_entropy_with_logits(output, target)

def masked_bce_with_logits_loss(output, target):
    output = output.view(-1).float().cuda()
    # print(output)
    # print(target)
    target = target.view(-1).float().cuda()
    bce_loss = F.binary_cross_entropy_with_logits(output, target, reduce=False)
    # print(bce_loss)
    max_item = bce_loss.max().item()
    # print(0.9*max_item)
    index = bce_loss > 0.95*max_item
    index[target==1] = False
    bce_loss[index] = 0.0
    # print(bce_loss)
    return bce_loss.mean()