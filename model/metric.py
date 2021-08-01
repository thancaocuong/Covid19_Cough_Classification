import torch
import sklearn
import numpy as np

def roc_auc_multi_label(output, target, label = 1):
    output = output.numpy()
    target = target.numpy()

    if len(target.shape) == 1:
        target = target.astype(int)
        n_values = np.max(target) + 1
        target = np.eye(n_values)[target]
        
    # print(target)
    return sklearn.metrics.roc_auc_score(target, output, multi_class = 'ovr', average=None)[1]

def roc_auc(output, target):
    output = torch.sigmoid(output.float())
    output = output.numpy()
    target = target.numpy()
    return sklearn.metrics.roc_auc_score(target, output)

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
