# -*- coding:utf-8 -*-
import torch 
import torch.nn.functional as F

# F.binary_cross_entropy_with_logits
def bce_metric(output, target):
    prob = F.sigmoid(output)
    prob[prob > 0.5] = 1
    prob[prob < 0.5] = 0
    correct = prob.eq(target.float().view(-1, 1).expand_as(prob))
    correct = correct.float().sum(0, keepdim=False)
    acc = correct/len(target)

    return acc.data[0]

