# -*- coding:utf-8 -*- 

import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch
from models.lovasz import *

def sigmoid_loss(results, labels, topk=10):
    if len(results.shape) == 1:
        results = results.view(1, -1)
    batch_size, class_num = results.shape
    labels = labels.view(-1, 1)
    one_hot_target = torch.zeros(batch_size, class_num + 1).cuda().scatter_(1, labels, 1)[:, :5004 * 2]
    lovasz_loss = lovasz_hinge(results, one_hot_target)
    error = torch.abs(one_hot_target - torch.sigmoid(results))
    error = error.topk(topk, 1, True, True)[0].contiguous()
    target_error = torch.zeros_like(error).float().cuda()
    error_loss = nn.BCELoss(reduce=True)(error, target_error)
    labels = labels.view(-1)
    indexs_new = (labels != 5004 * 2).nonzero().view(-1)
    if len(indexs_new) == 0:
        return error_loss
    results_nonew = results[torch.arange(0, len(results))[indexs_new], labels[indexs_new]].contiguous()
    target_nonew = torch.ones_like(results_nonew).float().cuda()
    nonew_loss = nn.BCEWithLogitsLoss(reduce=True)(results_nonew, target_nonew)

    return nonew_loss + error_loss + lovasz_loss * 0.5
