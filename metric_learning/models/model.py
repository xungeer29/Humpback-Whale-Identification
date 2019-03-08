# -*- coding:utf-8 -*-

from models.modelZoo import *

import torch
import torch.nn.functional as F
from torchvision import models

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

class model_whale(nn.Module):
    def __init__(self, num_classes=5005, inchannels=3, model_name='resnet18'):
        super(self, model_whale).__init__()
        planes = 512
        self.model_name = model_name

        if model_name == 'resnet18':
            pass
        elif model_name == 'seresnext50':
            self.basemodel = se_resnext50_32x4d(inchannels=inchannels, pretrained='imagenet')
            planes = 2048
        elif model_name == 'seresnext101':
            self.basemodel = se_resnext101_32x4d(inchannels=inchannels, pretrained='imagenet')
            planes = 2048
        else:
            assert False, '{} is error'.format(model_name)

        self.bottlenect_g = nn.BatchNormal1d(planes)
        self.bottlenect_g.bias.requires_grad_(False)

        self.fc = nn.Linear(planes, num_classes)
        init.normal_(self.fc.weight, std=0.001)
        init.constant_(self.fc.bias, 0)

    def forward(self, x, label=None):
        feat = self.basemodel(x)
        # global feature
        global_feat = F.avg_pool2d(feat, feat.size()[2:])
        global_feat = global_feat.view(global_feat.size(0), -1)
        global_feat = F.dropout(global_feat, p=0.2)
        global_feat = self.bottleneck_g(global_feat)
        global_feat = l2_norm(global_feat)

        out = self.fc(global_feat) * 16

        return global_feat, out

    def freeze_bn(self):
        for m in self.named_modules():
            if 'layer' in m[0]:
                if isinstance(m[1], nn.BatchNorm2d):
                    m[1].eval()

    def freeze(self):
        for param in self.basemodel.parameters():
            param.requires_grad = False
        if self.model_name.find('res') > -1 or self.model_name.find('senet154') > -1:
            for param in self.basemodel.layer3.parameters():
                param.requires_grad = True
            for param in self.basemodel.layer4.parameters():
                param.requires_grad = True

    def getLoss(self, global_feat, local_feat, results,labels):
        loss_ = sigmoid_loss(results, labels, topk=30)

        self.loss = loss_

if __name__ == '__main__':
    basemodel = se_resnext50_32x4d(inchannels=3, pretrained='imagenet')
    print basemodel

