# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet18, self).__init__()
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(512, num_classes)


    def forward(self, x):
        print self.backbone
        x = self.backbone(x)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    resnet18 = models.resnet18(pretrained=True)
    models = ResNet18(resnet18, 1200)
    data = torch.randn(1, 3, 224, 224)
    x = models(data)
