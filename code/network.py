# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def whitening(im):
    batch_size, channel, h, w = im.shape
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    im = torch.cat([(im[:,[0]]-mean[0])/std[0],
                    (im[:,[1]]-mean[1])/std[1],
                    (im[:,[2]]-mean[2])/std[2]], 1)
    return im

def l2_norm(x):
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    x = torch.div(x, norm)
    return x

class ResNet34(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet34, self).__init__()
        self.backbone = model

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)


    def forward(self, x):
        x = whitening(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    resnet18 = models.resnet18(pretrained=True)
    models = ResNet18(resnet18, 5004)
    data = torch.randn(1, 3, 224, 224)
    x = models(data)
    print(x)
    print(x.size())
