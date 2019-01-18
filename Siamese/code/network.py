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

        x = self.backbone.avgpool(x) # 1*1*512
        x = x.view(x.size(0), -1)
        
        return x

class ResNet50(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet50, self).__init__()
        self.backbone = model

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, num_classes)


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


class ResNet101(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet101, self).__init__()
        self.backbone = model

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, num_classes)


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

class ResNet152(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet152, self).__init__()
        self.backbone = model

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, num_classes)


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

class Head01(nn.Module):
    def __init__(self):
        super(Head01, self).__init__()

    def forward(self, x1, x2):
        x_mul = x1.mul(x2)
        x_add = torch.add(x1, x2)
        x_norm_l1 = torch.abs(x1-x2)
        x_norm_l2 = x_norm_l1.mul(x_norm_l1)

        x = torch.cat((x_mul, x_add, x_norm_l1, x_norm_l2), 1)

        x = x.view(x1.size(0), 1, 4, x1.size(1)) # batch_size, c, h, w
        x = nn.Conv2d(1, 32, (1, 4))(x)
        x = x.view(x1.size(0), x1.size(1), 32, 1)
        x = nn.conv2d(512, 1, (1, 32))(x)

        x = x.view(x1.size(0), -1)

        return x

class Head(nn.Module):
    def __init__(self, dim=512*2):
        super(Head, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(dim, dim//2)
        self.bn = nn.BatchNorm1d(dim//2)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim//2, 1)

    def forward(self, x1, x2):
        x1 = l2_norm(x1)
        x2 = l2_norm(x2)
        # x1, x2: batch_size*512
        x = torch.cat([x1, x2], 1) # BS*1024
        x = x.view(x1.size(0), -1) # BS*1024
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x



class Siamese(nn.Module):
    def __init__(self, branch, head):
        super(Siamese, self).__init__()
        self.branch = branch
        self.head = head

    def forward(self, x1, x2):
        x1 = self.branch(x1)
        x2 = self.branch(x2)
        x = self.head(x1, x2)

        return x


if __name__ == '__main__':
    backbone = models.resnet34(pretrained=True)
    branch = ResNet34(backbone, 5004)
    data1 = torch.randn(16, 3, 224, 224)
    data2 = torch.randn(16, 3, 224, 224)
    branch1 = branch(data1)
    branch2 = branch(data2)
    x = Head()(branch1, branch2) # siamese
    print(x)
    print(x.size())
