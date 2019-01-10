import torchvision.models as models
import torch

backbone = models.resnet50(pretrained=True)
print(backbone)
