# -*- coding:utf-8 -*-
import platform
print(platform.python_version())
import sys, os, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image
import pandas as pd
from network import *

import cv2

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Inference the testset and save in submission.csv')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--model', dest='model', help='which model to use',
            default='ResNet34', type=str)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
            default='../models/ResNet34_best_focalloss.pth', type=str)
    parser.add_argument('--test_path', dest='test_path', help='Path of test images', 
	        default='/data2/shentao/DATA/Kaggle/Whale/raw/test/')
    parser.add_argument('--output_string', dest='output_string', help='Submission file path.',
            default = '../results/submission.csv')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True

    batch_size = 1
    gpu = args.gpu_id
    snapshot_path = args.snapshot
    if args.model == 'ResNet34':
        backbone = models.resnet34(pretrained=True)
        model = ResNet34(backbone, num_classes=5004)

    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

    model.cuda(gpu)
    print('Loading snapshot.')
    # Load snapshot
    checkpoint = torch.load(snapshot_path)
    model.load_state_dict(checkpoint['state_dict'])
    #model.load_state_dict(saved_state_dict)

    print('Loading data.')

    transformations = transforms.Compose([transforms.Scale(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # create the dict of {onehot:name, ...}
    img_id_list = pd.read_csv('../data/sample_submission.csv')
    Images = img_id_list['Image']
    txt = open('../data/name_onehot.txt', 'r')
    lines = txt.readlines()
    onehot_name_dict = {}
    for line in lines:
        name, onehot = line.strip().split(' ')
        onehot_name_dict[onehot] = name

    print('Ready to test network.')
    sub = pd.read_csv(args.output_string)

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    total = len(Images)
    cnt = 1
    for imgname in Images:
        if imgname is None:
            break
        cnt += 1 
        imgpath = os.path.join(args.test_path, imgname)
        im = Image.open(imgpath).convert('RGB')
        im = im.resize((256,256))

        #im = Image.fromarray(im)
        # Tranimm
        im = transformations(im)
        img_shape = im.size()
        im = im.view(1, img_shape[0], img_shape[1], img_shape[2])
        im = Variable(im).cuda(gpu)
		
        output = model(im)

        pred = output.data.topk(5, dim=1, largest=True, sorted=True) # top5
        # pred = (tensor([[28.2273, 27.4925, 27.3513, 25.6014, 25.2821]], device='cuda:0'), 
        #          tensor([[4533, 1606, 2936, 3820, 3121]], device='cuda:0'))
        pred_top5 = pred[1].cpu().numpy().flatten()

        # save predict result in submission.csv
        names = []
        names.append('new_whale')
        for i in range(4):
            names.append(onehot_name_dict[str(pred_top5[i])])
        sub.loc[sub['Image'] == imgname, 'Id'] = ' '.join(name for name in names)
        print('{}/{}'.format(cnt, total))
    sub.to_csv(args.output_string, index=False)
            
