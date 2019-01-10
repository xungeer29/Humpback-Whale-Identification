import torch.utils.data as data

from PIL import Image
import pandas as pd
import os
import os.path
import numpy as np

def default_loader(path):
    img_bbox = pd.read_csv('../data/bounding_boxes.csv')
    imgs = img_bbox['Image']
    xmins = img_bbox['x0']
    ymins = img_bbox['y0']
    xmaxs = img_bbox['x1']
    ymaxs = img_bbox['y1']
    imgname = path.split('/')[-1]
    idx = imgs.isin([imgname])
    #print('idx:{}'.format(idx))
    xmin = int(xmins[idx])
    ymin = int(ymins[idx])
    xmax = int(xmaxs[idx])
    ymax = int(ymaxs[idx])
    w = xmax-xmin
    h = ymax-ymin
    #print('bbox: {} {} {} {} {}'.format(imgname, xmin, ymin, xmax, ymax))
    img = Image.open(path).convert('RGB')
    W, H = img.size
    img = img.crop((max(0,xmin-w//10), max(0,ymin-h//10), min(W,xmax+w//10), min(H,ymax+h//10)))
    img = img.resize((256, 256))
    #img.save('../log/'+imgname)
    return img

def default_list_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label = line.strip().split(' ')
            if label == '-1':
                continue
            #print(imgPath, label)
            imgList.append((imgPath, int(label)))
    return imgList

class Dataset(data.Dataset):
    def __init__(self, root, fileList, transform=None, list_reader=default_list_reader, loader=default_loader):
        self.root      = root
        self.imgList   = list_reader(fileList)
        self.transform = transform
        self.loader    = loader

    def __getitem__(self, index):
        imgPath, target = self.imgList[index]
        img = self.loader(os.path.join(self.root, imgPath))

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgList)
