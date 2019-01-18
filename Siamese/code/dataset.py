# -*- coding:utf-8 -*-
import torch.utils.data as data

from PIL import Image
import pandas as pd
import os
import os.path
import numpy as np
import random
import pickle

# return the croped whale tail image
def default_loader(path):
    imgname = path.split('/')[-1]
    img_bbox = pd.read_csv('../data/bounding_boxes.csv')
    imgs = img_bbox['Image']
    xmins = img_bbox['x0']
    ymins = img_bbox['y0']
    xmaxs = img_bbox['x1']
    ymaxs = img_bbox['y1']
    idx = imgs.isin([imgname])
    #print('idx:{}'.format(idx))
    xmin = int(xmins[idx])
    ymin = int(ymins[idx])
    xmax = int(xmaxs[idx])
    ymax = int(ymaxs[idx])
    w = xmax-xmin
    h = ymax-ymin
    #print('bbox: {} {} {} {} {}'.format(imgname, xmin, ymin, xmax, ymax))
    #print path
    img = Image.open(path).convert('RGB')
    W, H = img.size
    img = img.crop((max(0,xmin-w//10), max(0,ymin-h//10), min(W,xmax+w//10), min(H,ymax+h//10)))
    img = img.resize((224, 224))
    #img.save('../log/'+imgname)
    return img

# trainset image and label list
def default_list_reader(fileList):
    """
    # 图像哈希去重
    cleaned_pics = []
    with open('../data/hash2pic.pickle', 'rb') as f:
        hash2pic = pickle.load(f)
        for hash, pic in list(hash2pic.items()):
            cleaned_pics.append(pic)
    """
    imgList = []
    id2pics = {}
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label = line.strip().split(' ')
            if int(label) not in id2pics:
                id2pics[int(label)] = []
            img = imgPath.split('/')[-1]
            if img not in id2pics[int(label)]:
                id2pics[int(label)].append(img)
            if label == '-1':
                continue
            imgList.append((imgPath, int(label)))
    return imgList, id2pics


def whale_id_num(filepath):
    whales, ids, picnum = [], [], []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            whale, id_, num = line.strip().split(' ')
            whales.append(whale)
            ids.append(int(id_))
            picnum.append(int(num))
    return whales, ids, picnum


class WhaleDataset(data.Dataset):
    def __init__(self, root, fileList, transform=None, mode='train', list_reader=default_list_reader, loader=default_loader):
        self.root = root
        self.imgList, self.id2pics = list_reader(fileList) # trainset image and label list
        self.transform = transform
        self.loader = loader # croped image
        self.mode = mode # train, val, test

    def __getitem__(self, index):
        if self.mode == 'train':
            imgPath, target = self.imgList[index]
            imgname = imgPath.split('/')[-1]
            label = random.randint(0, 1)
            #print sum(self.id2pics)
            # 相同 ID 图像配对
            #print len(self.id2pics)
            if label == 1:
                # 任选一个id，该id图像数量要大于1
                while True:
                    ID = random.randint(0, len(self.id2pics)-2) # 去除new_whale，[0, 5003]
                    ID_num = len(self.id2pics[ID])
                    #print 'True', ID, ID_num
                    if ID_num > 1:
                        break
                while True:
                    im1_idx = random.randint(0, ID_num-1)
                    im2_idx = random.randint(0, ID_num-1)
                    #print 'im index', im1_idx, im2_idx
                    if im1_idx != im2_idx:
                        break
                imName1 = self.id2pics[ID][im1_idx]
                imName2 = self.id2pics[ID][im2_idx]
                #print('label:{}\tID:{}\tim1:{}\tim2:{}'.format(label, ID, imName1, imName2))

            else: # 不同图像配对
                while True:
                    ID1 = random.randint(0, len(self.id2pics)-2)
                    ID2 = random.randint(0, len(self.id2pics)-2)
                    if ID1 != ID2:
                        break
                im1_idx = random.randint(0, len(self.id2pics[ID1])-1)
                im2_idx = random.randint(0, len(self.id2pics[ID2])-1)
                imName1 = self.id2pics[ID1][im1_idx]
                imName2 = self.id2pics[ID2][im2_idx]
                #print('label:{}\tID1:{}\tID2:{}\tim1:{}\tim2:{}'.format(target, ID1, ID2, imName1, imName2))

            im1 = self.loader(os.path.join(self.root, 'train', imName1))
            im2 = self.loader(os.path.join(self.root, 'train', imName2))

            if self.transform is not None:
                im1 = self.transform(im1)
                im2 = self.transform(im2)
            return im1, im2, label

    def __len__(self):
        return len(self.imgList)

if __name__ == '__main__':
    root =  '/data2/shentao/DATA/Kaggle/Whale/raw/'
    dataset = WhaleDataset(root, '../data/train_full.txt')
    print(len(dataset))
    #print(dataset)
    for im1, im2 , target in dataset:
        print(im1, im2, target)
        break
