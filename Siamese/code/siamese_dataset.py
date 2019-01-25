# -*- coding:utf-8 -*-
import torch.utils.data as data

from PIL import Image
import pandas as pd
import os
import os.path
import numpy as np
import random
import pickle

from configs import *
from utils import expand_path

def read_raw_image(p):
    img = Image.open(expand_path(p))

    return img

def build_transform(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    rotation = np.deg2rad(rotation)
    shear = np.deg2rad(shear)
    rotation_matrix = np.array(
        [[np.cos(rotation), np.sin(rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
    shift_matrix = np.array([[1, 0, height_shift], [0, 1, width_shift], [0, 0, 1]])
    shear_matrix = np.array([[1, np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
    zoom_matrix = np.array([[1.0 / height_zoom, 0, 0], [0, 1.0 / width_zoom, 0], [0, 0, 1]])
    shift_matrix = np.array([[1, 0, -height_shift], [0, 1, -width_shift], [0, 0, 1]])

    return np.dot(np.dot(rotation_matrix, shear_matrix), np.dot(zoom_matrix, shift_matrix))

def read_cropped_image(p, h2p, p2size, p2bb, augment=True):
    """
    p: 图像名
    h2p: hash2pic dict
    p2size: pic2size dict
    p2bb: pic2bbox dict
    """
    # If an image id was given, convert to filename
    if p in h2p:
        p = h2p[p]
    size_x, size_y = p2size[p]

    # Determine the region of the original image we want to capture based on the bounding box.
    row = p2bb.loc[p]
    x0, y0, x1, y1 = row['x0'], row['y0'], row['x1'], row['y1']
    dx = x1 - x0
    dy = y1 - y0
    x0 -= dx * crop_margin
    x1 += dx * crop_margin + 1
    y0 -= dy * crop_margin
    y1 += dy * crop_margin + 1
    if x0 < 0:
        x0 = 0
    if x1 > size_x:
        x1 = size_x
    if y0 < 0:
        y0 = 0
    if y1 > size_y:
        y1 = size_y
    dx = x1 - x0
    dy = y1 - y0
    if dx > dy * anisotropy:
        dy = 0.5 * (dx / anisotropy - dy)
        y0 -= dy
        y1 += dy
    else:
        dx = 0.5 * (dy * anisotropy - dx)
        x0 -= dx
        x1 += dx

    # Generate the transformation matrix
    trans = np.array([[1, 0, -0.5 * HEIGHT], [0, 1, -0.5 * WIDTH], [0, 0, 1]])
    trans = np.dot(np.array([[(y1 - y0) / HEIGHT, 0, 0], [0, (x1 - x0) / WIDTH, 0], [0, 0, 1]]), trans)
    if augment:
        trans = np.dot(build_transform(
                    random.uniform(-5, 5),
                    random.uniform(-5, 5),
                    random.uniform(0.8, 1.0),
                    random.uniform(0.8, 1.0),
                    random.uniform(-0.05 * (y1 - y0), 0.05 * (y1 - y0)),
                    random.uniform(-0.05 * (x1 - x0), 0.05 * (x1 - x0))),
                    trans)
    trans = np.dot(np.array([[1, 0, 0.5 * (y1 + y0)], 
                             [0, 1, 0.5 * (x1 + x0)], 
                             [0, 0, 1]]), 
                             trans)
    img_gray = read_raw_image(p).convert('L')
    img_crop = img_gray.crop((x0, y0, x1, y1))

    img_crop.save(p)

    return img_crop

def load_data(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
    
def for_train(w2hs):
    train = [] # A list of training image ids
    for hs in w2hs.values():
        if len(hs) > 1:
            train += hs
    random.shuffle(train)
    train_set = set(train)

    w2ts = {}  # Associate the image ids from train to each whale id.
    for w, hs in w2hs.items():
        for h in hs:
            if h in train_set:
                if w not in w2ts:
                    w2ts[w] = []
                if h not in w2ts[w]:
                    w2ts[w].append(h)
    for w, ts in w2ts.items():
        w2ts[w] = np.array(ts)

    t2i = {}  # The position in train of each training image id
    for i, t in enumerate(train):
        t2i[t] = i

    return train, train_set, w2ts, t2i
    

class WhaleDataset(data.Dataset):
    def __init__(self, transform, score, steps=1000, batch_size=32, mode='train'):
        self.transform = transform
        self.mode = mode # train, val, test
        self.score = -score # Maximizing the score is the same as minimuzing -score.
        self.steps = steps
        self.batch_size = batch_size

        self.h2p = load_data('../data/h2p.pickle')
        self.p2size = load_data('../data/p2size.pickle')
        self.p2bb = pd.read_csv(BB_DF).set_index('Image')
        self.w2hs = load_data('../data/w2hs.pickle')
        self.train, self.train_set, self.w2ts, self.t2i = for_train(self.w2hs)
        for ts in self.w2ts.values():
            idxs = [self.t2i[t] for t in ts]
            for i in idxs:
                for j in idxs:
                    # Set a large value for matching whales -- eliminates this potential pairing
                    self.score[i, j] = 10000.0
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.steps <= 0:
            return  # Skip this on the last epoch.
        self.steps -= 1
        self.match = []
        self.unmatch = []
        _, _, x = lapjv(self.score)  # Solve the linear assignment problem
        y = np.arange(len(x), dtype=np.int32)

        # Compute a derangement for matching whales
        for ts in self.w2ts.values():
            d = ts.copy()
            while True:
                random.shuffle(d)
                if not np.any(ts == d): 
                    break
                for ab in zip(ts, d): self.match.append(ab)

        # Construct unmatched whale pairs from the LAP solution
        for i, j in zip(x, y):
            if i == j:
                print(self.score)
                print(x)
                print(y)
                print(i, j)
            assert i != j
            self.unmatch.append((self.train[i], self.train[j]))

        # Force a different choice for an eventual next epoch.
        self.score[x, y] = 10000.0
        self.score[y, x] = 10000.0
        random.shuffle(self.match)
        random.shuffle(self.unmatch)
        assert len(self.match) == len(train) and len(self.unmatch) == len(train)

    def __getitem__(self, index):
        start = self.batch_size * index
        end = min(start + self.batch_size, len(self.match) + len(self.unmatch))
        size = end - start
        assert size > 0
        a = np.zeros((size, HEIGHT, WIDTH, CHANNEL), dtype=float)
        b = np.zeros((size, HEIGHT, WIDTH, CHANNEL), dtype=float) 
        c = np.zeros((size, 1), dtype=float)
        j = start // 2
        for i in range(0, size, 2):
            a[]


    def __len__(self):
        return (len(self.match) + len(self.unmatch) + self.batch_size - 1) // self.batch_size


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
