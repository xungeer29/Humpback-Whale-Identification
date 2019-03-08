# -*- coding:utf-8 -*-

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import cv2
import random
import os
import pandas as pd
import numpy as np

class WhaleDataset(Dataset):
    def __init__(self, names, labels=None, mode='train', transform_train=None, min_num_classes=0):
        super(WhaleDataset, self).__init__()
        self.names = names
        self.labels = labels
        self.mode = mode
        self.transform_train = transform_train
        self.min_num_classes = min_num_classes
        self.label_dict = self.load_labels()
        self.bbox_dict = self.load_bbox()
        self.id_labels = {im:whale for im, whale in zip(self.names, self.labels)}
        if mode in ['train', 'valid']:
            self.train_dict = self.balance_train()
            self.labels = [k for k in self.train_dict.keys()
                             if len(self.train_dict[k]) >= min_num_classes]

    def load_labels(self):
        print('loading labels...')
        label_dict = {}
        with open('../data/whale2id.txt', 'r') as f:
            for line in f.readlines():
                whale, id = line.strip().split(' ')
                if whale == 'new_whale':
                    id = 5004 * 2
                label_dict[whale] = int(id)
        return label_dict

    def load_bbox(self):
        print('loading bbox...')
        root = '/media/gfx/data1/DATA/Kaggle/whale'
        ftd_dir = os.path.join(root, 'featured/bounding_boxes.csv')
        plgd_dir = os.path.join(root, 'playground/') # TODO: pickle from 3 to 2
        bbox_df = pd.read_csv(ftd_dir)
        bbox_dict = {}
        for im, x0, y0, x1, y1 in zip(bbox_df['Image'], bbox_df['x0'], bbox_df['y0'],
                                      bbox_df['x1'], bbox_df['y1']):
            bbox_dict[im] = [x0, y0, x1, y1]
        return bbox_dict

    def balance_train(self):
        train_dict = {}
        for name, label in zip(self.names, self.labels):
            if label not in train_dict.keys():
                train_dict[label] = []
            train_dict[label].append(name)
        return train_dict

    def __len__(self):
        return len(self.labels)

    def get_image(self, name, transform, label, mode='train'):
        root = '/media/gfx/data1/DATA/Kaggle/whale'
        image = cv2.imread(os.path.join(root, 'featured/train', name))
        if image is None:
            image = cv2.imread(os.path.join(root, 'fratured/test', name))
        if image is None:
            image = cv2.imread(os.path.join(root, 'playground/train', name))
        if image is None:
            image = cv2.imread(os.path.join(root, 'playground/test', name))
        x0, y0, x1, y1 = self.bbox_dict[name]
        image = image[int(y0):int(y1), int(x0):int(x1)]
        image, add_ = transform(image, label)
        return image, add_

    def __getitem__(self, index):
        label = self.labels[index]
        names = self.train_dict[label]
        nums = len(names)
        if nums == 1:
            anchor_name = names[0]
            positive_name = names[0]
        else:
            anchor_name, positive_name = random.sample(names, 2)
        negative_label = random.choice(list(set(self.labels) ^ set([label, 'new_whale']))) # ???
        negative_name = random.choice(self.train_dict[negative_label])
        negative_label2 = 'new_whale'
        negative_name2 = random.choice(self.train_dict[negative_label2])

        anchor_image, anchor_add = self.get_image(anchor_name, self.transform_train, label)
        positive_image, positive_add = self.get_image(positive_name, self.transform_train, label)
        negative_image, negative_add = self.get_image(negative_name, self.transform_train, negative_label)
        negative_image2, negative_add2 = self.get_image(negative_name2, self.transform_train, negative_label2)

        assert anchor_name != negative_name

        return [anchor_image, positive_image, negative_image, negative_image2],\
               [self.label_dict[label]+anchor_add, self.label_dict[label]+positive_add,
                self.label_dict[negative_label]+negative_add, 
                self.label_dict[negative_label2]+negative_add2]

class WhaleTestDataset(Dataset):
    def __init__(self, names, labels=None, mode='test', transform=None):
        super(WhaleTestDataset, self).__init__()
        self.names = names
        self.labels = labels
        self.mode = mode
        self.transform = transform
        self.label_dict = self.load_labels()
        self.bbox_dict = self.load_bbox()

    def load_labels(self):
        print('loading labels...')
        label_dict = {}
        with open('../data/whale2id.txt', 'r') as f:
            for line in f.readlines():
                whale, id = line.strip().split(' ')
                if whale == 'new_whale':
                    id = 5004 * 2
                label_dict[whale] = int(id)
        return label_dict

    def load_bbox(self):
        print('loading bbox...')
        root = '/media/gfx/data1/DATA/Kaggle/whale'
        ftd_dir = os.path.join(root, 'featured/bounding_boxes.csv')
        plgd_dir = os.path.join(root, 'playground/') # TODO: pickle from 3 to 2
        bbox_df = pd.read_csv(ftd_dir)
        bbox_dict = {}
        for im, x0, y0, x1, y1 in zip(bbox_df['Image'], bbox_df['x0'], bbox_df['y0'],
                                      bbox_df['x1'], bbox_df['y1']):
            bbox_dict[im] = [x0, y0, x1, y1]
        return bbox_dict

    def __len__(self):
        return len(self.labels)

    def get_image(self, name, transform, label, mode='train'):
        root = '/media/gfx/data1/DATA/Kaggle/whale'
        image = cv2.imread(os.path.join(root, 'featured/train', name))
        if image is None:
            image = cv2.imread(os.path.join(root, 'fratured/test', name))
        if image is None:
            image = cv2.imread(os.path.join(root, 'playground/train', name))
        if image is None:
            image = cv2.imread(os.path.join(root, 'playground/test', name))
        x0, y0, x1, y1 = self.bbox_dict[name]
        image = image[int(y0):int(y1), int(x0):int(x1)]
        image, add_ = transform(image) # mask
        return image

    def getitiem(self, index):
        if self.mode in ['test']:
            name = self.names[index]
            image = self.get_image(name, self.transform, mode='test')
            return image
        elif self.mode in ['valid', 'train']:
            name = self.names[index]
            label = self.label_dict[self.labels[index]]
            image = self.get_image(name, self.transform)
            return image, label, name


if __name__ == '__main__':
    print('check WhaleDataset ...')
    names_train, labels_train = [], []
    with open('../data/train_fold_0.txt', 'r') as f:
        for line in f.readlines():
            name, label = line.strip().split(' ')
            names_train.append(name)
            labels_train.append(label)
    def transform_train(image, label):
        add_ = 0
        image = cv2.resize(image, (512, 256))
        image = np.transpose(image, (2, 0, 1))
        image = image.copy().astype(np.float)
        image = torch.from_numpy(image).div(255).float()
        return image, add_

    dst_train = WhaleDataset(names_train, labels_train, mode='train', transform_train=transform_train, min_num_classes=0)
    dst_valid = WhaleTestDataset(names_train, labels_train, mode='test', transform=transform_train)
    dataloader_train = DataLoader(dst_train, shuffle=True, drop_last=True, batch_size=16, num_workers=0)
    for data in dataloader_train:
        [anchor_im, pos_im, neg_im, neg_im2], [label1, label2, label3, label4] = data
        print label1, label2, label3, label4
        print anchor_im
        cv2.imwrite('anchor.jpg', anchor_im)
        cv2.imwrite('psos.jpg', pos_im)
        cv2.imwrite('neg.jpg', neg_im)
        cv2.imwrite('neg2.jpg', neg_im2)
