# -*- coding:utf-8 -*-

"""
build 4 fold cross validation.
只有一张图像的保留，剩下的抽取 512 ID，抽取一张图像，new_whale中抽取230张
总计 512+230=742
"""

import os
import random
import pandas as pd
from tqdm import tqdm

root = '/media/gfx/data1/DATA/Kaggle/whale'

train_dir = os.path.join(root, 'featured/train.csv')
train_df = pd.read_csv(train_dir)
id2ims = {}
for im, id in zip(train_df['Image'], train_df['Id']):
    if id not in id2ims.keys():
        id2ims[id] = []
    id2ims[id].append(im)

for i in tqdm(range(5)):
    f_train = open('../data/train_fold_{}.txt'.format(i), 'w')
    f_val = open('../data/val_fold_{}.txt'.format(i), 'w')
    ids = []
    for id in id2ims.keys():
        if len(id2ims[id]) == 1 or id == 'new_whale':
            continue
        ids.append(id)
    val_im = []
    ids_val = []
    for id in random.sample(ids, 800):
        if len(ids_val) > 512:
            break
        if id in ids_val:
            continue
        ids_val.append(id)
        idx = random.randint(0, len(id2ims[id])-1)
        f_val.write('{} {}\n'.format(id2ims[id][idx], id))
        val_im.append(id2ims[id][idx])
    for im in random.sample(id2ims['new_whale'], 230):
        f_val.write('{} {}\n'.format(im, 'new_whale'))
        val_im.append(im)
    for im, id in zip(train_df['Image'], train_df['Id']):
        if im not in val_im:
            f_train.write('{} {}\n'.format(im, id))
    f_train.close()
    f_val.close()

'''
create whale2id
'''
f_w2id = open('../data/whale2id.txt', 'w')
whales = []
id = 0
f_w2id.write('new_whale -1\n')
for whale in tqdm(train_df['Id']):
    if whale == 'new_whale':
        continue
    if whale not in whales:
        f_w2id.write('{} {}\n'.format(whale, len(whales)))
        whales.append(whale)
f_w2id.close()
