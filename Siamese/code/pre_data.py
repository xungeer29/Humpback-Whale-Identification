# -*- coding:utf-8 -*-

import os
import pandas as pd
import random
import pickle
from tqdm import tqdm

root = '/data2/shentao/DATA/Kaggle/Whale/raw/'

trainDir = 'train.csv'

if not os.path.exists('../data/'):
    os.makedirs('../data/')
#savetrain = open('../data/train.txt', 'w')
#saveval = open('../data/val.txt', 'w')
#savetest = open('../data/test.txt', 'w')
savefull = open('../data/train_full.txt', 'w')
#savetrain_without_new_whale = open('../data/train_without_new_whale.txt', 'w')
#saveval_without_new_whale = open('../data/val_without_new_whale.txt', 'w')
#savefull_without_new_whale = open('../data/train_full_without_new_whale.txt', 'w')

f = open('../data/hash2pic.pickle', 'rb')
hash2pic = pickle.load(f)
pics_cleaned = []
for hash_, pic in list(hash2pic.items()):
    #print pic
    pics_cleaned.append(pic)

# train.csv
tagged_ftd = dict([(p,w) for _,p,w in pd.read_csv(os.path.join(root, trainDir)).to_records()])
pics, whales = [], []
for pic, whale in list(tagged_ftd.items()):
    pics.append(pic)
    whales.append(whale)
    #print(pic, whale)
#print pics

# 将去重后的图像与鲸鱼名对应
tagged_ftd_cleaned = {}
for i in range(len(pics)):
    if pics[i] in pics_cleaned:
        tagged_ftd_cleaned[pics[i]] = whales[i]

# 每个鲸鱼对应的图像
whale2pics = {}
for p, w in tagged_ftd_cleaned.items():
    if w not in whale2pics:
        whale2pics[w] = []
    if p not in whale2pics[w]:
        whale2pics[w].append(p)

# 生成id
ids_no_repeat = []
num = 0
for p, w in list(tagged_ftd_cleaned.items()):
    if w == 'new_whale':
        savefull.write('train/{} {}\n'.format(p, -1))
    elif w not in ids_no_repeat:
        savefull.write('train/{} {}\n'.format(p, len(ids_no_repeat)))
        ids_no_repeat.append(w)

# whale id number
with open('../data/whale_id_num.txt', 'w') as f:
    for i in range(len(ids_no_repeat)):
        f.write('{} {} {}\n'.format(ids_no_repeat[i], i, len(whale2pics[ids_no_repeat[i]])))


