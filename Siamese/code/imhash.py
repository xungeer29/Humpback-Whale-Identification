# -*- coding:utf-8 -*-
import pandas as pd
import os
from PIL import Image
import tqdm
import pickle
from imagehash import phash
import math
import numpy as np

if not os.path.exists('../data/'):
    os.makedirs('../data/')
if not os.path.exists('../log/'):
    os.makedirs('../log/')

# 后续加入playground 数据
root_featured = '/data2/shentao/DATA/Kaggle/Whale/raw/'
root_playground = '/data2/shentao/DATA/Kaggle/Whale/raw/playground/'

tagged_ftd = dict([(p,w) for _,p,w in pd.read_csv(
                  os.path.join(root_featured, 'train.csv')).to_records()])
tagged_plgd = dict([(p,w) for _,p,w in pd.read_csv(
                    os.path.join(root_playground, 'train.csv')).to_records()])
submit_ftd = [p for _,p,_ in pd.read_csv(
                    os.path.join(root_featured, 'sample_submission.csv')).to_records()]
submit_plgd = [p for _,p,_ in pd.read_csv(
                    os.path.join(root_playground, 'sample_submission.csv')).to_records()]

join_ftd = list(tagged_ftd.keys()) + submit_ftd
join_ftd_plgd = list(tagged_ftd.keys())+list(tagged_plgd.keys())+submit_ftd+submit_plgd

print('len(tagged_ftd):{} {}\n\nlen(submit_ftd):{} {}\n\nlen(join_ftd):{}\n\n\n'.format(
       len(tagged_ftd), list(tagged_ftd.items())[:5],
       len(submit_ftd), submit_ftd[:5], 
       len(join_ftd)))
print('len(tagged_plgd):{} {}\n\nlen(submit_plgd):{} {}\n\nlen(join_ftd_plgd):{}\n\n\n'.format(
       len(tagged_plgd), list(tagged_plgd.items())[:5],
       len(submit_plgd), submit_plgd[:5], 
       len(join_ftd_plgd)))

# 返回图像的绝对路径
def expand_path(imgname):
    if os.path.isfile(os.path.join(root_featured, 'train', imgname)): 
        return os.path.join(root_featured, 'train', imgname)
    if os.path.isfile(os.path.join(root_featured, 'test', imgname)): 
        return os.path.join(root_featured, 'test', imgname)
    if os.path.isfile(os.path.join(root_playdround, 'train', imgname)): 
        return os.path.join(root_playground, 'train', imgname)
    if os.path.isfile(os.path.join(root_playground, 'test', imgname)): 
        return os.path.join(root_playground, 'test', imgname)

    return imgname

# 将图像大小保存在dict中 {imgname:imgSize, ...}
if os.path.isfile('../data/pic2size.pickle'):
    with open('../data/pic2size.pickle', 'rb') as f:
        pic2size = pickle.load(f)
else:
    pic2size = {}
    for pic in tqdm.tqdm(join_ftd):
        size = Image.open(expand_path(pic)).size
        pic2size[pic] = size
    with open('../data/pic2size.pickle', 'wb') as f:
        pickle.dump(pic2size, f)
print('len(pic2size):{}, list(pic2size.items())[:5]:{}'.format(len(pic2size), list(pic2size.items())[:5]))

# 读取每张图像的hash/将每张图像的hash保存在pickle中
if os.path.isfile('../data/pic2hash.pickle'):
    with open('../data/pic2hash.pickle', 'rb') as f:
        pic2hash = pickle.load(f)
else:
    pic2hash = {}
    for pic in tqdm.tqdm(join_ftd):
        im = Image.open(expand_path(pic))
        imhash = phash(im)
        pic2hash[pic] = imhash
    with open('../data/pic2hash.pickle', 'wb') as f:
        pickle.dump(pic2hash, f)

# 查找与给定hash值相同的图像 {(hash1:pic1,pic2,...), ...}
hash2pics = {}
for pic, imhash in pic2hash.items():
    if imhash not in hash2pics:
        hash2pics[imhash] = []
    if pic not in hash2pics[imhash]:
        hash2pics[imhash].append(pic)

# 图像归一化 均值为0,方差为1
def standardization(im):
    im = np.array(im)
    im = im-im.mean()
    im = im/math.sqrt((im**2).mean())

    return im

# 判断图像是否相同，相同指标为：
# 1. 两张图像的颜色空间相同并且图像大小相同
# 2. 图像归一化处理后，两者的均方误差不超过0.1
def match(hash1, hash2):
    for pic1 in hash2pics[hash1]:
        for pic2 in hash2pics[hash2]:
            im1 = Image.open(expand_path(pic1))
            im2 = Image.open(expand_path(pic2))
            if im1.mode != im2.mode or im1.size != im2.size:
                return False

            a1 = standardization(im1)
            a2 = standardization(im2)
            mse = ((a1-a2)**2).mean()
            if mse > 0.1:
                return False
    return True


# 关联所有相似图像的hash值:
# 相似指标: 
# 1.hash值之差小于6
# 2.图像颜色空间相同并且图像大小相同
# 3.图像零均值标准化后的均方误差小于0.1

# 所有不同的hash值
hashes = list(hash2pics.keys())
hash2hash = {}
for i, hash1 in enumerate(tqdm.tqdm(hashes)):
    for hash2 in hashes[:i]:
        if hash1-hash2 <= 6 and match(hash1, hash2):
            s1, s2 = str(hash1), str(hash2)
            if s1 < s2:
                s1, s2 = s2, s1
            hash2hash[s1] = s2

# 将相同hash的图像组合在一起，并用字符串格式的phash替换, 更快，更可读
for pic, hash_ in pic2hash.items():
    hash_ = str(hash_)
    if hash_ in hash2hash:
        hash_ = hash2hash[hash_]
    pic2hash[pic] = hash_

print('len(pic2hash):{}, pic2hash:{}'.format(len(pic2hash), list(pic2hash.items())[:5]))

# 对于每个图像ID，生成图像列表
hash2pics = {}
for pic, hash_ in pic2hash.items():
    if hash_ not in hash2pics:
        hash2pics[hash_] = []
    if pic not in hash2pics[hash_]:
        hash2pics[hash_].append(pic)
print('len(hash2pics):{}\nhash2pics:{}'.format(len(hash2pics), list(hash2pics.items())[:5]))

# 将重复图像提取出来查看是否正确
repeatpath = '../log/repeat_img/'
if not os.path.exists(repeatpath):
    os.makedirs(repeatpath)
for hash_, pics in hash2pics.items():
    if len(pics) > 2:
        print('Repeat images:{}'.format(pics))
        for pic in pics:
            im = Image.open(expand_path(pic))
            im.save(repeatpath+pic)

# 对于每个ID，选择分辨率最高的图像作为首选图像
def perfer(pics):
    if len(pics) == 1:
        return pics[0]
    else:
        best_pic = pics[0]
        best_size = pic2size[best_pic]
        for i in range(1, len(pics)):
            size = pic2size[pics[i]]
            if size[0]*size[1] > best_size[0]*best_size[1]:
                best_pic = pics[i]
                best_size = size
        return best_pic
hash2pic = {}
for hash_, pics in hash2pics.items():
    hash2pic[hash_] = perfer(pics)
with open('../data/hash2pic.pickle', 'wb') as f:
    pickle.dump(hash2pic, f)
print('len(hash2pic):{}\nhash2pic:{}'.format(len(hash2pic), list(hash2pic.items())[:5]))

with open('../')
