# -*- coding:utf-8 -*-

import os
import pandas as pd
import random

root = '/data2/shentao/DATA/Kaggle/Whale/raw/'

trainDir = 'train.csv'

if not os.path.exists('../data/'):
    os.makedirs('../data/')
savetrain = open('../data/train.txt', 'w')
saveval = open('../data/val.txt', 'w')
savetest = open('../data/test.txt', 'w')
savefull = open('../data/train_full.txt', 'w')
savetrain_without_new_whale = open('../data/train_without_new_whale.txt', 'w')
saveval_without_new_whale = open('../data/val_without_new_whale.txt', 'w')
savefull_without_new_whale = open('../data/train_full_without_new_whale.txt', 'w')

img_id_list = pd.read_csv(os.path.join(root, trainDir))
Id = img_id_list['Id']
Image = img_id_list['Image']
ids_no_repeat = []
num = 0
for i in range(len(Id)):
    rand = random.random()
    if Id[i] == 'new_whale':
        savefull.write('train/{} {}\n'.format(Image[i], -1))
        if rand<0.15:
            saveval.write('train/{} {}\n'.format(Image[i], -1))
        else:
            savetrain.write('train/{} {}\n'.format(Image[i], -1))
    elif Id[i] not in ids_no_repeat:
        savefull.write('train/{} {}\n'.format(Image[i], len(ids_no_repeat)))
        savefull_without_new_whale.write('train/{} {}\n'.format(Image[i], len(ids_no_repeat)))
        if rand<0.15:
            saveval.write('train/{} {}\n'.format(Image[i], len(ids_no_repeat)))
            saveval_without_new_whale.write('train/{} {}\n'.format(Image[i], len(ids_no_repeat)))
        else:
            savetrain.write('train/{} {}\n'.format(Image[i], len(ids_no_repeat)))
            savetrain_without_new_whale.write('train/{} {}\n'.format(Image[i], len(ids_no_repeat)))
        ids_no_repeat.append(Id[i])
    else:
        savefull.write('train/{} {}\n'.format(Image[i], ids_no_repeat.index(Id[i])))
        savefull_without_new_whale.write('train/{} {}\n'.format(Image[i], ids_no_repeat.index(Id[i])))
        if rand<0.15:
            saveval.write('train/{} {}\n'.format(Image[i], ids_no_repeat.index(Id[i])))
            saveval_without_new_whale.write('train/{} {}\n'.format(Image[i], ids_no_repeat.index(Id[i])))
        else:
            savetrain.write('train/{} {}\n'.format(Image[i], ids_no_repeat.index(Id[i])))
            savetrain_without_new_whale.write('train/{} {}\n'.format(Image[i], ids_no_repeat.index(Id[i])))

# create the list of whale name and one-hot label
save_name_onehot = open('../data/name_onehot.txt', 'w')
for i in range(len(ids_no_repeat)):
    save_name_onehot.write('{} {}\n'.format(ids_no_repeat[i], i))
save_name_onehot.close()

for img in os.listdir(os.path.join(root, 'test')):
    savetest.write('test/{}\n'.format(img))

print 'Number of ID: {}'.format(len(set(Id)))
