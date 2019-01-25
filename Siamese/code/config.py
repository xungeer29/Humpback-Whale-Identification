# -*- coding:utf-8 -*-

TRAIN_DF = '/data2/shentao/DATA/Kaggle/Whale/raw/train.csv'
SUB_Df = '/data2/shentao/DATA/Kaggle/Whale/raw/sample_submission.csv'
TRAIN = '/data2/shentao/DATA/Kaggle/Whale/raw/train/'
TEST = '/data2/shentao/DATA/Kaggle/Whale/raw/test/'
P2H = '../data/p2h.pickle'
P2SIZE = '../data/p2size.pickle'
BB_DF = "/data2/gaofuxun/liveness/whale/Siamese/data/bounding_boxes.csv"

anisotropy = 2.15 # 图像的水平垂直压缩比
crop_margin = 0.05 # bbox crop 的裕度
HEIGHT = 384
WIDTH = 384
CHANNEL = 1
