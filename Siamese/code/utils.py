# -*- coding:utf-8 -*-

import gzip
import pickle
import os
import random
import sys
import pandas as pd
import time
import scipy

from lap import lapjv
from math import sqrt
from os.path import isfile
from tqdm import tqdm

from config import *

"""
将图像名拓展为绝对路径
input: 图像名
return: 图像绝对路径
"""
def expand_path(p):
    if isfile(TRAIN + p):
        return TRAIN + p
    if isfile(TEST + p):
        return TEST + p
    return p
