# -*- coding:utf-8 -*-
# Read the dataset description
import gzip
# Read or generate p2h, a dictionary of image name to image id (picture to hash)
import pickle
import platform
import random
# Suppress annoying stderr output when importing keras.
import sys
from lap import lapjv
from math import sqrt
# Determine the size of each image
from os.path import isfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as pil_image
from imagehash import phash
from pandas import read_csv
from scipy.ndimage import affine_transform
from tqdm import tqdm
import time

from config import *

tagged = dict([(p, w) for _, p, w in read_csv(TRAIN_DF).to_records()])
submit = [p for _, p, _ in read_csv(SUB_Df).to_records()]
join = list(tagged.keys()) + submit

def expand_path(p):
    if isfile(TRAIN + p):
        return TRAIN + p
    if isfile(TEST + p):
        return TEST + p
    return p

print('p2size:')
if isfile(P2SIZE):
    print("P2SIZE exists.")
    with open(P2SIZE, 'rb') as f:
        p2size = pickle.load(f)
    print('loaded p2size.pickle\n')
else:
    p2size = {}
    for p in tqdm(join):
        size = pil_image.open(expand_path(p)).size
        p2size[p] = size
    with open(P2SIZE, 'wb') as f:
        pickle.dump(p2size, f)
    print('finish p2size\n')

def match(h1, h2):
    for p1 in h2ps[h1]:
        for p2 in h2ps[h2]:
            i1 = pil_image.open(expand_path(p1))
            i2 = pil_image.open(expand_path(p2))
            if i1.mode != i2.mode or i1.size != i2.size: return False
            a1 = np.array(i1)
            a1 = a1 - a1.mean()
            a1 = a1 / sqrt((a1 ** 2).mean())
            a2 = np.array(i2)
            a2 = a2 - a2.mean()
            a2 = a2 / sqrt((a2 ** 2).mean())
            a = ((a1 - a2) ** 2).mean()
            if a > 0.1: return False
    return True

print('p2h:')
if isfile(P2H):
    print("P2H exists.")
    with open(P2H, 'rb') as f:
        p2h = pickle.load(f)
    print('loaded p2h.pickle\n')
else:
    # Compute phash for each image in the training and test set.
    p2h = {}
    for p in tqdm(join):
        img = pil_image.open(expand_path(p))
        h = phash(img)
        p2h[p] = h

    # Find all images associated with a given phash value.
    h2ps = {}
    for p, h in p2h.items():
        if h not in h2ps: h2ps[h] = []
        if p not in h2ps[h]: h2ps[h].append(p)

    # Find all distinct phash values
    hs = list(h2ps.keys())

    # If the images are close enough, associate the two phash values (this is the slow part: n^2 algorithm)
    h2h = {}
    for i, h1 in enumerate(tqdm(hs)):
        for h2 in hs[:i]:
            if h1 - h2 <= 6 and match(h1, h2):
                s1 = str(h1)
                s2 = str(h2)
                if s1 < s2: s1, s2 = s2, s1
                h2h[s1] = s2

    # Group together images with equivalent phash, and replace by string format of phash (faster and more readable)
    for p, h in p2h.items():
        h = str(h)
        if h in h2h: h = h2h[h]
        p2h[p] = h
    with open(P2H, 'wb') as f:
        pickle.dump(p2h, f)
# For each image id, determine the list of pictures
print('h2ps:')
if isfile('../data/h2ps.pickle'):
    with open('../data/h2ps.pickle', 'rb') as f:
        h2ps = pickle.load(f)
    print('loaded h2ps\n')
else:
    h2ps = {}
    for p, h in tqdm(p2h.items()):
        if h not in h2ps: h2ps[h] = []
        if p not in h2ps[h]: h2ps[h].append(p)
    with open('../data/h2ps.pickle', 'wb') as f:
        pickle.dump(h2ps, f)
    print('finish h2ps\n')

def show_whale(imgs, per_row=2):
    n = len(imgs)
    rows = (n + per_row - 1) // per_row
    cols = min(per_row, n)
    fig, axes = plt.subplots(rows, cols, figsize=(24 // per_row * cols, 24 // per_row * rows))
    for ax in axes.flatten(): ax.axis('off')
    for i, (img, ax) in enumerate(zip(imgs, axes.flatten())): ax.imshow(img.convert('RGB'))
        

def read_raw_image(p):
    img = pil_image.open(expand_path(p))
    return img

# For each images id, select the prefered image
def prefer(ps):
    if len(ps) == 1: return ps[0]
    best_p = ps[0]
    best_s = p2size[best_p]
    for i in range(1, len(ps)):
        p = ps[i]
        s = p2size[p]
        if s[0] * s[1] > best_s[0] * best_s[1]:  # Select the image with highest resolution
            best_p = p
            best_s = s
    return best_p

print('h2p:\n')
if isfile('../data/h2p.pickle'):
    with open('../data/h2p.pickle', 'rb') as f:
        h2p = pickle.load(f)
    print('loaded h2p\n')
else:
    h2p = {}
    for h, ps in h2ps.items():
        h2p[h] = prefer(ps)
    with open('../data/h2p.pickle', 'wb') as f:
        pickle.dump(h2p, f)
    print('finish h2p\n')
#len(h2p), list(h2p.items())[:5]

print('h2ws\n')
if isfile('../data/h2ws.pickle'):
    with open('../data/h2ws.pickle', 'rb') as f:
        h2ws = pickle.load(f)
    print('loaded h2ws\n')
else:
    h2ws = {}
    new_whale = 'new_whale'
    for p, w in tagged.items():
        if w != new_whale:  # Use only identified whales
            h = p2h[p]
            if h not in h2ws: h2ws[h] = []
            if w not in h2ws[h]: h2ws[h].append(w)
    for h, ws in h2ws.items():
        if len(ws) > 1:
            h2ws[h] = sorted(ws)
    with open('../data/h2ws.pickle', 'wb') as f:
        pickle.dump(h2ws, f)
    print('finish h2ws\n')

# For each whale, find the unambiguous images ids.
print('w2hs:\n')
if isfile('../data/w2hs.pickle'):
    with open('../data/w2hs.pickle', 'rb') as f:
        w2hs = pickle.load(f)
    print('loaded w2hs.\n')
else:
    w2hs = {}
    for h, ws in h2ws.items():
        if len(ws) == 1:  # Use only unambiguous pictures
            w = ws[0]
            if w not in w2hs: w2hs[w] = []
            if h not in w2hs[w]: w2hs[w].append(h)
    for w, hs in w2hs.items():
        if len(hs) > 1:
            w2hs[w] = sorted(hs)
    with open('../data/w2hs.pickle', 'wb') as f:
        pickle.dump(w2hs, f)
    print('finish w2hs.\n')


