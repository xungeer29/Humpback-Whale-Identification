# -*- coding:utf-8 -*-

import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# smooth
def smooth(scalar, weight):
    last = scalar[0] 
    smoothed = []
    for point in scalar:
        if point is None:
            break
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed
    
def draw_curve(y, weight=0.9, name='loss'):
    y = smooth(y, weight)
    x = [x for x in range(len(y))]
    plt.plot(x, y, color='b')
    plt.xlabel('Iterations')
    plt.ylabel(name)
    plt.savefig('../figs/{}.jpg'.format(name))
    plt.close()
