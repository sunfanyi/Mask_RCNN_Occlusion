# -*- coding: utf-8 -*-
# @File    : try.py
# @Time    : 23/03/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

import tensorflow as tf
from skimage.measure import find_contours
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import skimage.io

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.utils import expand_mask, resize_image
from mrcnn.utils_occlusion import mask2polygon

image_dir = '../../datasets/dataset_occluded/images'

sample_info = json.load(open('sample_mask_info.json', 'r'))
N = len(sample_info)

ids = [i['id'] for i in sample_info]
file_names = [i['file_names'] for i in sample_info]
mask_true = []
bbox_true = []

for i in range(N):
    # GT
    bbox = sample_info[i]['bbox_true']
    bbox = np.expand_dims(bbox, 0).astype('int')

    seg = sample_info[i]['mask_true']
    seg = np.expand_dims(seg, -1)
    image_shape = (int(sample_info[i]['height']), int(sample_info[i]['width']))
    seg = expand_mask(bbox, seg, image_shape)

    bbox_true.append(bbox[0])
    mask_true.append(seg[:, :, 0])
    if i == 1:
        break

mask_true = np.array(mask_true, dtype=np.bool)
polygons = []
for each in mask_true:
    poly = np.argwhere(each)
    poly = np.fliplr(poly)
    polygons.append(poly)



# plot to check if any loss in accuracy
for i in range(len(mask_true)):
    image = skimage.io.imread(os.path.join(image_dir, file_names[i]))
    image, _, _, _, _ = resize_image(image, min_dim=800, min_scale=0,
                                     max_dim=1024, mode='square')

    # plot gt polygons
    fig, ax = plt.subplots()
    # axes = get_ax(1, 2, size=6)
    ax.imshow(image)
    verts = polygons[i]
    # verts = verts[::10]  # make points less dense
    xs, ys = zip(*verts)
    ax.scatter(xs, ys, c='r', s=1)
    ax.axis('off')
    ax.set_title('GT')
    fig.show()

