# -*- coding: utf-8 -*-
# @File    : calc_bdry_score.py
# @Time    : 25/02/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm


import os
import sys
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skimage.io

from utils_occlusion import mask2polygon, calc_bdry_score, get_ax
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.utils import expand_mask, resize_image
from mrcnn import visualize
matplotlib.use('tkagg')

image_dir = '../../datasets/dataset_occluded/images'

sample_info = json.load(open('sample_mask_info.json', 'r'))
N = len(sample_info)

ids = [i['id'] for i in sample_info]
file_names = [i['file_names'] for i in sample_info]
mask_true = []
bbox_true = []
mask_pred = []
bbox_pred = []


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

    # predicted
    bbox = sample_info[i]['bbox_pred']
    bbox = np.expand_dims(bbox, 0).astype('int')

    seg = sample_info[i]['mask_pred']
    seg = np.expand_dims(seg, -1)
    image_shape = (int(sample_info[i]['height']), int(sample_info[i]['width']))
    seg = expand_mask(bbox, seg, image_shape)

    bbox_pred.append(bbox[0])
    mask_pred.append(seg[:, :, 0])

mask_true = np.array(mask_true, dtype=np.bool)
mask_pred = np.array(mask_pred, dtype=np.bool)

polygon_true, _ = mask2polygon(mask_true, concat_verts=True)
polygon_pred, _ = mask2polygon(mask_pred, concat_verts=True)


bdry_scores = calc_bdry_score(polygon_true, polygon_pred)

for i in range(N):
    if bdry_scores[i] > 0.1:
        continue
    image = skimage.io.imread(os.path.join(image_dir, file_names[i]))
    image, _, _, _, _ = resize_image(image, min_dim=800, min_scale=0,
                                     max_dim=1024, mode='square')
    # # plot gt masks
    # visualize.display_instances(image, np.expand_dims(bbox_true[i], 0),
    #                             np.expand_dims(mask_true[i], -1),
    #                             np.array([ids[i]]), '',
    #                             figsize=(6, 6),
    #                             captions=['gt mask'],
    #                             title='GT')
    #
    # # plot predicted masks
    # visualize.display_instances(image, np.expand_dims(bbox_pred[i], 0),
    #                             np.expand_dims(mask_pred[i], -1),
    #                             np.array([ids[i]]), '',
    #                             figsize=(6, 6),
    #                             captions=['predicted mask'],
    #                             title='bdry_score = %.5f' % bdry_scores[i])

    # plot gt polygons
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # axes = get_ax(1, 2, size=6)
    axes[0].imshow(image)
    verts = polygon_true[i]
    verts = verts[::10]  # make points less dense
    xs, ys = zip(*verts)
    axes[0].scatter(xs, ys, c='r', s=1)
    axes[0].axis('off')
    axes[0].set_title('GT')

    # plot predicted polygons
    axes[1].imshow(image)
    verts = polygon_pred[i]
    verts = verts[::10]  # make points less dense
    xs, ys = zip(*verts)
    axes[1].scatter(xs, ys, c='r', s=1)
    axes[1].axis('off')
    axes[1].set_title('Predicted')
    fig.suptitle('bdry_score = %.5f' % bdry_scores[i])

    # if i == 4:  # only show first two
    #     break
