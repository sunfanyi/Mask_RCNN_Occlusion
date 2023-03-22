# -*- coding: utf-8 -*-
# @File    : utils_occlusion.py
# @Time    : 25/02/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# Tools for this study

import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.measure import find_contours

from pycocotools import mask as maskUtils


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def annToMask(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(ann, height, width)
    m = maskUtils.decode(rle)
    return m


def annToRLE(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(segm['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, height, width)
    else:
        # rle
        rle = ann
    return rle


def mask2polygon(mask, ratio=1, concat_verts=False):
    """
    :param mask: [ROIs, height, width]
    :param ratio: point density, = 10 if choose every ten points
    :param concat_verts: if true, concat the vertices for each segmentation into
        a single list
    """
    # Pad to ensure proper polygons for masks that touch image edges.
    # print(mask)
    # print(mask.shape)
    padded_mask = np.zeros(
        (mask.shape[0], mask.shape[1] + 2, mask.shape[2] + 2), dtype=np.uint8)
    # padded_mask = tf.zeros(
    #     (mask.shape[0], mask.shape[1] + 2, mask.shape[2] + 2), dtype=mask.dtype)
    padded_mask[:, 1:-1, 1:-1] = mask

    contours_flat = []  # 2n x 1
    contours_xy = []  # n x 2


    for i in range(mask.shape[0]):  # for each segmentation
        contour = find_contours(padded_mask[i, :, :], 0.5)
        xy = []  # multiple verts (in xy)
        flat = []  # multiple verts (in flat)
        for verts in contour:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            verts = verts[::ratio]  # make points less dense
            xy.append(verts.tolist())
            # flatten the 2D list
            flat.append([item for sublist in verts for item in sublist])

        contours_xy.append(xy)
        contours_flat.append(flat)
    if concat_verts:
        contours_xy = [sum(i, []) for i in contours_xy]
        contours_flat = [sum(i, []) for i in contours_flat]

    return contours_xy, contours_flat


def calc_bdry_score(polygons_true, polygons_pred):
    if len(polygons_true) != len(polygons_pred):
        raise ValueError('Different numbers of segmentations')

    N = len(polygons_true)
    bdry_score = np.ndarray((N,))
    for i in range(N):
        true = np.array(polygons_true[i])
        pred = np.array(polygons_pred[i])
        # search for the cloest point
        diff = pred[:, np.newaxis, :] - true[np.newaxis, :, :]
        all_dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        min_dist = np.min(all_dist, axis=1)  # [num_points]
        bdry_score[i] = np.mean(min_dist)/len(min_dist)

    return bdry_score
