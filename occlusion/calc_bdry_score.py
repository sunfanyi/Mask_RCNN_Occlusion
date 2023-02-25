# -*- coding: utf-8 -*-
# @File    : calc_bdry_score.py
# @Time    : 25/02/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm


import os
import sys
import random
import math
import re
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import matplotlib.patches as patches

import occlusion
from utils_occlusion import *

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from skimage.measure import find_contours
from matplotlib.patches import Polygon
from pycococreatortools.pycococreatortools import binary_mask_to_polygon

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

MODEL_PATH = r"D:\Desktop\FYP - Robust Surgical Tool Detection and Occlusion Handling using Deep Learning\Mask_RCNN-Occulusion\logs\train_004_120_m351\mask_rcnn_occlusion_0120.h5"

matplotlib.use('tkagg')

config = occlusion.OcclusionConfig()
dataset_DIR = '../../datasets/dataset_occluded'


# ====================== Get gt mask and polygon =========================
dataset = occlusion.OcclusionDataset()
dataset.load_occlusion(dataset_DIR, "test")
dataset.prepare()
print("Images: {}\nClasses: {}".format(len(dataset.image_ids),
                                       dataset.class_names))
# bottleFGL1_BGL1/n02823428_2278, carFGL1_BGL1/n02814533_3155
# aeroplaneFGL1_BGL1/n02690373_3378 bug
target_id = "aeroplaneFGL1_BGL1/n02690373_3378"
for i in range(len(dataset.image_info)):
    if dataset.image_info[i]['id'] == target_id:
        target_idx = i
        break
image_id = target_idx
image_id = np.random.choice(dataset.image_ids, 1)[0]

image, image_meta, gt_class_id, gt_bbox, gt_mask = \
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                       dataset.image_reference(image_id)))
visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id,
                            dataset.class_names, title="GT")

ax = get_ax(1)

ax.imshow(image)

mask = gt_mask
# Mask Polygon
# Pad to ensure proper polygons for masks that touch image edges.
padded_mask = np.zeros(
    (mask.shape[0] + 2, mask.shape[1] + 2, mask.shape[2]), dtype=np.uint8)
padded_mask[1:-1, 1:-1, :] = mask

contours_flat = []  # 2n x 1
contours_xy = []  # n x 2

for i in range(mask.shape[2]):  # for each segmentation
    contour = find_contours(padded_mask[:, :, i], 0.5)
    # contour_xy = [i[::10].tolist() for i in contour_xy]
    contour_xy = []
    i = 0
    for verts in contour:
        # Subtract the padding and flip (y, x) to (x, y)
        verts = np.fliplr(verts) - 1
        verts = verts[::10].tolist()
        # contour_xy.append(verts)  # todo
        p = Polygon(verts, facecolor="none", linewidth=2, edgecolor='r')
        # ax.add_patch(p)
        i += 1
        if i == 1:
            xs, ys = zip(*verts)
            ax.scatter(xs, ys, c='r', s=1)

    contour_xy = verts  # todo
    contours_xy.append(contour_xy)

    # flatten the 2D list
    contour_flat = [[item for sublist in contour_xy for item in sublist]]
    contours_flat.append(contour_flat)


    # contour = [binary_mask_to_polygon(padded_mask[:, :, i], tolerance=2)[2]]
    # contours.append(contour)
    #
    # contour_xy = np.array(contour).reshape(-1, 2).tolist()
    # contours_xy.append(contour_xy)

    # xs, ys = zip(*contour_xy)
    # ax.scatter(xs, ys, c='r')
plt.show()


#
# contours_flat2 = []  # 2n x 1
# contours_xy2 = []  # n x 2
#
# for i in range(mask.shape[2]):
#     contour_flat2 = binary_mask_to_polygon(padded_mask[:, :, i], tolerance=2)
#     contour_flat2 = np.array(contour_flat2) - 1
#     contours_flat2.append(contour_flat2.tolist())
#
#     contour_xy2 = np.array(contour_flat2).reshape(-1, 2).tolist()
#     contours_xy2.append(contour_xy2)
#
#     xs, ys = zip(*contour_xy2)
#     ax.scatter(xs, ys, c='r')
# plt.show()

mask = []
for i in range(len(contours_xy)):
    binary_mask = annToMask(contours_flat[i], image.shape[0], image.shape[1])
    mask.append(binary_mask)
mask = np.transpose(mask, (1, 2, 0))
_, class_ids = dataset.load_mask(image_id)
# Compute Bounding box
bbox = utils.extract_bboxes(mask)

visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)


# # Override the training configurations with a few
# # changes for inferencing.
# class InferenceConfig(config.__class__):
#     # Run detection on one image at a time
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1
#
# config = InferenceConfig()
#



# model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
#                           config=config)
#
# # Set weights file path
# weights_path = MODEL_PATH
# # Or, uncomment to load the last model you trained
# # weights_path = model.find_last()
#
# # Load weights
# print("Loading weights ", weights_path)
# model.load_weights(weights_path, by_name=True)
#
# image_id = 0
# image, image_meta, gt_class_id, gt_bbox, gt_mask = \
#     modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
# info = dataset.image_info[image_id]
# print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
#                                        dataset.image_reference(image_id)))
#
# # Run object detection
# results = model.detect([image], verbose=1)
#
# # Display results
# # ax = get_ax(1)
# _, ax = plt.subplots(figsize=(16, 16))
# r = results[0]
# visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                             dataset.class_names, r['scores'], ax=ax,
#                             title="Predictions")
# log("gt_class_id", gt_class_id)
# log("gt_bbox", gt_bbox)
# log("gt_mask", gt_mask)
#
# # Create a figure and axis object
# _, ax = plt.subplots(1, figsize=(16, 16))
#
# ax.imshow(image)
#
# mask = r['masks'][:, :, 0]
# padded_mask = np.zeros(
#     (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
# padded_mask[1:-1, 1:-1] = mask
#
# contours = binary_mask_to_polygon(padded_mask, tolerance=2)
#
# polygons = np.array(contours).reshape(-1, 2)
#
# xs, ys = zip(*polygons)
# ax.scatter(xs, ys, c='r')

