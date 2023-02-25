# -*- coding: utf-8 -*-
# @File    : display_gt.py
# @Time    : 31/01/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm


import os
import sys
import json
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt


ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)
from mrcnn import visualize, utils
from mrcnn.model import log
# from mrcnn.visualize import display_images, draw_box
# from mrcnn.utils import minimize_mask, expand_mask
import occlusion

dataset_dir = '../../datasets/dataset_occluded'

dataset = occlusion.OcclusionDataset()
occlusion = dataset.load_occlusion(dataset_dir, "test", return_occlusion=True)
dataset.prepare()

target_id = "aeroplaneFGL1_BGL1/n02690373_3378"
for i in range(len(dataset.image_info)):
    if dataset.image_info[i]['id'] == target_id:
        target_idx = i
        break

matplotlib.use('tkagg')
# n = 1
image_ids = [target_idx]
# image_ids = np.random.choice(dataset.image_ids, 1)
for image_id in image_ids:
    info = dataset.image_info[image_id]
    # ======================== Load image ==================================
    # image_id = random.choice(dataset.image_ids)
    image = dataset.load_image(image_id)
    # visualize.display_images([images], cols=1)


    # ========================= Display images and masks ===================
    mask, class_ids = dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names)
    # plt.figure()
    # plt.title('manual_mask')
    # plt.axis('off')
    # plt.imshow(mask[:, :, 0].astype(np.uint8))
    # plt.show()

    # ============================== Bounding boxes =========================
    # Load mask
    mask, class_ids = dataset.load_mask(image_id)
    # Compute Bounding box
    bbox = utils.extract_bboxes(mask)

    # Display image and additional stats
    print("image_id ", image_id, dataset.image_reference(image_id))
    log("image", image)
    log("mask", mask)
    log("class_ids", class_ids)
    log("bbox", bbox)
    # Display image and instances
    visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)


