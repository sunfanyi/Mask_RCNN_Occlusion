# -*- coding: utf-8 -*-
# @File    : display_image.py
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
# import skimage.io
# from skimage import measure
# from skimage.draw import polygon
# from pycocotools import mask as maskUtils
# from make_json_occlusion import add_image_to_list

import occlusion

ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)
from mrcnn import visualize, utils
from mrcnn.model import log
# from mrcnn.visualize import display_images, draw_box
# from mrcnn.utils import minimize_mask, expand_mask
import occlusion

dataset_dir = '../../datasets/dataset_occluded'

dataset = occlusion.OcclusionDataset()
occlusion = dataset.load_occlusion(dataset_dir, "train", return_occlusion=True)
dataset.prepare()


matplotlib.use('tkagg')
# n = 1
# image_ids = np.random.choice(dataset.image_ids, 2)
image_ids = [4]
for image_id in image_ids:
    # ======================== Load image ==================================
    # image_id = random.choice(dataset.image_ids)
    image = dataset.load_image(image_id)

    # for image_id in image_ids:
    images = dataset.load_image(image_id)
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


# # load image
# for image in image_info:
#     height = image['height']
#     width = image['width']
#
#     _image = skimage.io.imread(image['path'])
#     plt.figure()
#     plt.title("H x W={}x{}".format(height, width))
#     plt.axis('off')
#     plt.imshow(_image.astype(np.uint8))
#     plt.show()
#
#     ann = image['annotations'][0]
#     segm = ann['segmentation']
#     mask = segm2mask(segm, height, width)
#     plt.figure()
#     plt.title('segmentation:' + image['annotations'][0]['category_name'])
#     plt.axis('off')
#     plt.imshow(mask.astype(np.uint8))
#     plt.show()
#
#     segm = ann['occluder_mask']
#     occluder_mask = segm2mask(segm, height, width)
#     plt.figure()
#     plt.title('occluder_mask:' + image['annotations'][0]['category_name'])
#     plt.axis('off')
#     plt.imshow(occluder_mask.astype(np.uint8))
#     plt.show()
#
#     box = [int(i) for i in ann['bbox']]
#     _image_temp = draw_box(_image, box, np.array([255, 0, 0]))
#     plt.figure()
#     plt.title("bbox")
#     plt.axis('off')
#     plt.imshow(_image_temp.astype(np.uint8))
#     plt.show()
#
#     _image_temp = _image.copy()
#     for occluder_box in ann['occluder_box']:
#         occluder_box = [int(i) for i in occluder_box]
#         _image_temp = draw_box(_image, occluder_box, np.array([255, 0, 0]))
#     plt.figure()
#     plt.title("occluder_box")
#     plt.axis('off')
#     plt.imshow(_image_temp.astype(np.uint8))
#     plt.show()
#
#     mini_mask = minimize_mask([box], np.expand_dims(mask, axis=-1), (56, 56))
#     plt.figure()
#     plt.title('mini_mask:' + image['annotations'][0]['category_name'])
#     plt.axis('off')
#     plt.imshow(mini_mask[:, :, 0].astype(np.uint8))
#     plt.show()
#
#     mask_expanded = expand_mask([box], mini_mask, (height, width))
#     plt.figure()
#     plt.title('expanded_mask:' + image['annotations'][0]['category_name'])
#     plt.axis('off')
#     plt.imshow(mask_expanded[:, :, 0].astype(np.uint8))
#     plt.show()
#     break
