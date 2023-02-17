# -*- coding: utf-8 -*-
# @File    : demo_occlusion.py
# @Time    : 01/02/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
# import skimage.io
# from skimage import measure
# from skimage.draw import polygon
# from pycocotools import mask as maskUtils
# from make_json_occlusion import add_image_to_list

import occlusion_depreciated

ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)
# from mrcnn import visualize
# from mrcnn.visualize import display_images, draw_box
# from mrcnn.utils import minimize_mask, expand_mask

dataset_dir = '../../datasets/dataset_occluded'


# take data from json files
annotation_file = os.path.join(dataset_dir, 'occlusion_coco_format_short.json')
dataset = json.load(open(annotation_file, 'r'))

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