# -*- coding: utf-8 -*-
# @File    : inspect_format_balloon.py
# @Time    : 29/01/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

import os
import json
import skimage.io
import skimage.draw
import numpy as np

dataset_dir = 'D:\Desktop\FYP - Robust Surgical Tool Detection and Occlusion Handling using Deep Learning\datasets'
path_balloon = os.path.join(dataset_dir,
                            'balloon/val/via_region_data_investigate.json')
balloon = json.load(open(path_balloon))
balloon = list(balloon.values())

annotations = [a for a in balloon if a['regions']]
dataset_dir = os.path.join(dataset_dir, 'balloon/val')

# ============================== 1. load_balloon() =======================
image_info = []
# Add images
for a in annotations:
    # Get the mask
    if type(a['regions']) is dict:
        polygons = [r['shape_attributes'] for r in a['regions'].values()]
    else:
        polygons = [r['shape_attributes'] for r in a['regions']]

    # add image size
    image_path = os.path.join(dataset_dir, a['filename'])
    image = skimage.io.imread(image_path)
    height, width = image.shape[:2]

    image_id = a['filename']
    source = 'balloon'
    path = image_path
    image_info_dic = {
        "id": image_id,
        "source": source,
        "path": path,
    }
    image_info_dic.update(width=width, height=height,
                          polygons=polygons)
    image_info.append(image_info_dic)


# =========================== 2. load_mask() ==========================
# we choose the second image as an example
image_id = 1
image2_info = image_info[image_id]

# Convert polygons to a bitmap mask of shape
# [height, width, instance_count]
info = image_info[image_id]
mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                dtype=np.uint8)
for i, p in enumerate(info["polygons"]):
    # Get indexes of pixels inside the polygon and set them to 1
    rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
    mask[rr, cc, i] = 1

# Return mask, and array of class IDs of each instance. Since we have
# one class ID only, we return an array of 1s
res_mask = mask.astype(np.bool)
res2 = np.ones([mask.shape[-1]], dtype=np.int32)
