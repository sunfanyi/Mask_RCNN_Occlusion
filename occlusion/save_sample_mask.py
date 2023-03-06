# -*- coding: utf-8 -*-
# @File    : save_sample_mask.py
# @Time    : 05/03/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

"""
Run detection and save sample masks in json file. So there is no need to load
model and run detection in later development, which saves time.
"""

import os
import sys
import json
import numpy as np

import occlusion

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

MODEL_PATH = r"D:\Desktop\FYP - Robust Surgical Tool Detection and Occlusion Handling using Deep Learning\Mask_RCNN-Occulusion\logs\train_004_120_m351\mask_rcnn_occlusion_0120.h5"

config = occlusion.OcclusionConfig()
dataset_DIR = '../../datasets/dataset_occluded'

sample_info = []

# ==================== Get gt mask ========================
dataset = occlusion.OcclusionDataset()
dataset.load_occlusion(dataset_DIR, "test")
dataset.prepare()

# bottleFGL1_BGL1/n02823428_2278, carFGL1_BGL1/n02814533_3155
# aeroplaneFGL1_BGL1/n02690373_3378 trivial case
target_id_1 = "aeroplaneFGL1_BGL1/n02690373_3378"
for i in range(len(dataset.image_info)):
    if dataset.image_info[i]['id'] == target_id_1:
        target_idx1 = i
        break
target_id_2 = "bottleFGL1_BGL1/n02823428_2278"
for i in range(len(dataset.image_info)):
    if dataset.image_info[i]['id'] == target_id_2:
        target_idx2 = i
        break

image_ids = [target_idx1, target_idx2]

for i in range(len(image_ids)):
    image_id = image_ids[i]
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)

    gt_mask = np.transpose(gt_mask, (2, 0, 1))
    sample_info.append({'image_id': float(image_id),
                        'image': image.astype('float').tolist(),
                        'mask_gt': gt_mask.astype('float').tolist()})


# ==================== Get predicted mask ========================
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                          config=config)

weights_path = MODEL_PATH
model.load_weights(weights_path, by_name=True)

for i in range(len(sample_info)):
    image = np.array(sample_info[i]['image'])
    results = model.detect([image], verbose=1)

    r = results[0]
    mask_pred = r['masks']
    mask_pred = np.transpose(mask_pred, (2, 0, 1))
    sample_info[i].update({'mask_pred': mask_pred.astype('float').tolist()})


# sample_info = {'sample_info': sample_info}
with open('sample_mask_info.json', "w") as outfile:
    json.dump(sample_info, outfile)
