# -*- coding: utf-8 -*-
# @File    : check_dataset_info.py
# @Time    : 31/01/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

import os
import sys
import json

ROOT_DIR = os.path.abspath("../")

sys.path.append(ROOT_DIR)
from occlusion import occlusion

dataset_dir = '../../datasets/dataset_occluded'
# lists_dir = os.path.join(dataset_dir, 'lists')
#
path = os.path.join(dataset_dir, 'jsons_my_annos', 'occlusion_train.json')
image_info = json.load(open(path))

config = occlusion.OcclusionConfig()
# config.NUM_CLASSES -= 1
# config.__init__()

config.display()

# class_ids = list(range(1, 13))
class_ids = None
dataset_val = occlusion.OcclusionDataset()
occ = dataset_val.load_occlusion(dataset_dir, "val", return_occlusion=True,
                             class_ids=class_ids)
dataset_val.prepare()

dataset_train = occlusion.OcclusionDataset()
occ = dataset_train.load_occlusion(dataset_dir, "train", return_occlusion=True,
                             class_ids=class_ids)
dataset_train.prepare()
