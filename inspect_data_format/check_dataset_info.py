# -*- coding: utf-8 -*-
# @File    : check_dataset_info.py
# @Time    : 31/01/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath("../")

sys.path.append(ROOT_DIR)
from occlusion import occlusion

dataset_dir = '../../datasets/dataset_occluded'
# lists_dir = os.path.join(dataset_dir, 'lists')
#
path = os.path.join(dataset_dir, 'occlusion_short.json')
image_info = json.load(open(path))

# class_ids = list(range(1, 13))
class_ids = None
dataset = occlusion.OcclusionDataset()
occlusion = dataset.load_occlusion(dataset_dir, "short", return_occlusion=True,
                                   class_ids=class_ids)
dataset.prepare()
