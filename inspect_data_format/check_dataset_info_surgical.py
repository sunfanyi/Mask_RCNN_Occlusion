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
from surgical_data import surgical

dataset_dir = '../../datasets/3dStool'
# lists_dir = os.path.join(dataset_dir, 'lists')
#
path = os.path.join(dataset_dir, 'test', 'manual_json', 'surgical_tool_test2020.json')
image_info = json.load(open(path))

config = surgical.SurgicalConfig()
# config.NUM_CLASSES -= 1
# config.__init__()

config.display()

# class_ids = list(range(1, 13))
class_ids = None
dataset_val = surgical.SurgicalDataset()
sur = dataset_val.load_surgical(dataset_dir, "val", return_surgical=True,
                             class_ids=class_ids)
dataset_val.prepare()

# dataset_train = surgical.SurgicalDataset()
# sur = dataset_train.load_surgical(dataset_dir, "train", return_surgical=True,
#                              class_ids=class_ids)
# dataset_train.prepare()

mask, class_ids = dataset_val.load_mask(1)
