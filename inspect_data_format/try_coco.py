# -*- coding: utf-8 -*-
# @File    : try_coco.py
# @Time    : 30/01/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm


import os
import sys

ROOT_DIR = os.path.abspath("../")
COCO_DIR = os.path.abspath("../samples/coco")

sys.path.append(ROOT_DIR)
sys.path.append(COCO_DIR)

from coco import CocoDataset

# dataset_train = CocoDataset()
# dataset_train.load_coco("../../datasets/coco/", "train", year=2017)
# dataset_train.prepare()

dataset_val = CocoDataset()
coco = dataset_val.load_coco("../../datasets/coco/", "val", year=2017, return_coco=True)
dataset_val.prepare()
