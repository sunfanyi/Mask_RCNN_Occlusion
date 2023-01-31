# -*- coding: utf-8 -*-
# @File    : try_coco.py
# @Time    : 30/01/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm



from coco import CocoDataset

dataset_train = CocoDataset()
dataset_train.load_coco("../../../datasets/coco/", "train", year=2017)
dataset_train.prepare()
