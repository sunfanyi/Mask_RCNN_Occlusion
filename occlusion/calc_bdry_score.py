# -*- coding: utf-8 -*-
# @File    : calc_bdry_score.py
# @Time    : 25/02/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm


import os
import sys
import json
import numpy as np

sample_info = json.load(open('sample_mask_info.json', 'r'))

images = [i['image'] for i in sample_info]
images = np.array(images)
mask_true = [i['mask_gt'] for i in sample_info]
mask_true = np.array(mask_true)
mask_pred = [i['mask_pred'] for i in sample_info]
mask_pred = np.array(mask_pred)
