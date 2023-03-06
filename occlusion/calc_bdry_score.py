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

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.utils import expand_mask

sample_info = json.load(open('sample_mask_info.json', 'r'))

ids = [i['id'] for i in sample_info]
file_names = [i['file_names'] for i in sample_info]
mask_true = [i['mask_true'] for i in sample_info]
mask_true = np.array(mask_true, dtype=np.bool)
mask_pred = [i['mask_pred'] for i in sample_info]
mask_pred = np.array(mask_pred, dtype=np.bool)


