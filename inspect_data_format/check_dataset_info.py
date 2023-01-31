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

dataset_dir = r'..\..\datasets\dataset_occluded'
lists_dir = os.path.join(dataset_dir, 'lists')

path = os.path.join(dataset_dir, 'annotations_occlusion_all.json')
image_info = json.load(open(path))
